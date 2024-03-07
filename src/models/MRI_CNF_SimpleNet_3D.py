from features_utils import FeatureAggregator
from components import Projection, Discriminator, positionalencoding2d
import logging
import math
import torch.nn.functional as F
import tqdm
from utils import RescaleSegmentor, setup_result_path, TBWrapper, CosineWarmupScheduler
from synthesize import simple_noise_on_features, coarse_noise_on_images
import metrics
import torch
import os
import numpy as np
from cond_flow import conditional_flow_model
import time
from losses import get_logp, normal_fl_weighting, abnormal_fl_weighting, get_logp_boundary, calculate_bg_spp_loss,calculate_bg_spp_loss_normal


log_theta = torch.nn.LogSigmoid() # log(1/(1+e^-x))
LOGGER = logging.getLogger(__name__)
class MRI_CNF_SimpleNet_3D(torch.nn.Module):
    def __init__(self,
                 backbone,
                 layers,
                 device,
                 preprocessing_dimension,
                 target_embed_dimension,
                 args):
        super(MRI_CNF_SimpleNet_3D, self).__init__()

        self.noise_std = args.noise_std
        self.mix_noise = args.mix_noise
        self.dsc_margin = args.dsc_margin

        self.args = args
        self.device = device
        self.layers = layers
        self.preprocessing_dimension = preprocessing_dimension
        self.target_embed_dimension = target_embed_dimension
        self.patch_size = args.patch_size
        self.volume_size = args.resize_size
        self.featureExtractor = FeatureAggregator(
            backbone = backbone,
            layers = layers,
            volume_size = args.resize_size,
            patch_size = args.patch_size,
            stride = 1,
            output_dim = preprocessing_dimension,
            target_dim = target_embed_dimension,
            device=device
        ).to(self.device)

        # Normalizing flow
        self.flow_model = conditional_flow_model(args,args.flow_hidden).to(device)
        #self.flow_model = conditional_flow_model(args,target_embed_dimension).to(device)
        self.flow_opt = torch.optim.Adam(self.flow_model.parameters(), lr=args.flow_lr)
        
        self.flow_schl = torch.optim.lr_scheduler.StepLR(self.flow_opt, step_size=args.flow_lr_step, gamma=0.9)
        # attention
        #self.use_attention = args.use_attention
        

        # backbone
        self.fine_tune = args.fine_tune
        if self.fine_tune:
            self.backbone_opt = torch.optim.AdamW(self.featureExtractor.featureExtractor.backbone.parameters(), 1e-3)

        # projection
        if args.proj_layers > 0:
            self.projection = Projection(self.target_embed_dimension, 
                                     self.target_embed_dimension,
                                     n_layers=args.proj_layers,
                                     layer_type=0).to(self.device)
            self.proj_opt = torch.optim.AdamW(self.projection.parameters(),lr=1e-4)
        else:
            self.projection = None

        # anomaly segmentor
        self.segmentor = RescaleSegmentor(self.device,target_size=list(self.volume_size[:2]))

        self.tensorboard_path, self.savedModels_path, self.log_path = setup_result_path(args.result_path,
                                                                                        args.model_dir,
                                                                                        args.sub_dir,
                                                                                        args.run_name)
        self.logger = TBWrapper(self.tensorboard_path)

    def train_epoch(self, epoch,train_loader):
        #if self.use_attention:
        #    self.attention.train()
        if self.fine_tune:
            self.featureExtractor.train()
        else:
            self.featureExtractor.eval()
        if self.projection:
            self.projection.train()
        self.flow_model.train()
        total_loss = 0
        loss_count = 0
        start_time = time.time()
        for data in train_loader:
            images = data['image']
            images = images.to(self.device)
            B,D,_,_,_ = images.shape
            if self.fine_tune:
                features, patch_shapes = self.featureExtractor(images)
            else:
                with torch.no_grad():
                    features, patch_shapes = self.featureExtractor(images) # [B*D*h*w, target_dim]
            h, w = patch_shapes[0]
            _,dim = features.shape
            if self.projection:
                true_feats = self.projection(features) # [B*D*h*w,target_dim]
            else:
                true_feats = features
            
            _,target_dim = true_feats.shape
            true_feats = true_feats.reshape(B,D,h,w,target_dim) #[B,D,h,w,target_dim]

            true_feats = true_feats.permute(0,2,3,4,1) # [B,h,w,target_dim,D]
            true_feats = true_feats.reshape(-1,*true_feats.shape[-2:]) #[B*h*w,target_dim,D]
            unfolder = torch.nn.Unfold(kernel_size=(1,5),stride=(1,1),padding=(0,2))
            true_feats = true_feats.unsqueeze(-1) # [B*h*w, target_dim, 1, D]
            true_feats = unfolder(true_feats) # [B*h*w, target_dim*3, D]
            true_feats = true_feats.permute(0,2,1)
            true_feats = true_feats.reshape(B*h*w*D,-1).unsqueeze(1)
            true_feats = F.adaptive_avg_pool1d(true_feats,self.args.flow_hidden).squeeze(1)
            
            
            noise = simple_noise_on_features(self.noise_std, self.mix_noise, true_feats)
            fake_feats = true_feats + noise
            
            pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).unsqueeze(0).repeat(2*B,D,1,1,1) # [2*bs,D,dim, h, w]
            pos_embed = pos_embed.permute(0,3,4,1,2).reshape(-1, 128) # [bs*h*w*D,128]
            N_batch = 4096

            perm = torch.randperm(2*B*h*w*D)
            num_N_batches = 2*B*h*w*D//N_batch
            e = torch.cat([true_feats, fake_feats])
            m = torch.cat([torch.zeros(len(true_feats)), torch.ones(len(fake_feats))]).to(self.device)
            for i in range(num_N_batches):
                
                idx = torch.arange(i*N_batch, (i+1)*N_batch)
                p_b = pos_embed[perm[idx]]
                e_b = e[perm[idx]]
                m_b = m[perm[idx]]
                z, log_jac_det = self.flow_model(e_b, [p_b,])
                if epoch == 0:
                    logps = get_logp(dim, z, log_jac_det)
                    logps = logps/dim

                    logps_detach = logps.detach()
                    normal_logps = logps_detach[m_b == 0]
                    anomaly_logps = logps_detach[m_b == 1]
                    nor_weights = normal_fl_weighting(normal_logps)
                    loss_ml = -log_theta(logps[m_b == 0]) * nor_weights

                    #loss_ml = -log_theta(logps[m_b==0])
                    loss = loss_ml.mean()
                elif m_b.sum() == 0:
                    logps = get_logp(dim, z, log_jac_det)
                    logps = logps/dim
                    normal_weights = normal_fl_weighting(logps.detach())
                    loss = -log_theta(logps)*normal_weights
                    loss = loss.mean()
                    # loss = -log_theta(logps[m_b==0]).mean()
                else:
                    logps = get_logp(dim, z, log_jac_det)
                    logps = logps/dim

                    logps_detach = logps.detach()
                    normal_logps = logps_detach[m_b == 0]
                    anomaly_logps = logps_detach[m_b == 1]
                    nor_weights = normal_fl_weighting(normal_logps)
                    ano_weights = abnormal_fl_weighting(anomaly_logps)
                    weights = nor_weights.new_zeros(logps_detach.shape)
                    weights[m_b == 0] = nor_weights
                    weights[m_b == 1] = ano_weights
                    loss_ml = -log_theta(logps[m_b == 0]) * nor_weights

                    #loss_ml = -log_theta(logps[m_b==0])
                    loss_ml = loss_ml.mean()
                    boundaries = get_logp_boundary(logps, m_b, self.args.pos_beta, self.args.margin_tau, self.args.normalizer)
                    loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, m_b, boundaries,self.args.normalizer,weights=weights)
                    loss = loss_ml + (loss_n_con+loss_a_con)

                if self.fine_tune:
                    self.backbone_opt.zero_grad()
                if self.projection:
                    self.proj_opt.zero_grad()
                self.flow_opt.zero_grad()
                loss.backward(retain_graph=True)
                #loss.backward()
                if self.fine_tune:
                    self.backbone_opt.step()
                if self.projection:
                    self.proj_opt.step()
                self.flow_opt.step()
                #self.flow_schl.step()
                if math.isnan(loss.item()):
                    total_loss += 0.0
                    loss_count += 0
                else:
                    total_loss += loss.item()
                    loss_count += 1
        end_time = time.time()
        #print('Time: {}'.format(end_time-start_time))
        epoch_time = end_time - start_time
        all_loss = total_loss/(loss_count+1e-8)
        
        return all_loss, epoch_time
    
    def predict(self, data):
        """ Receive a batch of images
        Args: 
            images: [B, H, W]
        
        """
        
        data = data.to(self.device) # [B,D,3,H,W]
        B,D,_,_,_ = data.shape
        #if self.use_attention:
        #    self.attention.eval()
        self.featureExtractor.eval()
        if self.projection:
            self.projection.eval()
        self.flow_model.eval()
        with torch.no_grad():
            embeded_features, patch_shapes = self.featureExtractor(data)
            h, w = patch_shapes[0]
            if self.projection:
                true_feats = self.projection(embeded_features) # [B*D*h*w,target_dim]
            else:
                true_feats = embeded_features
            
            _,target_dim = true_feats.shape
            assert(true_feats.shape == (B*D*h*w,target_dim))
            true_feats = true_feats.reshape(B,D,h,w,target_dim) #[B,D,h,w,target_dim]

            true_feats = true_feats.permute(0,2,3,4,1) # [B,h,w,target_dim,D]
            true_feats = true_feats.reshape(-1,*true_feats.shape[-2:]) #[B*h*w,target_dim,D]
            
            unfolder = torch.nn.Unfold(kernel_size=(1,5),stride=(1,1),padding=(0,2))
            true_feats = true_feats.unsqueeze(-1) # [B*h*w, target_dim, 1, D]
            true_feats = unfolder(true_feats) # [B*h*w, target_dim*3, D]
            #assert(true_feats.shape == (B*h*w,target_dim*3,D))
            true_feats = true_feats.permute(0,2,1) 
            true_feats = true_feats.reshape(B*h*w*D,-1).unsqueeze(1)
            true_feats = F.adaptive_avg_pool1d(true_feats,self.args.flow_hidden).squeeze(1)
            
            pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).unsqueeze(0).repeat(B,D,1,1,1) # [bs,D,dim, h, w]
            pos_embed = pos_embed.permute(0,3,4,1,2).reshape(-1, 128) # [bs*h*w*D,dim]

            z, log_jac_det = self.flow_model(true_feats, [pos_embed,])
            dim = target_dim
            logps = get_logp(dim, z, log_jac_det)
            logps = logps/dim

            logps = logps
            scales = patch_shapes[0]
            logps = logps.reshape(B,scales[0],scales[1],D)# [B,h,w]
            logps = logps.permute(0,3,1,2).reshape(B*D,scales[0],scales[1])
            
            # new version
            logps = self.segmentor.convert_to_segmentation(logps)
            logps = logps.reshape(B,D,*logps.shape[-2:])
            logps = logps.permute(0,2,3,1) # [B,H,W,D]
            return logps
            '''
            #features = embeded_features.reshape(batchsize,scales[0],scales[1],-1)# [B,h,w,target_dim]
            logps-=torch.max(logps) # normalize log-likelilhood to (-Inf: 0] by subtracting a constant
            probs = torch.exp(logps) # conver to probs in range[0:1]
            patch_scores = self.segmentor.convert_to_segmentation(probs) #[B*D,H,W]
            patch_scores = torch.max(patch_scores) - patch_scores # reverse it to anomaly score
            patch_scores = patch_scores.reshape(B,D,*patch_scores.shape[-2:])
            patch_scores = patch_scores.permute(0,2,3,1) # [B,H,W,D]
            return patch_scores
            '''
    def train_model(self,train_loader,val_loader,epochs):
        LOGGER.info("Training...")
        best_loss = math.inf
        best_result = [0,0,0,0] # [AUROC, AUPRC, s_AUROC,_s_AUPRC]
        total_time = 0
        losses = []
        best_output = []
        best_gt = []
        slice_auroc_list = []
        with tqdm.tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                loss, epoch_time = self.train_epoch(epoch,train_loader)
                losses.append(loss)
                total_time += epoch_time
                if (epoch+1)%10 == 0:
                    last_model = {
                        'projection':self.projection.state_dict() if self.projection else None,
                        'backbone': self.featureExtractor.featureExtractor.backbone.state_dict(),
                        'flow_model': self.flow_model.state_dict(),
                        'loss': loss,
                        'epoch': epoch+1,
                        'args': self.args
                    }
                    torch.save(last_model,os.path.join(self.savedModels_path,'last_model.pth'))
                pbar_str = "epoch: {}, loss: {}".format(epoch+1,loss)

                if epoch == 0 or (epoch+1)%self.args.val_per_epochs == 0 or (epoch+1)==epochs:
                    results, all_scores, all_gt= self.evaluate(val_loader)
                    output_str = ""
                    for key, value in results.items():
                        output_str += "{}: {}, ".format(key,value)
                        self.logger.logger.add_scalar(key, value, epoch)
                    
                    print(output_str)
                pbar.set_description_str(pbar_str)
                pbar.update(1)
        return all_scores, all_gt
    '''
    def evaluate(self, test_loader):
        LOGGER.info("Evaluation")
        #if self.use_attention:
        #    self.attention.eval()
        self.featureExtractor.eval()
        if self.projection:
            self.projection.eval()
        self.flow_model.eval()
        all_slice_scores = []
        all_slice_gt = []
        all_scores = []
        all_gt = []
        with torch.no_grad():
            for data in test_loader:
                ground_truth = data['mask'].cpu().numpy() # [B,H,W,D]
                B,H,W,D = ground_truth.shape
                scores = self.predict(data['image']).cpu().numpy() #[B,H,W,D]
                #min_score = np.min(scores,axis=(1,2,3),keepdims=True)
                #max_score = np.max(scores, axis=(1,2,3),keepdims=True)
                #scores = (scores-min_score)/(max_score-min_score)
                slice_scores = np.sum(scores, axis=(1,2))
                slice_gts = np.amax(ground_truth, axis=(1,2))
                slice_scores = (slice_scores-slice_scores.min())/(slice_scores.max()-slice_scores.min())
                slice_scores = 1-slice_scores
                slice_gts = 1-slice_gts
                for gt, score, slice_gt, slice_score in zip(ground_truth, scores, slice_gts, slice_scores):
                    all_gt.append(gt)
                    all_scores.append(score)
                    all_slice_gt.append(slice_gt)
                    all_slice_scores.append(slice_score)
                    #all_volume_gt.append(volume_gt)
                    #all_volume_scores.append(volume_score)
            
            #all_gt = np.array(all_gt)
            #all_scores = np.array(all_scores)
            all_slice_gt = np.array(all_slice_gt)
            all_slice_scores = np.array(all_slice_scores)
            #min_slice_score = all_slice_scores.min()
            #
            #max_slice_score = all_slice_scores.max()
            #all_slice_scores = (all_slice_scores-min_slice_score)/(max_slice_score-min_slice_score)
            
            #min_score = all_scores.min()
            #max_score = all_scores.max()
            #all_scores = (all_scores-min_score)/(max_score-min_score)
            #all_scores = 1-all_scores
            #all_gt = 1-all_gt
            #all_volume_gt = np.array(all_volume_gt)
            #all_volume_scores = np.array(all_volume_scores)

            #auroc = metrics.auroc_score(all_gt, all_scores)
            #auprc = metrics.auprc_score(all_gt, all_scores)
            auroc = 0
            auprc = 0
            slice_auroc = metrics.auroc_score(all_slice_gt, all_slice_scores)
            slice_auprc = metrics.auprc_score(all_slice_gt, all_slice_scores)

        return {'auroc':auroc, 
                'auprc':auprc, 
                'slice_auroc':slice_auroc, 
                'slice_auprc': slice_auprc}, all_scores, all_gt
    '''
    def evaluate(self, test_loader):
        LOGGER.info("Evaluation")
        #if self.use_attention:
        #    self.attention.eval()
        self.featureExtractor.eval()
        if self.projection:
            self.projection.eval()
        self.flow_model.eval()
        all_logps = []
        all_gt = []
        with torch.no_grad():
            for data in test_loader:
                ground_truth = data['mask'].cpu().numpy() # [B,H,W,D]
                B,H,W,D = ground_truth.shape
                logps = self.predict(data['image']).cpu().numpy() #[B,H,W,D]
                for logp,gt in zip(logps, ground_truth):
                    all_logps.append(logp)
                    all_gt.append(gt)

            # all_logps = [#,H,W,D]
            all_logps = np.array(all_logps)
            all_gt = np.array(all_gt)
            all_logps -= np.max(all_logps) # normalize log-likelilhood to (-Inf: 0] by subtracting a constant
            scores = np.exp(all_logps) # conver to probs in range[0:1]
            scores = np.max(scores)-scores # reverse it to anomaly score

            slice_scores = np.sum(scores, axis=(1,2))
            slice_gts = np.amax(all_gt, axis=(1,2))
            slice_scores = (slice_scores-slice_scores.min())/(slice_scores.max()-slice_scores.min())
            slice_scores = 1-slice_scores
            slice_gts = 1-slice_gts
                
   
            auroc = 0
            auprc = 0
            slice_auroc = metrics.auroc_score(slice_gts, slice_scores)
            slice_auprc = metrics.auprc_score(slice_gts, slice_scores)

        return {'auroc':auroc, 
                'auprc':auprc, 
                'slice_auroc':slice_auroc, 
                'slice_auprc': slice_auprc}, scores, all_gt
            