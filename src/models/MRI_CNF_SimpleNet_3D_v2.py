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
import scipy.ndimage as ndimage

log_theta = torch.nn.LogSigmoid() # log(1/(1+e^-x))
LOGGER = logging.getLogger(__name__)


class RescaleSegmentor_3D:
    def __init__(self, device, target_size=[192,192,96]):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1) # [bs,1,h,w,D]
            _scores = F.interpolate(
                _scores, size=self.target_size,mode='trilinear',align_corners=False
            )
            _scores = _scores.squeeze(1)
            _scores = _scores.cpu().numpy()
            return  torch.from_numpy(np.stack([
                [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in slice]
                for slice in _scores])).to(self.device)

class MRI_CNF_SimpleNet_3D_v2(torch.nn.Module):
    def __init__(self,
                 backbone,
                 layers,
                 device,
                 preprocessing_dimension,
                 target_embed_dimension,
                 args):
        super(MRI_CNF_SimpleNet_3D_v2, self).__init__()

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
        self.flow_model = conditional_flow_model(args,args.target_embed_dimension).to(device)
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
        self.segmentor = RescaleSegmentor_3D(self.device,target_size=list(args.resize_size))

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
            true_feats = true_feats.reshape(B*h*w,target_dim,D)
            true_feats = F.avg_pool1d(true_feats,kernel_size=3) #[B*h*w,target_dim,d]
            d = true_feats.shape[-1]
            true_feats = true_feats.permute(0,2,1)
            true_feats = true_feats.reshape(B*h*w*d,target_dim)
            
            
            
            noise = simple_noise_on_features(self.noise_std, self.mix_noise, true_feats)
            fake_feats = true_feats + noise
            
            pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).unsqueeze(0).repeat(2*B,d,1,1,1) # [2*bs,D,dim, h, w]
            pos_embed = pos_embed.permute(0,3,4,1,2).reshape(-1, 128) # [bs*h*w*D,128]
            N_batch = 4096

            perm = torch.randperm(2*B*h*w*d)
            num_N_batches = 2*B*h*w*d//N_batch
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
            true_feats = true_feats.reshape(B,D,h,w,target_dim) #[B,D,h,w,target_dim]

            true_feats = true_feats.permute(0,2,3,4,1) # [B,h,w,target_dim,D]
            true_feats = true_feats.reshape(B*h*w,target_dim,D)
            true_feats = F.avg_pool1d(true_feats,kernel_size=3) #[B*h*w,target_dim,d]
            d = true_feats.shape[-1]
            true_feats = true_feats.permute(0,2,1)
            true_feats = true_feats.reshape(B*h*w*d,target_dim)
            
            pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).unsqueeze(0).repeat(B,d,1,1,1) # [bs,D,dim, h, w]
            pos_embed = pos_embed.permute(0,3,4,1,2).reshape(-1, 128) # [bs*h*w*D,dim]

            z, log_jac_det = self.flow_model(true_feats, [pos_embed,])
            dim = target_dim
            logps = get_logp(dim, z, log_jac_det)
            logps = logps/dim

            logps = logps
            scales = patch_shapes[0]
            logps = logps.reshape(B,scales[0],scales[1],d)# [B,h,w,d]
            #logps = logps.permute(0,3,1,2).reshape(B*d,scales[0],scales[1])
            # new version
            logps = self.segmentor.convert_to_segmentation(logps)
            #logps = logps.reshape(B,D,*logps.shape[-2:])
            #logps = logps.permute(0,2,3,1) # [B,H,W,D]
            return logps
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
                    results, images,all_scores, all_gt= self.evaluate(val_loader)
                    output_str = ""
                    for key, value in results.items():
                        output_str += "{}: {}, ".format(key,value)
                        self.logger.logger.add_scalar(key, value, epoch)
                    
                    print(output_str)
                pbar.set_description_str(pbar_str)
                pbar.update(1)
        return images,all_scores, all_gt
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
        images = []
        with torch.no_grad():
            for data in test_loader:
                ground_truth = data['mask'].cpu().numpy() # [B,H,W,D]
                B,H,W,D = ground_truth.shape
                logps = self.predict(data['image']).cpu().numpy() #[B,H,W,D]

                imgs = data['original'].cpu().numpy()
                for img, logp,gt in zip(imgs,logps, ground_truth):
                    images.append(img)
                    all_logps.append(logp)
                    all_gt.append(gt)

            # all_logps = [#,H,W,D]
            all_logps = np.array(all_logps)
            all_gt = np.array(all_gt)
            all_logps -= np.max(all_logps) # normalize log-likelilhood to (-Inf: 0] by subtracting a constant
            scores = np.exp(all_logps) # conver to probs in range[0:1]
            scores = np.max(scores)-scores # reverse it to anomaly score
            scores = (scores - scores.min())/(scores.max()-scores.min())
            
            slice_scores = np.sum(scores, axis=(1,2))
            slice_gts = np.amax(all_gt, axis=(1,2))
            
            slice_min = slice_scores.min(axis=-1,keepdims=True)
            slice_max = slice_scores.max(axis=-1,keepdims=True)
            #slice_min = slice_scores.min()
            #slice_max = slice_scores.max()
            slice_scores = (slice_scores-slice_min)/(slice_max-slice_min)
            #slice_scores = (slice_scores-slice_scores.min())/(slice_scores.max()-slice_scores.min())
            #slice_scores = 1-slice_scores
            #slice_gts = 1-slice_gts
            
            #pix_min = scores.min(axis=(1,2,3),keepdims=True)
            #pix_max = scores.max(axis=(1,2,3),keepdims=True)
            #scores = (scores - pix_min)/(pix_max-pix_min)
            #scores = (scores-scores.min())/(scores.max()-scores.min())
            #scores = 1-scores
            #all_gt = 1-all_gt
   
            #auroc = metrics.auroc_score(all_gt,scores)
            #auprc = metrics.auprc_score(all_gt, scores)
            #auroc = 0
            #auprc = 0
            #slice_auroc = metrics.auroc_score(slice_gts, slice_scores)
            #slice_auprc = metrics.auprc_score(slice_gts, slice_scores)

            slice_gts = slice_gts.ravel()
            slice_scores = slice_scores.ravel()
            thres = metrics.return_best_thr(slice_gts, slice_scores)
            #acc = metrics.accuracy_score(slice_gts, slice_scores>=thres)
            #f1 = metrics.f1_score(slice_gts,slice_scores>=thres)
            
            p_thres = metrics.return_best_thr(all_gt, scores)
            
            #pro = metrics.pro_score(all_gt.transpose(0,3,1,2).reshape(-1,*all_gt.shape[1:3]),scores.transpose(0,3,1,2).reshape(-1,*scores.shape[1:3]))
            auroc = 0
            auprc = 0
            slice_auroc = 0
            slice_auprc = 0
            acc, f1, pro = 0,0,0
        return {'auroc':auroc, 
                'auprc':auprc, 
                'slice_auroc':slice_auroc, 
                'slice_auprc': slice_auprc,
                'p_thres':p_thres,
                'thres': thres,
                'acc':acc,
                'f1':f1,
                'pro':pro}, images,scores, all_gt