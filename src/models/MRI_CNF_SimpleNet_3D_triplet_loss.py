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
            #return  torch.from_numpy(np.stack([
            #    [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in slice]
            #    for slice in _scores])).to(self.device)
            return torch.from_numpy(ndimage.gaussian_filter(_scores, sigma=self.smoothing)).to(self.device)
class MRI_CNF_SimpleNet_3D_triplet_loss(torch.nn.Module):
    def __init__(self,
                 backbone,
                 layers,
                 device,
                 preprocessing_dimension,
                 target_embed_dimension,
                 args):
        super(MRI_CNF_SimpleNet_3D_triplet_loss, self).__init__()

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
        for i,data in tqdm.tqdm(enumerate(train_loader)):
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
            
            pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).unsqueeze(0).repeat(B,d,1,1,1) # [2*bs,D,dim, h, w]
            pos_embed = pos_embed.permute(0,3,4,1,2).reshape(-1, 128) # [bs*h*w*D,128]
            N_batch = 4096

            perm = torch.randperm(B*h*w*d)
            num_N_batches = B*h*w*d//N_batch
            for i in range(num_N_batches):
                idx = torch.arange(i*N_batch,(i+1)*N_batch)
                p_b = pos_embed[perm[idx]]
                true_e_b = true_feats[perm[idx]]
                fake_e_b = fake_feats[perm[idx]]

                true_z, true_log_jac_det = self.flow_model(true_e_b,[p_b])
                true_logps = get_logp(dim,true_z,true_log_jac_det)
                true_logps = true_logps/dim
                loss_ml = -log_theta(true_logps)
                loss_ml = loss_ml.mean()

                fake_z, fake_log_jac_det = self.flow_model(fake_e_b,[p_b])
                fake_logps = get_logp(dim,fake_z,fake_log_jac_det)
                fake_logps = fake_logps/dim
            
                positive_indices = torch.randperm(true_logps.size(0))
                positives = true_logps[positive_indices]

                margin=1.0
                triplet_loss = torch.nn.TripletMarginLoss(margin=margin,p=2)
                loss_an = triplet_loss(true_logps,positives,fake_logps)
                loss = loss_ml + loss_an
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
        best_result = [0,0,0,0] # [I_AUROC, I_AUPRC, P_AUROC,_P_AUPRC]
        total_time = 0
        losses = []
        best_output = []
        best_gt = []
        slice_auroc_list = []
        run_times = []
        with tqdm.tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                start_time = time.time()
                loss, epoch_time = self.train_epoch(epoch,train_loader)
                epoch_duration = time.time() - start_time
                print("Time for epoch {}: {}".format(epoch,epoch_duration))
                run_times.append(epoch_duration)
                losses.append(loss)
                total_time += epoch_time
                if loss < best_loss:
                    best_loss = loss
                    best_loss_model = {
                        'projection':self.projection.state_dict() if self.projection else None,
                        'backbone': self.featureExtractor.featureExtractor.backbone.state_dict(),
                        'flow_model': self.flow_model.state_dict(),
                        'loss': loss,
                        'epoch': epoch+1,
                        'args': self.args
                    }
                    torch.save(best_loss_model,os.path.join(self.savedModels_path,'best_loss_model.pth'))
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

                if (epoch+1)%self.args.val_per_epochs == 0:
                    results= self.evaluate(val_loader)
                    output_str = ""
                    for key, value in results.items():
                        output_str += "{}: {}, ".format(key,value)
                        self.logger.logger.add_scalar(key, value, epoch)
                    if results['I_auroc'] > best_result[0]:
                        best_result[0] = results['I_auroc']
                        best_result[1] = results['I_auprc']
                        best_result[2] = results['p_auroc']
                        best_result[3] = results['p_auprc']
                        best_model = {
                        'projection':self.projection.state_dict() if self.projection else None,
                        'backbone': self.featureExtractor.featureExtractor.backbone.state_dict(),
                        'flow_model': self.flow_model.state_dict(),
                        'loss': loss,
                        'epoch': epoch+1,
                        'args': self.args
                        }
                        torch.save(best_model,os.path.join(self.savedModels_path,'best_val_model.pth'))
                    elif (results['I_auroc'] == best_result[0]) and (results['p_auroc'] > best_result[2]):
                        best_result[0] = results['I_auroc']
                        best_result[1] = results['I_auprc']
                        best_result[2] = results['p_auroc']
                        best_result[3] = results['p_auprc']
                        best_model = {
                        'projection':self.projection.state_dict() if self.projection else None,
                        'backbone': self.featureExtractor.featureExtractor.backbone.state_dict(),
                        'flow_model': self.flow_model.state_dict(),
                        'loss': loss,
                        'epoch': epoch+1,
                        'args': self.args
                        }
                        torch.save(best_model,os.path.join(self.savedModels_path,'best_val_model.pth'))
                    print(output_str)
                pbar.set_description_str(pbar_str)
                pbar.update(1)
        print(np.mean(run_times),np.std(run_times))
    def calculate(self,all_logps, all_gt):
        
        all_logps = np.array(all_logps)
        all_gt = np.array(all_gt)
        all_logps -= np.max(all_logps) # normalize log-likelilhood to (-Inf: 0] by subtracting a constant
        scores = np.exp(all_logps) # conver to probs in range[0:1]
        scores = np.max(scores)-scores # reverse it to anomaly score
        scores = (scores - scores.min())/(scores.max()-scores.min())
        total, H, W, D = all_logps.shape
        assert(all_logps.shape[1:]==self.volume_size)
        assert(scores.shape ==(total,H,W,D))
        slice_scores = np.amax(scores, axis=(1,2))
        slice_gts = np.amax(all_gt, axis=(1,2))
        assert(slice_gts.shape == (total,D))
        slice_min = slice_scores.min(axis=-1,keepdims=True)
        slice_max = slice_scores.max(axis=-1,keepdims=True)
        assert(slice_min.shape == (total,1))
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
        
        auroc = metrics.auroc_score(all_gt,scores)
        #auprc = metrics.auprc_score(all_gt, scores)
        #auroc = 0
        #auprc = 0
        slice_auroc = metrics.auroc_score(slice_gts, slice_scores)
        #slice_auprc = metrics.auprc_score(slice_gts, slice_scores)

        #slice_gts = slice_gts.ravel()
        #slice_scores = slice_scores.ravel()
        #thres = metrics.return_best_thr(slice_gts, slice_scores)
        #acc = metrics.accuracy_score(slice_gts, slice_scores>=thres)
        #f1 = metrics.f1_score(slice_gts,slice_scores>=thres)
                    
        #p_thres = metrics.return_best_thr(all_gt, scores)
                    
        #pro = metrics.pro_score(all_gt.transpose(0,3,1,2).reshape(-1,*all_gt.shape[1:3]),scores.transpose(0,3,1,2).reshape(-1,*scores.shape[1:3]))
        #auroc = 0
        #auprc = 0
        #slice_auroc = 0
        #slice_auprc = 0
        #acc, f1, pro = 0,0,0
        return auroc, slice_auroc
    def evaluate(self, test_loader):
        LOGGER.info("Evaluation")
        #if self.use_attention:
        #    self.attention.eval()
        self.eval()
        self.featureExtractor.eval()
        if self.projection:
            self.projection.eval()
        self.flow_model.eval()
        gt_list_px = []
        pr_list_px = []
        gt_list_sp = []
        pr_list_sp = []
        aupro_list = []
        auroc_list = []
        auprc_list = []
        iou_list = []
        specificity_list = []
        sensitivity_list = []
        precision_list = []
        dice_list = []
        acc_list = []
        f1_list = []
        dice_list = []
        max_value = -1000
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(test_loader)):

                ground_truth = data['mask'].cpu().numpy() # [B,H,W,D]
                B,H,W,D = ground_truth.shape
                logps = self.predict(data['image']).cpu().numpy() #[B,H,W,D]
                gts = ground_truth.squeeze().astype(bool)
                prs = logps.squeeze()
                prs = prs.transpose(2,0,1)
                gts = gts.transpose(2,0,1)
                prs -= np.max(prs)
                prs = np.exp(prs)
                prs = np.max(prs)-prs
                pr_list_sp.extend(np.amax(prs,axis=(1,2)).tolist())
                gt_list_sp.extend(np.amax(gts,axis=(1,2)).tolist())
                prs = (prs-prs.min())/(prs.max()-prs.min())
                #aupro_list.append(metrics.pro_score(gts,prs))
                auroc_list.append(metrics.auroc_score(gts,prs))
                auprc_list.append(metrics.auprc_score(gts,prs))

                #thres = metrics.return_best_thr(gts,prs)
                #iou_list.append(metrics.IoU_score(gts, prs>=thres))
                #gts = gts.ravel()
                #prs = prs.ravel()
                #specificity_list.append(metrics.specificity_score(gts,prs>=thres))
                #sensitivity_list.append(metrics.recall_score(gts,prs>=thres))
                
                #precision_list.append(metrics.precision_score(gts,prs>=thres))
                #acc_list.append(metrics.accuracy_score(gts,prs>=thres))
                #f1_list.append(metrics.f1_score(gts,prs>=thres))
                #dice_list.append(metrics.calculate_maximum_dice(prs>=thres,gts))
            pr_sp = np.array(pr_list_sp)
            gt_sp = np.array(gt_list_sp)
            pr_sp = (pr_sp-pr_sp.min())/(pr_sp.max()-pr_sp.min())
            Iauroc = metrics.auroc_score(gt_sp,pr_sp)
            Iauprc = metrics.auprc_score(gt_sp,pr_sp)
            #Ithres = metrics.return_best_thr(gt_sp,pr_sp)
            #Ispecificity = metrics.specificity_score(gt_sp,pr_sp>=Ithres)
            #Isensitivity = metrics.recall_score(gt_sp, pr_sp>=Ithres)
            #Iprecision = metrics.precision_score(gt_sp,pr_sp>=Ithres)
            #Iacc = metrics.accuracy_score(gt_sp,pr_sp>=Ithres)
            #If1 = metrics.f1_score(gt_sp,pr_sp>=Ithres)
            #Idice = metrics.calculate_maximum_dice(pr_sp>=Ithres,gt_sp)
        result = {
                'p_auroc': round(np.mean(auroc_list),4),
                'p_auprc':round(np.mean(auprc_list),4),
                #'p_aupro':round(np.mean(aupro_list),4),
                #'p_dice': round(np.mean(dice_list),4),
                #'p_specificity': round(np.mean(specificity_list),4),
                #'p_sensitivity': round(np.mean(sensitivity_list),4),
                #'p_iou': round(np.mean(iou_list), 4),
                #'p_precision':round(np.mean(precision_list),4),
                #'p_acc': round(np.mean(acc_list),4),
                #'p_f1': round(np.mean(f1_list),4),
                'I_auroc': round(Iauroc,4),
                'I_auprc': round(Iauprc,4),
                #'i_dice': round(Idice,4),
                #'i_specificity': round(Ispecificity,4),
                #'i_sensitivity': round(Isensitivity, 4),
                #'i_precision': round(Iprecision,4),
                #'i_acc': round(Iacc,4),
                #'i_f1': round(If1,4),
            }

        for item, value in result.items():
            print('{}: {}\n'.format(item, value))
        return result
