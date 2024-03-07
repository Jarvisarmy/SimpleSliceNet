
from components import Projection, Discriminator, SelfAttentionModule, positionalencoding2d
import logging
import math
import torch.nn.functional as F
import tqdm
from utils import RescaleSegmentor, setup_result_path, TBWrapper, CosineWarmupScheduler
from features_utils import FeatureAggregator
from synthesize import simple_noise_on_features, coarse_noise_on_images
import metrics
import torch
import os
import numpy as np
from losses import get_logp, normal_fl_weighting, abnormal_fl_weighting, get_logp_boundary, calculate_bg_spp_loss,calculate_bg_spp_loss_normal
from cond_flow import conditional_flow_model
import time
log_theta = torch.nn.LogSigmoid() # log(1/(1+e^-x))
LOGGER = logging.getLogger(__name__)
class MRI_SimpleNet_2D(torch.nn.Module):
    def __init__(self,
                 backbone,
                 layers,
                 device,
                 preprocessing_dimension,
                 target_embed_dimension,
                 args):
        super(MRI_SimpleNet_2D, self).__init__()

        self.noise_std = args.noise_std
        self.mix_noise = args.mix_noise
        self.dsc_margin = args.dsc_margin

        self.args = args
        self.device = device
        self.layers = layers
        self.preprocessing_dimension = preprocessing_dimension
        self.target_embed_dimension = target_embed_dimension
        self.patch_size = args.patch_size

        self.featureExtractor = FeatureAggregator(
            backbone = backbone,
            layers = layers,
            volume_size = args.volume_size,
            patch_size = args.patch_size,
            stride = 1,
            output_dim = preprocessing_dimension,
            target_dim = target_embed_dimension,
            device=device
        ).to(self.device)


        # attention
        #self.use_attention = args.use_attention
        #if self.use_attention:
        #    self.attention = SelfAttentionModule(self.target_embed_dimension).to(device)

        #    self.attention_opt = torch.optim.Adam(self.attention.parameters(),1e-5)
        #    self.attention_scheduler = CosineWarmupScheduler(optimizer=self.attention_opt, warmup=100,max_iters=2000)

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
        # discriminator
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=args.dsc_layers,hidden=args.dsc_hidden).to(self.device)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(),lr=2*1e-4) # 2*1e-4
        self.disc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.disc_opt, 120, 2*1e-4*.4)

        # anomaly segmentor
        self.segmentor = RescaleSegmentor(self.device,target_size=list(args.volume_size))

        self.tensorboard_path, self.savedModels_path, self.log_path = setup_result_path(args.result_path,
                                                                                        args.model_dir,
                                                                                        args.sub_dir,
                                                                                        args.run_name)
        self.logger = TBWrapper(self.tensorboard_path)

    def train_epoch(self, epoch, train_loader):
        #if self.use_attention:
        #    self.attention.train()
        if self.fine_tune:
            self.featureExtractor.train()
        else:
            self.featureExtractor.eval()
        if self.projection:
            self.projection.train()
        self.discriminator.train()
        all_loss = []
        all_p_true = []
        all_p_fake = []
        all_std = []
        start_time = time.time()
        for data in train_loader:
            #if self.use_attention:
            #    self.attention_opt.zero_grad()
            if self.fine_tune:
                self.backbone_opt.zero_grad()
            if self.projection:
                self.proj_opt.zero_grad()
            self.disc_opt.zero_grad()
            images = data['image']
            images = images.to(self.device)
            B = images.size(0)
            if self.fine_tune:
                features, patch_shapes = self.featureExtractor(images)
            else:
                with torch.no_grad():
                    features, patch_shapes = self.featureExtractor(images) # [B*h*w, target_dim]
            h, w = patch_shapes[0]
            _, dim = features.shape
            if self.projection:
                true_feats = self.projection(features) # [B*h*w,target_dim]
            else:
                true_feats = features
            #if self.use_attention:
            #    h,w = patch_shapes[0]
            #    true_feats = true_feats.reshape(B,h, w,-1).permute(0,3,1,2) #[B,target_dim,h,w]
            #    true_feats = self.attention(true_feats) # [B, target_dim,h,w]
            #    true_feats = true_feats.permute(0,2,3,1).reshape(B*h*w,-1)
            if self.args.noise_type == "simple":
                noise = simple_noise_on_features(self.noise_std, self.mix_noise, true_feats)
            elif self.args.noise_type == 'coarse':
                noise = coarse_noise_on_images(self.args.noise_std, self.args.noise_res, true_feats.reshape(B,h,w,-1).permute(0,3,1,2))
                noise = noise.permute(0,2,3,1).reshape(B*h*w,-1)
            fake_feats = true_feats + noise
            if self.args.stop_gradient:
                fake_feats = fake_feats.detach()
            #pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).repeat(B,1,1,1) # [bs,dim, h, w]
            #pos_embed = pos_embed.permute(0,2,3,1).reshape(-1, 128)
            #true_feats = torch.cat((true_feats, pos_embed),dim=1)
            #fake_feats = torch.cat((fake_feats, pos_embed),dim=1)
            scores = self.discriminator(torch.cat([true_feats, fake_feats]))
            true_scores = scores[:len(true_feats)]
            fake_scores = scores[len(fake_feats):]
            th = self.dsc_margin
            p_true = (true_scores.detach() >=th).sum()/len(true_scores)
            p_fake = (fake_scores.detach() < -th).sum()/len(fake_scores)
            true_loss = torch.clip(-true_scores + th, min=0)
            fake_loss = torch.clip(fake_scores+th, min=0)
            loss = true_loss.mean() + fake_loss.mean()
            loss.backward()
            #if self.use_attention:
            #    self.attention_opt.step()
            #    self.attention_scheduler.step()
            if self.fine_tune:
                self.backbone_opt.step()
            if self.projection:
                self.proj_opt.step()
            self.disc_opt.step()
            self.disc_schl.step()
            #with torch.no_grad():
            #    std = F.normalize(true_feats.permute(1,0),dim=0).std(dim=1).mean()
            loss = loss.detach().cpu()
            self.logger.logger.add_scalar("p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar("p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            #self.logger.logger.add_scalar('std', std, self.logger.g_iter)
            self.logger.step()
            all_loss.append(loss.item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            #all_std.append(std.cpu().item())
        end_time = time.time()
        epoch_time = end_time - start_time
        #print('Time: {}'.format(epoch_time))
        return all_loss, all_p_true, all_p_fake, all_std, epoch_time
    
    def predict(self, data):
        """ Receive a batch of images
        Args: 
            images: [B, H, W]
        
        """
        
        data = data.to(self.device)
        batchsize = data.shape[0]
        #if self.use_attention:
        #    self.attention.eval()
        self.featureExtractor.eval()
        if self.projection:
            self.projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            embeded_features, patch_shapes = self.featureExtractor(data)
            h,w = patch_shapes[0]
            if self.projection:
                true_feats = self.projection(embeded_features) # [B*h*w,target_dim]
            else:
                true_feats = embeded_features
            _, dim = true_feats.shape
            #if self.use_attention:
                
            #    true_feats = true_feats.reshape(batchsize,h, w,-1).permute(0,3,1,2) #[B,target_dim,h,w]
            #    true_feats = self.attention(true_feats) # [B, target_dim,h,w]
            #    true_feats = true_feats.permute(0,2,3,1).reshape(batchsize*h*w,-1)
            #pos_embed = positionalencoding2d(128,h,w).to(self.device).unsqueeze(0).repeat(batchsize,1,1,1) # [bs,dim, h, w]
            #pos_embed = pos_embed.permute(0,2,3,1).reshape(-1, 128)


            #true_feats = torch.cat((true_feats, pos_embed),dim=1)
            patch_scores = image_scores = -self.discriminator(true_feats) #[B*h*w]
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy() #[B*h*w]
            patch_scores = patch_scores.reshape(batchsize,-1)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize,scales[0],scales[1])# [B,h,w]
            #features = embeded_features.reshape(batchsize,scales[0],scales[1],-1)# [B,h,w,target_dim]
            patch_scores = self.segmentor.convert_to_segmentation(patch_scores) #[B,H,W]
            patch_scores = patch_scores.reshape(batchsize,*patch_scores.shape[-2:])
            #patch_scores = patch_scores.permute(0,2,3,1) # [B,H,W]
            
            return patch_scores
    def train_model(self,train_loader,val_loader,epochs):
        LOGGER.info("Training...")
        best_loss = math.inf
        best_result = [0,0,0,0,0] # [I_AUROC, I_AUPRC, P_AUROC,_P_AUPRC]
        total_time = 0
        losses = []
        best_output = []
        best_gt = []
        I_auroc_list = []
        specificities = []
        with tqdm.tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_loss, all_p_true,all_p_fake,all_std, epoch_time = self.train_epoch(epoch, train_loader)

                all_loss = sum(all_loss)/len(train_loader)
                all_p_true = sum(all_p_true)/len(train_loader)
                all_p_fake = sum(all_p_fake)/len(train_loader)
                all_std = sum(all_std)/len(train_loader)
                losses.append(all_loss)
                total_time += epoch_time
                if all_loss < best_loss:
                    best_loss = all_loss
                    best_loss_model = {
                            'projection':self.projection.state_dict() if self.projection else None,
                            'discriminator':self.discriminator.state_dict(),
                            'backbone': self.featureExtractor.featureExtractor.backbone.state_dict(),
                            #'attention': self.attention.state_dict() if self.use_attention else None,
                            'loss': all_loss,
                            'epoch': epoch+1,
                            'p_true': all_p_true,
                            "p_fake": all_p_fake,
                            'args': self.args
                    }
                    torch.save(best_loss_model,os.path.join(self.savedModels_path,'best_loss_model.pth'))
                if (epoch+1)%10 == 0:
                    last_model = {
                            'projection':self.projection.state_dict() if self.projection else None,
                            'discriminator':self.discriminator.state_dict(),
                            'backbone': self.featureExtractor.featureExtractor.backbone.state_dict(),
                            #'attention': self.attention.state_dict() if self.use_attention else None,
                            'loss': all_loss,
                            'epoch': epoch+1,
                            'p_true': all_p_true,
                            "p_fake": all_p_fake,
                            'args': self.args
                    }
                    torch.save(last_model,os.path.join(self.savedModels_path,'last_model.pth'))
                    print('last model saved')
                pbar_str = "epoch: {}, loss: {}, p_true: {}, p_fake: {}, std: {}".format(epoch+1,all_loss,all_p_true,all_p_fake, all_std)
                
                if epoch == 0 or (epoch+1)%self.args.val_per_epochs == 0 or (epoch+1)==epochs:
                    results, output , gt= self.evaluate(val_loader)
                    I_auroc_list.append(results['I_auroc'])
                    specificities.append(results['specificity'])
                    output_str = ""
                    for key, value in results.items():
                        output_str += "{}: {}, ".format(key,value)
                        self.logger.logger.add_scalar(key, value, epoch)
                    if results['I_auroc'] > best_result[0]:
                        best_output = output
                        best_gt = gt
                        best_result[0] = results['I_auroc']
                        best_result[1] = results['I_auprc']
                        best_result[2] = results['p_auroc']
                        best_result[3] = results['p_auprc']
                        best_result[4] = results['specificity']
                    elif (results['I_auroc'] == best_result[0]) and (results['p_auroc'] > best_result[2]):
                        best_output = output
                        best_gt = gt
                        best_result[0] = results['I_auroc']
                        best_result[1] = results['I_auprc']
                        best_result[2] = results['p_auroc']
                        best_result[3] = results['p_auprc']
                        best_result[4] = results['specificity']
                    print(output_str)
                pbar.set_description_str(pbar_str)
                pbar.update(1)
        return best_result, losses, best_output, best_gt, I_auroc_list, specificities
    def evaluate(self, test_loader):
        LOGGER.info("Evaluation")
        #if self.use_attention:
        #    self.attention.eval()
        self.featureExtractor.eval()
        if self.projection:
            self.projection.eval()
        self.discriminator.eval()
        all_Iscores = []
        all_Igt = []
        all_scores = []
        all_gt = []
        with torch.no_grad():
            for data in test_loader:
                ground_truth = data['mask'].cpu().numpy() # [B,H,W]
                
                image_labels = data['label'].to(self.device)
                scores = self.predict(data['image']).cpu().numpy() #[B,H,W]
                ##ground_truth = ground_truth.cpu().numpy()
                image_labels = image_labels.cpu().numpy()
                #scores = scores.cpu().numpy()
                #scores = (scores-min_scores)/(max_scores-min_scores)
                Iscores = scores.reshape(len(scores),-1).max(axis=-1)
                for Iscore, image_label, score, label in zip(Iscores, image_labels, scores, ground_truth):
                    all_Iscores.append(Iscore)
                    all_Igt.append(image_label)
                    all_scores.append(score)
                    all_gt.append(label.squeeze(0))
                ##p_auroc = metrics.compute_auroc(scores,ground_truth)
                ##p_auprc = metrics.compute_auprc(scores,ground_truth)
                ##p_aurocs.append(p_auroc)
                ##p_auprcs.append(p_auprc)
            all_Iscores = np.array(all_Iscores)
            all_Igt = np.array(all_Igt)
            all_scores = np.array(all_scores)
            all_gt = np.array(all_gt)
            img_min_scores = all_Iscores.min(axis=-1)
            img_max_scores = all_Iscores.max(axis=-1)
            all_Iscores = (all_Iscores-img_min_scores)/(img_max_scores-img_min_scores)
            #min_scores = (
            #        scores.reshape(len(scores),-1).min(axis=-1).reshape(-1,1,1)
            #    )
            #max_scores = (
            #        scores.reshape(len(scores),-1).max(axis=-1).reshape(-1,1,1)
            #)
            #norm_scores = np.zeros_like(all_scores)
            #for min_score, max_score in zip(min_scores, max_scores):
            #    norm_scores += (all_scores-min_score)/max(max_score-min_score, 1e-2)
            #norm_scores = norm_scores/len(all_Iscores)
            min_score = all_scores.min()
            max_score = all_scores.max()
            norm_scores = (all_scores-min_score)/(max_score-min_score)
            I_auroc = metrics.auroc_score(all_Igt, all_Iscores)
            I_auprc = metrics.auprc_score(all_Igt, all_Iscores)
            p_auroc = metrics.auroc_score(all_gt, norm_scores)
            p_auprc = metrics.auprc_score(all_gt, norm_scores)
            pro = metrics.pro_score(all_gt, norm_scores)
            #pro = metrics.pro_score(all_gt, norm_scores)
            thres = metrics.return_best_thr(all_Igt, all_Iscores)
            original_p_thres = metrics.return_best_thr(all_gt,all_scores)
            p_thres = metrics.return_best_thr(all_gt, norm_scores)
            acc = metrics.accuracy_score(all_Igt, all_Iscores >= thres)
            f1 = metrics.f1_score(all_Igt, all_Iscores >= thres)
            recall = metrics.recall_score(all_Igt, all_Iscores >= thres)
            specificity = metrics.specificity_score(all_Igt, all_Iscores>=thres)
            ##p_auroc = sum(p_aurocs)/len(p_aurocs)
            ##p_auprc = sum(p_auprcs)/len(p_auprcs)
        return {
            "I_auroc": I_auroc,
            "I_auprc": I_auprc,
            "p_auroc": p_auroc,
            "p_auprc": p_auprc,
            "thres": thres,
            'original_p_thres':original_p_thres,
            'p_thres':p_thres,
            "acc": acc,
            "f1": f1,
            "recall": recall,
            "specificity": specificity,
            'pro':pro
        }, norm_scores, all_gt
    

def get_angles(pos, i, d_model):
    angle_rates = 1/torch.pow(10000, (2*(i//2))/d_model)
    return pos * angle_rates
def positional_encoding(d_model, height,width):
    pos_h = torch.arange(height).unsqueeze(1) # [height,1]
    pos_w = torch.arange(width).unsqueeze(1) # [width,1]
    angle_rads_h = get_angles(pos_h, torch.arange(d_model//2).unsqueeze(0),d_model)
    angle_rads_w = get_angles(pos_w, torch.arange(d_model //2).unsqueeze(0),d_model)

    pos_encoding_h = torch.zeros(height, d_model)
    pos_encoding_w = torch.zeros(width, d_model)
    pos_encoding_h[:,0::2] = torch.sin(angle_rads_h)
    pos_encoding_h[:,1::2] = torch.cos(angle_rads_h)
    pos_encoding_w[:,0::2] = torch.sin(angle_rads_w)
    pos_encoding_w[:,1::2] = torch.cos(angle_rads_w)
    pos_encoding = pos_encoding_h.unsqueeze(1)+pos_encoding_w.unsqueeze(0)
    pos_encoding = pos_encoding.unsqueeze(0).permute(0,3,1,2)
    return pos_encoding