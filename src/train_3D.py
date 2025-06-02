import argparse
import ast
import torch
from torch.utils.data import DataLoader,Dataset
import os
from dataloaders import dataloader_IXI_BraTS_setup
from utils import  load_backbone
from models import MRI_CNF_SimpleNet_3D, MRI_CNF_SimpleNet_3D_v2,MRI_CNF_SimpleNet_3D_v3,MRI_SimpleNet_3D,MRI_CNF_SimpleNet_3D_triplet_loss,MRI_CNF_SimpleNet_3D_pair_loss
from runners.Br35H_studies import study_architecture
from models.MRI_CNF_SimpleNet_3D_triplet_loss import MRI_CNF_SimpleNet_3D_triplet_loss
import metrics
from models.MRI_CNF_SimpleNet_3D_pair_loss import MRI_CNF_SimpleNet_3D_pair_loss

import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import json
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

import nibabel as nib
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def crop_and_pad_volume_to_size(volume, target_size=(192,192,96)):
    """
    Crop and/or zero-pad a 3D volume to match the target size.

    Parameters:
        volume (Tensor or ndarray): Input 3D volume, shape [H, W, D]
        target_size (tuple): Desired output size (H_target, W_target, D_target)

    Returns:
        torch.Tensor: Volume with shape exactly equal to target_size
    """
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)
    elif isinstance(volume, list):
        volume = torch.tensor(volume)
    elif not isinstance(volume, torch.Tensor):
        raise TypeError("Input must be a numpy array, list, or torch tensor.")

    input_shape = volume.shape
    output = torch.zeros(target_size, dtype=volume.dtype, device=volume.device)

    input_slices = []
    output_slices = []

    for i in range(3):
        in_dim = input_shape[i]
        out_dim = target_size[i]

        if in_dim >= out_dim:
            start_in = (in_dim - out_dim) // 2
            end_in = start_in + out_dim
            input_slices.append(slice(start_in, end_in))
            output_slices.append(slice(0, out_dim))
        else:
            start_out = (out_dim - in_dim) // 2
            end_out = start_out + in_dim
            input_slices.append(slice(0, in_dim))
            output_slices.append(slice(start_out, end_out))

    # Use advanced slicing for tensors
    output[output_slices[0], output_slices[1], output_slices[2]] = \
        volume[input_slices[0], input_slices[1], input_slices[2]]
    
    return output
def normalize_3d_volume_to_imagenet(volume):
    """
    Args:
      volume [H,W,D]
    """
    H,W,D = volume.shape
    volume = (volume - volume.min())/(volume.max()-volume.min())
    volume = volume.permute(2,0,1).unsqueeze(1).repeat(1,3,1,1) # [D,3,H,W]
    for d in range(D):
        volume[d,:,:,:] = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(volume[d,:,:,:]).float()
    return volume
class MRIDataset_3D(Dataset):
  def __init__(self, data_list,
               mask_list=None,
               volume_size =(192,192,96),
               resize_size=(192,192,96)):
    super().__init__()
    self.data_list = data_list
    self.mask_list = mask_list
    self.resize_size=resize_size
    self.volume_size = volume_size

  def __len__(self):
    return len(self.data_list)
  def __getitem__(self, idx):
    # get the normalized volume
    volume_path = self.data_list[idx]
    volume = torch.from_numpy(nib.load(volume_path).get_fdata())
    volume = volume[:,:,24:120]
    cropped_volume= crop_and_pad_volume_to_size(volume, target_size=(192,192,96))
    #normalized_volume = normalize_3d_volume(cropped_volume)

    # get the cropped mask

    shape = cropped_volume.shape #[H,W,D]
    if self.mask_list and self.mask_list[idx]:
      mask_path = self.mask_list[idx]
      mask = torch.from_numpy(nib.load(mask_path).get_fdata())
      mask = mask[:,:,24:120]
      cropped_mask= crop_and_pad_volume_to_size(mask, target_size=(192,192,96))
      if cropped_mask.max() ==0:
         print(volume_path)
      cropped_mask = (cropped_mask != 0).int()
    else:
      cropped_mask = torch.zeros(shape)
    label = torch.max(cropped_mask) > 0
    # apply transform to normalized_volume and cropped_mask

    return {'image': normalize_3d_volume_to_imagenet(cropped_volume).float(), # [D,3,H,W]
            'original': cropped_volume.float(), # [H,W,D]
            'label': label,
            'mask': cropped_mask} # [H,W,D]
def dataloader_IXI_BraTS_setup(data_root, resize_size,volume_size,batch_size):
    train_data = []
    for file in os.listdir(os.path.join(data_root,'IXI_registered')):
        if file.startswith('IXI'):
            train_data.append(os.path.join(data_root,'IXI_registered', file))

    test_data = []
    test_mask = []
    for file in os.listdir(os.path.join(data_root,'BraTS_registered')):
        if file.startswith('BraTS2021'):
            test_data.append(os.path.join(data_root,'BraTS_registered',file))
            if args.modality == "T2":
                temp = 't2.nii.gz'
            else:
                temp = 't1.nii.gz'
            test_mask.append(os.path.join(data_root,'BraTS_mask_registered',file.replace(temp,'seg.nii.gz')))
    
    #val_data = train_data[500:] + test_data[:50]
    #val_mask = [None]*(len(train_data)-500) + test_mask[:50]

    train_data = train_data

    val_data = test_data[:251]
    val_mask = test_mask[:251]
    test_data = test_data[251:]
    test_mask = test_mask[251:]

    train_dataset = MRIDataset_3D(train_data, resize_size=resize_size,volume_size=volume_size)
    val_dataset = MRIDataset_3D(val_data, val_mask,resize_size=resize_size,volume_size=volume_size)
    test_dataset = MRIDataset_3D(test_data, test_mask,resize_size=resize_size, volume_size = volume_size)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    #val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }
def evaluate(model,dataloaders):
    model.eval()
    model.featureExtractor.eval()
    if model.projection:
        model.projection.eval()
    model.flow_model.eval()
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
        for i, data in enumerate(tqdm(dataloaders['test_loader'])):

            ground_truth = data['mask'].cpu().numpy() # [B,H,W,D]
            B,H,W,D = ground_truth.shape
            logps = model.predict(data['image']).cpu().numpy() #[B,H,W,D]
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

            thres = metrics.return_best_thr(gts,prs)
            #iou_list.append(metrics.IoU_score(gts, prs>=thres))
            gts = gts.ravel()
            prs = prs.ravel()
            specificity_list.append(metrics.specificity_score(gts,prs>=thres))
            sensitivity_list.append(metrics.recall_score(gts,prs>=thres))
            
            #precision_list.append(metrics.precision_score(gts,prs>=thres))
            #acc_list.append(metrics.accuracy_score(gts,prs>=thres))
            #f1_list.append(metrics.f1_score(gts,prs>=thres))
            dice_list.append(metrics.calculate_maximum_dice(prs,gts))
        pr_sp = np.array(pr_list_sp)
        gt_sp = np.array(gt_list_sp)
        pr_sp = (pr_sp-pr_sp.min())/(pr_sp.max()-pr_sp.min())
        Iauroc = metrics.auroc_score(gt_sp,pr_sp)
        Iauprc = metrics.auprc_score(gt_sp,pr_sp)
        Ithres = metrics.return_best_thr(gt_sp,pr_sp)
        Ispecificity = metrics.specificity_score(gt_sp,pr_sp>=Ithres)
        Isensitivity = metrics.recall_score(gt_sp, pr_sp>=Ithres)
        #Iprecision = metrics.precision_score(gt_sp,pr_sp>=Ithres)
        #Iacc = metrics.accuracy_score(gt_sp,pr_sp>=Ithres)
        #If1 = metrics.f1_score(gt_sp,pr_sp>=Ithres)
        Idice = metrics.calculate_maximum_dice(pr_sp,gt_sp)
    result = {
            'p_auroc': round(np.mean(auroc_list),4),
            'p_auprc':round(np.mean(auprc_list),4),
            #'p_aupro':round(np.mean(aupro_list),4),
            'p_dice': round(np.mean(dice_list),4),
            'p_specificity': round(np.mean(specificity_list),4),
            'p_sensitivity': round(np.mean(sensitivity_list),4),
            #'p_iou': round(np.mean(iou_list), 4),
            #'p_precision':round(np.mean(precision_list),4),
            #'p_acc': round(np.mean(acc_list),4),
            #'p_f1': round(np.mean(f1_list),4),
            'i_auroc': round(Iauroc,4),
            'i_auprc': round(Iauprc,4),
            'i_dice': round(Idice,4),
            'i_specificity': round(Ispecificity,4),
            'i_sensitivity': round(Isensitivity, 4),
            #'i_precision': round(Iprecision,4),
            #'i_acc': round(Iacc,4),
            #'i_f1': round(If1,4),
        }

    for item, value in result.items():
        print('{}: {}\n'.format(item, value))
    return result
def load_model(model,result_path):
    savedModel = torch.load(result_path,weights_only=False)
    args = savedModel['args']
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
        
    if args.edc:
        backbone = load_backbone(args.backbone, args.edc)
    else:
        backbone = load_backbone(args.backbone)
    #args.score_model = 'flow'
    torch.cuda.empty_cache()
    
    if args.use_attention:
        model.attention.load_state_dict(savedModel['attention'])
    if args.proj_layers > 0:
        model.projection.load_state_dict(savedModel['projection'])
    if args.fine_tune:
        model.featureExtractor.featureExtractor.backbone.load_state_dict(savedModel['backbone'])
    if args.score_model == 'disc':
        model.discriminator.load_state_dict(savedModel['discriminator'])
    else:
        model.flow_model.load_state_dict(savedModel['flow_model'])
    model.eval()
    return model
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print('device: {}'.format(device))
    dataloaders = dataloader_IXI_BraTS_setup(args.data_root,
                                             resize_size = args.resize_size,
                                            volume_size = args.volume_size,
                                            batch_size = args.batch_size)
    print('train_size: {}, val_size: {}, test_size: {}'.format(
        len(dataloaders['train_loader'].dataset),
        len(dataloaders['val_loader'].dataset) if dataloaders['val_loader'] else 0,
        len(dataloaders['test_loader'].dataset)))
    
    
    # train with discriminator only
    if args.score_model=='flow':
        args.pos_embed_dim = 128
        args.clamp_alpha = 1.9

        args.pos_beta = 0.05
        args.margin_tau = 0.1
        args.normalizer = 10
    print('args: ',args)
    if args.edc:
        backbone = load_backbone(args.backbone, args.edc)
    else:
        backbone = load_backbone(args.backbone)
    
    torch.cuda.empty_cache()
    '''
    model = MRI_CNF_SimpleNet_3D_v2(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    '''
    '''
    model = MRI_SimpleNet_3D(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    '''
    '''
    model = MRI_CNF_SimpleNet_3D_pair_loss(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    '''
    #'''
    model = MRI_CNF_SimpleNet_3D_triplet_loss(backbone,
                            ['layer2','layer3'],
                            device,
                            preprocessing_dimension=args.preprocessing_dimension,
                            target_embed_dimension=args.target_embed_dimension,
                            args=args)
    
    #'''
    #model.train_model(dataloaders['train_loader'], dataloaders['val_loader'],args.epochs)
    saved_path = os.path.join(args.result_path, args.model_dir, args.sub_dir,args.run_name,'savedModels/best_val_model.pth')
    model = load_model(model,saved_path)
    result = evaluate(model,dataloaders)
    folder_path = os.path.join(args.result_path, args.model_dir, args.sub_dir,args.run_name)
    saved_model_path = os.path.join(folder_path, 'result.json')
    os.makedirs(folder_path, exist_ok=True)
    with open(saved_model_path,'w') as f:
        json.dump(result,f,indent=4)
if __name__ == '__main__':
    def parse_tuple(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError(f"Invalid tuple: {value}")

    parser = argparse.ArgumentParser()
    # environment setup
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    # path setup
    parser.add_argument("--data_root", type=str, default= "./../../data/BraTS2021_PREC_V2/T2")
    parser.add_argument("--result_path", type=str, default="./../results")
    parser.add_argument("--model_dir", type=str, default="MRI_3D_SimpleNet")
    parser.add_argument("--sub_dir", type=str, default="new_data_T2")
    parser.add_argument("--run_name", type=str, default='3D_triplet_T2_0.08_512_new')
    parser.add_argument("--run_info", type=str, default="trianing")
    parser.add_argument("--modality", type=str, default="T2")
    # data setup
    parser.add_argument("--volume_size", type=parse_tuple, default="(192,192,96)")
    #parser.add_argument("--resize_size", type=parse_tuple, default="(160,192,96)")
    parser.add_argument("--resize_size", type=parse_tuple, default="(192,192,96)")
    parser.add_argument("--test_size", type=float, default=0.5)
    # feature extraction setup
    parser.add_argument("--backbone", type=str,default="wide_resnet50_2")
    parser.add_argument("--patch_size",type=int, default=3)
    parser.add_argument("--edc", action="store_true")

    # noise settup
    parser.add_argument("--noise_type", type=str, default="simple")
    parser.add_argument("--noise_std", type=float, default=0.06)
    parser.add_argument("--noise_res", type=int, default=4)
    parser.add_argument("--mix_noise", type=int, default=1)
    
    # projection setup
    parser.add_argument("--proj_layers", type=int, default=0)
    parser.add_argument("--preprocessing_dimension", type=int, default=512)

    # conditional normalizing flow setup
    parser.add_argument("--flow_lr", type=float, default=1e-3)
    parser.add_argument("--flow_lr_step",type=int, default=10)
    parser.add_argument("--coupling_layers", type=int, default=8)
    parser.add_argument("--flow_hidden", type=int, default=1024)

    # discriminator setup
    parser.add_argument("--score_model", type=str, default='flow')
    parser.add_argument("--target_embed_dimension", type=int, default=512)
    parser.add_argument("--dsc_margin", type=float, default=0.5)
    parser.add_argument("--dsc_layers", type=int, default=2)
    parser.add_argument("--dsc_hidden", type=int, default=1024)

    # training setup
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_per_epochs",type=int,default=6)
    parser.add_argument("--stop_gradient",action="store_true",help="perform stop-gradient operation")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--soft_loss", type=str, default="BGSPP")

    # learning rate
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dsc_lr", type=float, default=2*1e-4)
    args = parser.parse_args()
    print(args.data_root, args.modality, args.sub_dir, args.run_name)
    main(args)
