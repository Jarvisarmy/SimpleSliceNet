import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import nibabel as nib
import numpy as np


import nibabel as nib
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def crop_3d_volume_to_size(data,resize_size=None, target_size=None):
    """ Resize the volume to resize_size, and then center-crop to target_size
    Args:
        data [H,W,D] (any of np.ndarray, list, torch.tensor): 3D volume data 

    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    elif isinstance(data, list):
        data = torch.tensor(data)
    
    h, w, d = data.shape
    if target_size:
        crop_h, crop_w, crop_d= target_size

        crop_d_start = max(0, (d - crop_d) // 2)
        crop_h_start = max(0, (h - crop_h) // 2)
        crop_w_start = max(0, (w - crop_w) // 2)

        crop_d_end = min(d, crop_d_start + crop_d)
        crop_h_end = min(h, crop_h_start + crop_h)
        crop_w_end = min(w, crop_w_start + crop_w)
        
        #crop_h_end = h-1
        #crop_h_start = crop_h_end - crop_h

        # Perform center crop
        cropped_volume = data[crop_h_start:crop_h_end, crop_w_start:crop_w_end,crop_d_start:crop_d_end]
    else:
        cropped_volume = data
    if resize_size:
        resized_volume = F.interpolate(cropped_volume.unsqueeze(0).unsqueeze(0), 
                                   size = resize_size,
                                   mode = 'trilinear',
                                   align_corners=False).squeeze(0).squeeze(0)
    else:
        resized_volume = cropped_volume
    #cropped_volume = cropped_volume.permute(2,0,1)
    #cropped_volume = F.interpolate(cropped_volume.unsqueeze(0), 
    #                               size = (224,224),
    #                               mode = 'bilinear',
    #                               align_corners=False).squeeze(0)
    #cropped_volume = cropped_volume.permute(1,2,0)
    return resized_volume

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

    cropped_volume = crop_3d_volume_to_size(volume, resize_size = self.resize_size,target_size=self.volume_size) #[H,W,D]
    #normalized_volume = normalize_3d_volume(cropped_volume)

    # get the cropped mask

    shape = cropped_volume.shape #[H,W,D]
    if self.mask_list and self.mask_list[idx]:
      mask_path = self.mask_list[idx]
      mask = torch.from_numpy(nib.load(mask_path).get_fdata())
      cropped_mask = crop_3d_volume_to_size(mask, resize_size=self.resize_size,target_size=self.volume_size)
      assert(cropped_mask.max() > 0)
      cropped_mask[cropped_mask > 0] = 1
    else:
      cropped_mask = torch.zeros(shape)
    label = torch.max(cropped_mask) > 0
    # apply transform to normalized_volume and cropped_mask

    return {'image': normalize_3d_volume_to_imagenet(cropped_volume).float(), # [D,3,H,W]
            'original': cropped_volume.float(), # [H,W,D]
            'label': label,
            'mask': cropped_mask} # [H,W,D]
  

if __name__ == '__main__':
   volume = torch.ones(8,64,64,10)
   normalize_3d_volume_to_imagenet(volume)
