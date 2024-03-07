from torch.utils.data import Dataset
from torchvision import transforms
import torch
import PIL
import os
import random
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MRIDataset_2D(Dataset):
  def __init__(self, data_list,
               mask_list=None,
               label_list=None,
               resize_size=(224,224), 
               target_size=(224,224)):
    super().__init__()
    self.data_list = data_list
    self.mask_list = mask_list
    self.resize_size = resize_size
    self.target_size = target_size
    self.label_list = label_list
    self.transform_img = transforms.Compose([
      transforms.Resize(resize_size),
      transforms.CenterCrop(target_size),
      transforms.ToTensor()
    ])
    self.transform_mask = transforms.Compose([
      transforms.Resize(resize_size),
      transforms.CenterCrop(target_size),
      transforms.PILToTensor()
    ])

  def __len__(self):
    return len(self.data_list)
  def __getitem__(self, idx):
    # get the image
    image_path = self.data_list[idx]
    image = PIL.Image.open(image_path).convert('RGB')
    image = self.transform_img(image)
    shape = image.shape

    # get the mask
    if self.mask_list:
      mask_path = self.mask_list[idx]
      mask = PIL.Image.open(mask_path)
      mask = self.transform_mask(mask)
      mask[mask > 0] = 1

    else:
      mask = torch.zeros(self.target_size)

    # get the image label
    if self.label_list:
      label = self.label_list[idx]
    else:
      label = torch.max(mask) > 0


    
    return {'image': transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(image).float(), 
            'original': image.float(),
            'label': label,
            'mask': mask,
            'file_name': image_path}