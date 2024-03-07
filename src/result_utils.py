from ipywidgets import interact
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from utils import  load_backbone
from models import MRI_SimpleNet_2D


def load_MRI_SimpleNet(result_path):
    savedModel = torch.load(result_path)
    print("epoch: {}, loss: {}, p_true: {}, p_fake: {}".format(savedModel['epoch'],
                                               savedModel['loss'],
                                               savedModel['p_true'],
                                               savedModel["p_fake"]))
    args = savedModel['args']
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if args.edc:
        backbone = load_backbone(args.backbone, args.edc)
    else:
        backbone = load_backbone(args.backbone)
    model = MRI_SimpleNet_2D(backbone,
                         ['layer2','layer3'],
                         device,
                         preprocessing_dimension=args.preprocessing_dimension,
                         target_embed_dimension=args.target_embed_dimension,
                         args=args).to(device)
    if args.use_attention:
        model.attention.load_state_dict(savedModel['attention'])
    if args.proj_layers > 0:
        model.projection.load_state_dict(savedModel['projection'])
    if args.fine_tune:
        model.featureExtractor.featureExtractor.backbone.load_state_dict(savedModel['backbone'])
    model.discriminator.load_state_dict(savedModel['discriminator'])
    model.eval()
    return model, args

def visualize_nii_3d(volume):
    def f(layer):
        plt.figure(figsize=(10,5))
        plt.imshow(volume[:,:,layer],cmap="gray")
        plt.axis('off')
        return layer
    interact(f, layer=(0,volume.shape[2]-1))

def load_and_visualize_2D(result_path,test_loader):
    model, args = load_MRI_SimpleNet(result_path)
    fig, axs = plt.subplots(4,6,figsize=(9,6))
    min_normal = []
    max_normal=[]
    min_abnormal=[]
    max_abnormal=[]
    random.seed(1)
    random_samples = random.sample(range(len(test_loader.dataset)),k=6)
    for i in range(len(random_samples)):
        data = test_loader.dataset[random_samples[i]]
        mask = model.predict(data['image'].unsqueeze(0)).squeeze().cpu().numpy()
        img = data['original'].squeeze().permute(1,2,0).cpu().numpy()
        ground_truth = data['mask'].squeeze().cpu().numpy()
        min_normal.append(mask[ground_truth==0].min())
        max_normal.append(mask[ground_truth==0].max())
        print(mask[ground_truth==0].max())
        if (ground_truth == 1).any() ==1:
            min_abnormal.append(mask[ground_truth==1].min())
            max_abnormal.append(mask[ground_truth==1].max())
        #mask = (mask-min_score)/(max_score-min_score)
        bn_mask = mask > 0
        axs[0,i].imshow(img,cmap="gray")
        axs[0,i].axis('off')
        axs[1,i].imshow(mask)
        axs[1,i].axis('off')
        axs[2,i].imshow(bn_mask,cmap='gray')
        axs[2,i].axis('off')
        axs[3,i].imshow(ground_truth,cmap='gray')
        axs[3,i].axis('off')
    plt.tight_layout()
    plt.show()
    print('min_normal',sum(min_normal)/len(min_normal))
    print('max_normal', sum(max_normal)/len(max_normal))
    print('min_abnormal',sum(min_abnormal)/len(min_abnormal))
    print('max_abnormal', sum(max_abnormal)/len(max_abnormal))