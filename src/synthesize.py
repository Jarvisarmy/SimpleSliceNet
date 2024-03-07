import torch
import random


def simple_noise_on_features(noise_std, mix_noise, true_feats):
    """ Create simple noise of features
    Args:
        noise_std (int): standard deviation of the gaussian noise
        mix_noise (int): number of gaussian noise used
        true_feats (Tensor)
    """
    noise_idxs = torch.randint(0,mix_noise,torch.Size([true_feats.shape[0]]))
    noise_one_hot = torch.nn.functional.one_hot(noise_idxs,num_classes=mix_noise).to(true_feats.device)
    noise = torch.stack([
                        torch.normal(0,noise_std*1.1**(k), true_feats.shape)
                        for k in range(mix_noise)], dim=1).to(true_feats.device)
    noise = (noise*noise_one_hot.unsqueeze(-1)).sum(1)
    return noise

def coarse_noise_on_images(noise_std, noise_res, data,foreground_margin=0):
    B,C,H,W = data.shape
    ns = torch.normal(mean=torch.zeros(B,C,noise_res,noise_res),std=noise_std)
    ns = torch.nn.functional.interpolate(ns, size=(H,W),mode='bilinear',align_corners=False)
    roll_x = random.choice(range(H))
    roll_y = random.choice(range(W))
    ns = torch.roll(ns, shifts=[roll_x,roll_y],dims=[-2,-1]).to(data.device)

    mask = data.sum(dim=1,keepdim=True) > foreground_margin
    ns *= mask 
    return ns