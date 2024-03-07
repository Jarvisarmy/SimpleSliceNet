import torch

import numpy as np
import torch.nn as nn
import math

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1,layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i==0 else _out
            _out = out_planes
            self.layers.add_module("fc{}".format(i),torch.nn.Linear(_in,_out))
            if i < n_layers-1:
                if layer_type > 1:
                    self.layers.add_module("relu{}".format(i),torch.nn.LeakReLU(.2))
        self.apply(init_weight)
    def forward(self,x):
        x = self.layers(x)
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers = 1, hidden=None):
        super(Discriminator, self).__init__()
        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i==0 else _hidden
            _hidden = int(_hidden //1.5) if hidden is None else hidden
            self.body.add_module('block{}'.format(i),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in,_hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)
    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x






class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels,num_heads=8):
        super(SelfAttentionModule, self).__init__()
        self.in_channels=in_channels
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.in_channels, num_heads=num_heads)
    def forward(self, x):
        # Rearrange dimensions for self-attention
        N,C,H,W = x.size()
        #positional_encoding = self.positional_encoding(C,H,W).to(x.device)
        #x = torch.cat((x,positional_encoding.repeat(N,1,1,1)),dim=1) #[N,2C,H,W]
        #x = x + positional_encoding
        x = x.view(N,C,-1).transpose(1,2)
        x, _ = self.multihead_attn(x,x,x)
        x = x.transpose(1,2).view(N,C,H,W)
        

        return x

    def get_angles(self,pos, i, d_model):
        angle_rates = 1/torch.pow(10000, (2*(i//2))/d_model)
        return pos * angle_rates
    def positional_encoding(self,d_model, height,width):
        pos_h = torch.arange(height).unsqueeze(1) # [height,1]
        pos_w = torch.arange(width).unsqueeze(1) # [width,1]
        angle_rads_h = self.get_angles(pos_h, torch.arange(d_model//2).unsqueeze(0),d_model)
        angle_rads_w = self.get_angles(pos_w, torch.arange(d_model //2).unsqueeze(0),d_model)

        pos_encoding_h = torch.zeros(height, d_model)
        pos_encoding_w = torch.zeros(width, d_model)
        pos_encoding_h[:,0::2] = torch.sin(angle_rads_h)
        pos_encoding_h[:,1::2] = torch.cos(angle_rads_h)
        pos_encoding_w[:,0::2] = torch.sin(angle_rads_w)
        pos_encoding_w[:,1::2] = torch.cos(angle_rads_w)
        pos_encoding = pos_encoding_h.unsqueeze(1)+pos_encoding_w.unsqueeze(0)
        pos_encoding = pos_encoding.unsqueeze(0).permute(0,3,1,2)
        return pos_encoding

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    
    return P

if __name__ == '__main__':
    model = SelfAttentionModule(in_channels=512)
    x = torch.ones((8,512,64,64))
    res = model(x)
    print(res.shape)

