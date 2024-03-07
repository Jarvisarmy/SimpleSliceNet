import torch.nn.functional as F
import torch

class MeanMapper(torch.nn.Module):
    """
        input :[B*D*h*w,c,ps,ps]
    """
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim
    def forward(self,features):
        features = features.reshape(len(features),1,-1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims,output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        self.input_dims = input_dims
        for input_dim in input_dims:
            module = MeanMapper(self.output_dim)
            self.preprocessing_modules.append(module)
    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
    
class LayerConcat(torch.nn.Module):
    def __init__(self, target_dim):
        super(LayerConcat, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns rehspaed and average pooled features."""
        features = features.reshape(len(features),1,-1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features),-1)
    
class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.layers = layers

    def forward(self, data):
        data = data
        features = {layer: [] for layer in self.layers}

        def hook_fn(layer):
            def hook(module,input,output):
                features[layer] = (output)
            return hook
        hooks = []
        for layer in self.layers:
            hook = self.backbone._modules.get(layer).register_forward_hook(hook_fn(layer))
            hooks.append(hook)
            
        if (data.shape[1] == 3):
            _ = self.backbone(data)
        else:
            B,D,_,H,W = data.size()
            #new_input = torch.cat([data[:,:,:,i] for i in range(data.size(3))],dim=0) # [B*D, H, W]
            #new_input = new_input.unsqueeze(-1) # [B*D, H, W, 1]
            #new_input = torch.cat([new_input,new_input,new_input],dim=-1) #[B*D, H, W, 3]
            #new_input = torch.swapaxes(new_input,1,3) 
            #new_input = torch.swapaxes(new_input,2,3) # [B*D,3,H,W]
            new_input = data.reshape(B*D,3,H,W)
            _ = self.backbone(new_input)
        for hook in hooks:
            hook.remove()
        return features
    
class PatchMaker(torch.nn.Module):
    def __init__(self, patchsize, stride=None):
        super(PatchMaker, self).__init__()
        self.patchsize = patchsize
        self.stride = stride

    def forward(self, extracted_features,layers):
        #print(extracted_features['layer2'][0].shape) # [B*D, C_2,H_2,W_2]
        all_features = []
        for layer in layers:
            features = extracted_features[layer]
            features = features.reshape(-1,*features.shape[1:]) # [B*D,C_2,H_2,W_2]
            # [_,_,512,24,24]
            # [_,_,1024,12,12]
            #features = features.permute(0,2,3,4,1) # [B,C_2,H_2,W_2,D]
            all_features.append(features)
        all_features = [
            self.patchify(x,return_spatial_info=True) for x in all_features
        ]
        #print(all_features[0][0].shape) # [B*D, H_2//stride*W_2//stride, patchsize,patchsize]
        #print(all_features[1][0].shape)
        patch_shapes = [x[1] for x in all_features]
        all_features = [x[0] for x in all_features]
        ref_num_patches = patch_shapes[0]

        for i in range(1,len(all_features)):
            _features = all_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:] # [B*D, H_2//stride,W_2//stride,c_l,patchsize,patchsize]
            )
            _features = _features.permute(0,-3,-2,-1,1,2) # [B*D, c_l,patchsize,patchsize, H_2//stride,W_2//stride]
            perm_base_shape = _features.shape
            _features = _features.reshape(-1,*_features.shape[-2:]) # [B*D*C_L*patchsize*patchsize,H_2//stride, W_2//stride]
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode='bilinear',
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2],ref_num_patches[0], ref_num_patches[1] #[B*D,C_l,patchsize,patchsize,H_2,W_2]
            )
            _features = _features.permute(0,-2,-1,1,2,3) # [B*D,H_0*W_0,C_l,patchsize,patchsize]
            _features = _features.reshape(len(_features),-1,*_features.shape[-3:]) #[bs,H_0*W_0,C_l,patchsize,patchsize]
            all_features[i] = _features
        # [nlayer, B*D, h*w,c,ps,ps]
        all_features = [x.reshape(-1,*x.shape[-3:]) for x in all_features] # [nlayer,B*D*h*w,c,ps,ps]
        #print(all_features[0].shape) #[B*D*h*w, c,ps,ps]
        return all_features, patch_shapes

    
    def patchify(self, features, return_spatial_info = False):
        """ Convert a tensor into a tensor of respective patches
        Args:
            x: [torch.Tensor, b
            s x c x w x h]
        Returns:
            x: [torch.Tensor, bs, w//stride * h//stride, c, patchsize]
        """
        padding = int((self.patchsize-1)/2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features) # [bs,c*patchsize*patchsize,Lw*Lh]
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) -1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2],self.patchsize, self.patchsize,-1
        ) # [bs, c, patchsize,patchsize,Lw*Lh]
        unfolded_features = unfolded_features.permute(0,4,1,2,3) # [bs, Lw*Lh, c,patchsize,patchsize]
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features
    
class FeatureAggregator(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 layers,
                 volume_size,
                 patch_size,
                 stride,
                 output_dim,
                 target_dim,
                 device):
        super(FeatureAggregator, self).__init__()
        self.layers = layers
        self.device = device
        self.featureExtractor = FeatureExtractor(backbone, layers).to(device)
        self.featureExtractor.eval()
        with torch.no_grad():
            if len(volume_size)==2:
                outputs = self.featureExtractor(torch.ones([1,3]+list(volume_size)).to(device))
            else:
                H,W,D = volume_size
                outputs = self.featureExtractor(torch.ones([1,D,3,H,W]).to(device))
        for layer in outputs:
            print(outputs[layer].shape)
        input_dims = [outputs[layer].shape[1] for layer in self.layers]
        self.patchMaker = PatchMaker(patch_size, stride).to(device)
        self.preprocessing = Preprocessing(input_dims, output_dim).to(device)
        self.layerConcat = LayerConcat(target_dim).to(device)

    def forward(self, data):
        data.to(self.device)
        data = self.featureExtractor(data) #[nlayers,]
        data, patch_shapes= self.patchMaker(data,self.layers)
        data = self.preprocessing(data) # [B*D*h*w,nlayers,preprocessing_dim]
        data = self.layerConcat(data) # [B*D*h*w, target_dim]
        return data, patch_shapes

    