import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import torch.optim
def get_nnmodule_param_count(module:nn.Module):
    param_count = 0
    for param in module.state_dict().values():
        param_count += int(np.prod(param.shape))
    return param_count
########################


class PosEncodingNeRF(nn.Module):
    def __init__(self, in_channel,frequencies=10):
        super().__init__()

        self.in_channel = in_channel
        self.frequencies = frequencies
        self.out_channel = in_channel + 2 * in_channel * self.frequencies

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_channel)
        coords_pos_enc = coords
        for i in range(self.frequencies):
            for j in range(self.in_channel):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * math.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_channel).squeeze(1)
class NeRF(nn.Module):
    """
    B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, 
    and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,” 
    arXiv:2003.08934 [cs], Aug. 2020, Accessed: May 04, 2021. [Online]. 
    Available: http://arxiv.org/abs/2003.08934
    """
    def __init__(self,coords_channel=3,data_channel=1,frequencies=10,features=256,layers=5,skip=True):
        super().__init__()
        self.skip = skip
        self.skip_layer = (layers-1)//2 if skip else -1
        self.positional_encoding = PosEncodingNeRF(in_channel=coords_channel,frequencies=frequencies)
        in_channel = self.positional_encoding.out_channel
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(in_channel,features),nn.ReLU(inplace=True)))
        for i in range(layers-2):
            if self.skip_layer == i+1:
                self.net.append(nn.Sequential(nn.Linear(in_channel+features,features),nn.ReLU(inplace=True)))
            else:
                self.net.append(nn.Sequential(nn.Linear(features,features),nn.ReLU(inplace=True)))
        if self.skip_layer==layers-1:
            self.net.append(nn.Sequential(nn.Linear(in_channel+features,data_channel),nn.Sigmoid()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.ModuleList(self.net)
    def forward(self,coords):
        codings = self.positional_encoding(coords)
        output = codings
        for idx,model in enumerate(self.net):
            if idx == self.skip_layer:
                output = torch.cat([codings,output],1)
            output = model(output)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,frequencies,layers,skip,**kwargs):
        d =  coords_channel + 2 * coords_channel * frequencies
        if skip:
            param_count=d*features+features+(layers-2)*(features**2+features)+d*features+features*data_channel+data_channel
        else:
            param_count=d*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,frequencies,layers,skip,**kwargs):
        d =  coords_channel + 2 * coords_channel * frequencies
        a = layers-2
        if skip:
            b = 2*d+1+layers-2+data_channel
        else:
            b = d+1+layers-2+data_channel
        c = -param_count+data_channel
        features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features
###########################
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, in_channel,embsize=256,scale=10):
        super().__init__()
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.in_channel = in_channel
        self.embsize = embsize
        self.out_channel = 2 * self.embsize
        self.bvals = nn.Parameter(torch.normal(0,1,size=(embsize,in_channel))*scale)
        self.bvals.requires_grad=False
    def forward(self, coords):
        # coords = coords.view(coords.shape[0], -1, self.in_channel)
        emb = torch.cat([
            torch.sin((2.*math.pi*coords)@self.bvals.T),
            torch.cos((2.*math.pi*coords)@self.bvals.T)],-1)
        return emb

    
###########################
class HalfResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return 0.5*(self.fn(x, **kwargs) + x)
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
class Sine(nn.Module):
    def __init__(self,w0=30):
        super().__init__()
        self.w0=w0

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)
class SIREN(nn.Module):
    """
    V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein, 
    “Implicit Neural Representations with Periodic Activation Functions,” 
    arXiv:2006.09661 [cs, eess], Jun. 2020, Accessed: May 04, 2021. [Online].
    Available: http://arxiv.org/abs/2006.09661

    Y. Lu, K. Jiang, J. A. Levine, and M. Berger,
    “Compressive Neural Representations,” 
    Computer Graphics Forum, p. 12, 2021.
    """
    def __init__(self, coords_channel=3,data_channel=1,features=256,layers=5,w0=30,res=False,output_act=False,**kwargs):
        super().__init__()
        self.net=[]
        self.net.append(nn.Sequential(nn.Linear(coords_channel,features),Sine(w0)))
        for i in range(layers-2):
            if res:
                self.net.append(
                    HalfResidual(
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),
                        nn.Sequential(nn.Linear(features,features),
                        Sine(),))))
            else:
                self.net.append(nn.Sequential(nn.Linear(features,features),Sine()))
        if output_act:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel),Sine()))
        else:
            self.net.append(nn.Sequential(nn.Linear(features,data_channel)))
        self.net=nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)
        
    def forward(self,coords):
        output = self.net(coords)
        return output
    def forward_syn(self,coords,mods):
        """
        mods不调制最后一层
        """
        n,pc_d,pc_h,pc_w,pop,c = coords.shape
        coords = rearrange(coords,'n pc_d pc_h pc_w pop c -> (n pc_d pc_h pc_w) pop c')
        output = coords
        for layer, mod in zip(self.net, mods):
            output = layer(output)
            mod = rearrange(mod,'n c pc_d pc_h pc_w -> (n pc_d pc_h pc_w) () c')
            output *= mod.sigmoid()
        output = self.net[-1](output)
        output = rearrange(output,'(n pc_d pc_h pc_w) pop c -> n pc_d pc_h pc_w pop c',n=n,pc_d=pc_d,pc_h=pc_h,pc_w=pc_w)
        return output
    def forward_syn_wocrop(self,coords,mods):
        """mods不调制最后一层
        """
        output = coords
        for layer, mod in zip(self.net, mods):
            output = layer(output)
            mod = rearrange(mod,'n c -> n () c')
            output *= mod.sigmoid()
        output = self.net[-1](output)
        return output
    @staticmethod
    def calc_param_count(coords_channel,data_channel,features,layers,res,**kwargs):
        if res:
            param_count=coords_channel*features+features+2*(layers-2)*(features**2+features)+features*data_channel+data_channel
        else:
            param_count=coords_channel*features+features+(layers-2)*(features**2+features)+features*data_channel+data_channel
        return int(param_count)
    @staticmethod
    def calc_features(param_count,coords_channel,data_channel,layers,res,**kwargs):
        if res:
            a = (layers-2)*2
            b = coords_channel+1+2*layers-4+data_channel
            c = -param_count+data_channel
            # features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        else:
            a = layers-2
            b = coords_channel+1+layers-2+data_channel
            c = -param_count+data_channel
            
        if a == 0:
            features = round(-c/b)
        else:
            features = round((-b+math.sqrt(b**2-4*a*c))/(2*a))
        return features


ALLPHI={'SIREN':SIREN}
ALL_CALC_PHI_FEATURES = {'SIREN':SIREN.calc_features}
ALL_CALC_PHI_PARAM_COUNT = {'SIREN':SIREN.calc_param_count}
ALL_CHECK_PARAM_COUNT = {}

def init_phi(kwargs):
    kwargs = copy.deepcopy(kwargs)
    return ALLPHI[kwargs.pop('name')](**kwargs)


###########################################
if __name__=='__main__':
    siren=SIREN(res=False)
    data = torch.rand((1,1000,3))
    output = siren(data)
    print(output.shape)