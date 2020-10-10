import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import List
import torch.nn.functional as F
import encoding
from encoding.nn import Encoding
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
import pytorch_utils as pt_utils
from IPython import embed

class PointnetSAModuleVotes_enc(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            dilation: int = 1,
            K: int = 32
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.mlp = mlp
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.dilation= dilation
        self.K = K
        self.PointnetSAModuleVotes = PointnetSAModuleVotes(
            npoint = self.npoint,
            radius = self.radius,
            nsample = self.nsample,
            mlp = self.mlp,
            use_xyz = self.use_xyz,
            normalize_xyz = self.normalize_xyz,
            dilation = self.dilation
        )
        self.enc= enc_layer(D=mlp[-1],K=self.K)
        '''
        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt,dilation = dilation)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)
        '''

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        xyz, features, fps_inds = self.PointnetSAModuleVotes(xyz, features)
        feature_context = self.enc(features)
        features_out = feature_fusion(features,feature_context)
        del features

        return xyz, features_out, fps_inds

class PointnetSAModuleVotes_group_enc(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            dilation: int = 1,
            K: int = 32,
            G: int = 4

    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.mlp = mlp
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.dilation= dilation
        self.K = K
        self.G = G
        self.PointnetSAModuleVotes = PointnetSAModuleVotes(
            npoint = self.npoint,
            radius = self.radius,
            nsample = self.nsample,
            mlp = self.mlp,
            use_xyz = self.use_xyz,
            normalize_xyz = self.normalize_xyz,
            dilation = self.dilation
        )
        self.enc= enc_layer(D= int(int(mlp[-1])/G), K=self.K)



    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        xyz, features, fps_inds = self.PointnetSAModuleVotes(xyz, features)
        B, C, N = features.size()
        features_group = features.view(B,int(C/self.G),self.G, N)
        #features_group= features.view(B,N,self.G,int(C/self.G))
        #Wembed()

        feature_context = self.enc(features_group)
        features_out = feature_fusion_group(features_group,feature_context)
        features_out = features_out.view(B,C,N)
        del features

        return xyz, features_out, fps_inds


class PointnetSAModuleVotes_shuffle_enc(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            dilation: int = 1,
            K: int = 32,
            G: int = 4

    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.mlp = mlp
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.dilation= dilation
        self.K = K
        self.G = G
        self.PointnetSAModuleVotes = PointnetSAModuleVotes(
            npoint = self.npoint,
            radius = self.radius,
            nsample = self.nsample,
            mlp = self.mlp,
            use_xyz = self.use_xyz,
            normalize_xyz = self.normalize_xyz,
            dilation = self.dilation
        )
        self.enc= enc_layer(D= int(int(mlp[-1])/G), K=self.K)



    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        xyz, features, fps_inds = self.PointnetSAModuleVotes(xyz, features)
        B, C, N = features.size()

        features = channel_shuffle(self, features)
        features_group = features.view(B,int(C/self.G),self.G, N)

        feature_context = self.enc(features_group)
        features_out = feature_fusion_group(features_group,feature_context)
        features_out = features_out.view(B,C,N)
        del features

        return xyz, features_out, fps_inds



class PointnetSAModuleVotes_group_enc_v2(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            dilation: int = 1,
            K: int = 32,
            G: int = 4

    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.mlp = mlp
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.dilation= dilation
        self.K = K
        self.G = G
        self.PointnetSAModuleVotes = PointnetSAModuleVotes(
            npoint = self.npoint,
            radius = self.radius,
            nsample = self.nsample,
            mlp = self.mlp,
            use_xyz = self.use_xyz,
            normalize_xyz = self.normalize_xyz,
            dilation = self.dilation
        )
        '''
        for i in range(self.G):
            self.enc+str(i)= enc_layer(D= int(int(mlp[-1])/G), K=self.K)
        '''



    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        xyz, features, fps_inds = self.PointnetSAModuleVotes(xyz, features)
        B, C, N = features.size()
        features_group = features.view(B,int(C/self.G),self.G, N)
        #features_group= features.view(B,N,self.G,int(C/self.G))
        #Wembed()

        features_group_out = []#[:,:,0,:]

        for i in range( self.G):
            features_group_i = features_group[:,:,i,:]
            feature_context_i = self.enc(features_group_i)
            group_i_out = feature_fusion(features_group_i, feature_context_i)
            #feature_context = self.enc(features_group[:,:,i,:])
            #group_i_out = feature_fusion(features_group[:,:,i,:],feature_context)
            #features_group[:, :, i, :] = features_group_out
            #features_out = features_out.view(B,C,N)
            #del feature_context
            #features_group_out[:,:,i,:] = group_i_out
            features_group_out = features_group_out.append(group_i_out)
        features_group_out = torch.cat(features_group_out,dim =2)

        features_out = features_group_out.view(B,C,N)


        return xyz, features_out, fps_inds

'''
class PointnetSAModuleVotes_group_enc_v2(nn.Module):
    #Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    #with extra support for returning point indices for getting their GT votes 

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            dilation: int = 1,
            K: int = 32,
            G: int = 4

    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.mlp = mlp
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.dilation= dilation
        self.K = K
        self.G = G
        self.PointnetSAModuleVotes = PointnetSAModuleVotes(
            npoint = self.npoint,
            radius = self.radius,
            nsample = self.nsample,
            mlp = self.mlp,
            use_xyz = self.use_xyz,
            normalize_xyz = self.normalize_xyz,
            dilation = self.dilation
        )
        #self.enc = enc_layer(D=self.mlp[-1], K=self.K)
        self.enc= enc_layer_v2(D= int(int(mlp[-1])/G), K=self.K, G=self.G)



    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        xyz, features, fps_inds = self.PointnetSAModuleVotes(xyz, features)
        B, C, N = features.size()
        features_group = features.view(B,int(C/self.G),self.G, N)
        #features_group= features.view(B,N,self.G,int(C/self.G))
        #Wembed()

        feature_context = self.enc(features_group)
        features_out = feature_fusion(features,feature_context)
        #features_out = feature_fusion_group(features_group,feature_context)
        #features_out = features_out.view(B,C,N)
        del features

        return xyz, features_out, fps_inds
'''


class PointnetSAModuleVotes_gap(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            dilation: int = 1
            #K: int = 32
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.mlp = mlp
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.dilation= dilation
        #self.K = K
        self.PointnetSAModuleVotes = PointnetSAModuleVotes(
            npoint = self.npoint,
            radius = self.radius,
            nsample = self.nsample,
            mlp = self.mlp,
            use_xyz = self.use_xyz,
            normalize_xyz = self.normalize_xyz,
            dilation = self.dilation
        )
        self.gap=nn.Linear(mlp[-1],1)
        #self.enc= enc_layer(D=mlp[-1],K=self.K)
        '''
        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt,dilation = dilation)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)
        '''

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):

        xyz, features, fps_inds = self.PointnetSAModuleVotes(xyz, features)
        global_context = torch.mean(features.view(features.size(0), features.size(1), -1), dim=2)
        #global_contex = self.gap(features)
        #features_out = features* global_contex
        #feature_context = self.enc(features)
        features_out = feature_fusion(features,global_context)
        del features

        return xyz, features_out, fps_inds


class PointnetFPModule_enc(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True, K: int = 32):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)
        self.K = K
        self.enc = enc_layer(D=mlp[-1], K=self.K)



    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *known_feats.size()[0:2], unknown.size(1)
            )

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats],
                                   dim=1)  #(B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        new_features = new_features.squeeze(-1)

        feature_context = self.enc(new_features)

        new_features_out = feature_fusion(new_features,feature_context)
        del new_features

        return new_features_out






def enc_layer(D:int,K:int):
    net = nn.Sequential(Encoding(D, K),
                        # encoding.nn.EncodingV3(D=out_channel,K=K),
                        nn.ReLU(inplace=True),
                        #nn.ReLU(),
                        encoding.nn.View(-1, D * K),
                        encoding.nn.Normalize(),
                        nn.Linear(D * K, D),
                        nn.Sigmoid()
                        )
    return net




def enc_layer_v2(D:int,K:int, G:int ):
    net = nn.Sequential(Encoding(D, K),
                        # encoding.nn.EncodingV3(D=out_channel,K=K),
                        nn.ReLU(inplace=True),
                        #nn.ReLU(),
                        encoding.nn.View(-1, D * K),
                        encoding.nn.Normalize(),
                        nn.Linear(D * K, D*G),
                        nn.Sigmoid()
                        )
    return net

def feature_fusion(features,context):
    b, c, _ = features.size()
    features_context = context.view(b, c, 1)
    #features *=features_context
    out = features*features_context
    del features,features_context
    return out

def feature_fusion_group (features,context):
    b, c, g, _ = features.size()
    features_context = context.view(b, c, 1, 1)
    #features *=features_context
    out = features*features_context
    del features,features_context
    return out



def feature_fusion_res(features,context):
    b, c, _ = features.size()
    features_context = context.view(b, c, 1)

    #features_context *=features
    out = F.relu_(features+features*features_context)
    return out

def feature_fusion_FP(features,context):
    b, c, _ = features.size()
    features_context = context.view(b, c, 1)
    #features *=features_context
    out = features*features_context
    del features,features_context
    return out

def channel_shuffle(self, x):
    #batchsize, num_channels, height, width = x.data.size()
    B, C, N = x.size() ###Batch_Size, Channel, PointNum
    assert C % self.G == 0
    group_channels = C // self.G
    x = x.reshape(B, group_channels, self.G, N)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, C, N)
    return x