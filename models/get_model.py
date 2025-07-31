from torch import nn
from .A2S2KResNet import A2S2KResNet
from .ACB import ACB
from .AMF import AMF
from .BS2T.network import BS2T
from .CLOLN import CLOLN
from .CSCANet import CSCANet
from .DSNet.DSNet import DSNet
from .ELS2T import ELS2T
from .Ghostnet import GhostNet
from .HybridFormer import HybridFormer
from .HybridSN import HybridSN
from .LRDTN import LRDTN
from .LS2CM import LS2CMNet
from .S2VNet.S2VNet import S2VNet
from .ULite_FDRNet import ULite_FDRNet
from .cnn3d import cnn3d
from .dffn import dffn
from .gaht import gaht
from .morphFormer import morphFormer
from .simFormer import simFormer
from .speformer import speformer
from .ssftt import ssftt
from .MHIAIFormer import MHIAIFormer
from .LMSS_NAS import LMSS_NAS
from .ssftt_FDR import SSFTT_FDR


def get_model(model_name,
              dataset_name,
              in_chans=103,
              patch_size=7,
              out_channels_3d=1,
              num_heads=2,
              embed_dim=64,
              mlp_ratios=1,
              act_layer='gelu',
              num_extras=3,
              depths_te=3,
              ratio_2d=2,
              ratio_3d=2,
              threshold=0.8,
              group_size=16,
              mode='train',
              topk=1,
              Scale=5):
    if act_layer == 'relu':
        act_layer = nn.ReLU
    elif act_layer == 'gelu':
        act_layer = nn.GELU
    else:
        assert False, "unknown act type {}".format(act_layer)

    # example: model_name='cnn3d', dataset_name='pu'
    if model_name == 'ULite_FDRNet':
        model = ULite_FDRNet(dataset_name,
                          in_chans=in_chans,
                          patch_size=patch_size,
                          out_channels_3d=out_channels_3d,
                          num_heads=num_heads,
                          embed_dim=embed_dim,
                          mlp_ratios=mlp_ratios,
                          act_layer=act_layer,
                          num_extras=num_extras,
                          depths_te=depths_te,
                          ratio_2d=ratio_2d,
                          ratio_3d=ratio_3d,
                          Scale=Scale
                          )

    elif model_name == 'MHIAIFormer':
        model = MHIAIFormer(dataset_name, patch_size)

    elif model_name == '3DCNN':
        model = cnn3d(dataset_name, patch_size)

    elif model_name == 'A2S2K':
        model = A2S2KResNet(dataset_name, patch_size)

    elif model_name == 'CLOLN':
        model = CLOLN(dataset_name, patch_size)

    elif model_name == 'LS2CM':
        model = LS2CMNet(dataset_name, patch_size)

    elif model_name == 'GhostNet':
        model = GhostNet(dataset_name, patch_size)

    elif model_name == 'AMF':
        model = AMF(dataset_name, patch_size)

    elif model_name == 'DFFN':
        model = dffn(dataset_name, patch_size)

    elif model_name == 'HybridSN':
        model = HybridSN(dataset_name, patch_size)

    elif model_name == 'SpectalFormer':
        model = speformer(dataset_name, patch_size)

    elif model_name == 'SSFTT':
        # model = ssftt(dataset_name, patch_size, pca=True)
        model = ssftt(dataset_name, patch_size)

    elif model_name == 'GAHT':
        model = gaht(dataset_name, patch_size)

    elif model_name == 'BS2T':
        model = BS2T(dataset_name, patch_size)

    elif model_name == 'morphFormer':
        model = morphFormer(dataset_name, patch_size)

    elif model_name == 'LRDTN':
        model = LRDTN(dataset_name, patch_size)

    elif model_name == 'LMSS_NAS':
        model = LMSS_NAS(dataset_name, patch_size)

    elif model_name == 'ELS2T':
        model = ELS2T(dataset_name, patch_size)

    elif model_name == 'HybridFormer':
        model = HybridFormer(dataset_name, patch_size)

    elif model_name == 'CSCANet':
        model = CSCANet(dataset_name, patch_size, pca=True)  # pca=30

    elif model_name == 'S2VNet':
        model = S2VNet(dataset_name, patch_size)

    elif model_name == 'DSNet':
        model = DSNet(dataset_name, patch_size)

    elif model_name == 'ACB':
        model = ACB(dataset_name, patch_size)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model
