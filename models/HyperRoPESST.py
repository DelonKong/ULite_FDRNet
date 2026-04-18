# -*- coding: utf-8 -*-

"""
Z. Khan, N. Dey, K. Kathiravan, S.-C. Yoon, and S. M. Bhandarkar,
“Spatial–Spectral Transformer With Patch-Local Mixed-Axis 2-D Rotary Position Embedding for Hyperspectral Image Classification,”
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 19, pp. 1910–1925, 2026, doi: 10.1109/JSTARS.2025.3643365.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict

from torchinfo import summary

deviceType = "cuda"


class RoPEUtils:

    @staticmethod
    def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1) # [mag * cos(theta), mag * cos(theta + pi/2)] We know that cos(theta + pi/2) = -sin(theta), [mag[1]*cos(theta), mag[2]*cos(theta), mag[1]* -sin(theta), mag[2]* -sin(theta)]
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1) # [mag * sin(theta), mag * sin(theta + pi/2)] We know that sin(theta + pi/2) = cos(theta), [mag[1]*sin(theta), mag[2]*sin(theta), mag[1]*cos(theta), mag[2]*cos(theta)]
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)  #shape is [2, num_heads, 4] 
        return freqs

    @staticmethod
    def init_coords_xy(end_x: int, end_y: int):
        t = torch.arange(end_x * end_y, dtype=torch.float32)  #shape is [patchsize * patchsize = 25]
        t_x = (t % end_x).float() #[0 ,1 , 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        t_y = torch.div(t, end_x, rounding_mode='floor').float() ##[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        return t_x, t_y
    
    @staticmethod
    def compute_axial_cis(freqs_x: torch.Tensor, freqs_y: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
        freqs_x = torch.outer(t_x, freqs_x)
        freqs_y = torch.outer(t_y, freqs_y)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x) # e^(i . (freqs_x * t_x))  shape is [25, 8] since total number of pixels (positions) in the 5x5 patch is 25, and The Number of Thetas in each Head for x-axis is 32/4 = 8  
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y) # e^(i . (freqs_y * t_y))  shape is [25, 8] since total number of pixels (positions) in the 5x5 patch is 25, and The Number of Thetas in each Head for y-axis is 32/4 = 8  
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1) #shape is [25, 16] in complex numbers for each head

    @staticmethod
    def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
        N = t_x.shape[0]
        depth = freqs.shape[1]
        with torch.amp.autocast(enabled=False, device_type=deviceType): #This line ensures that the following operations run in full precision (float32)
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3) # [N , 3, 32]  --> [3, 25, 8, 4] --> [3, 8, 25, 4]
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3) # [N , 3, 32]  --> [3, 25, 8, 4] --> [3, 8, 25, 4]
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y) #e^(i . (freqs_x + freqs_y)) this outputs a complex number with real and imaginary parts derived from combined x and y frequencies
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
        elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) #This converts the last two dimensions of xq into a complex number [65, 8, 25, 32] --> [65, 8, 25, 16]
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) #This converts the last two dimensions of xk into a complex number [65, 8, 25, 32] --> [65, 8, 25, 16]
        freqs_cis = RoPEUtils.reshape_for_broadcast(freqs_cis, xq_) #This reshapes freqs_cis to match the shape of xq_ [ 8, 25, 16] --> [65, 8, 25, 16
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) #This applies the rotation to the complex numbers in xq_, then converts it back to real numbers [65, 8, 25, 16, 2] and flattens the last two dimensions [65, 8, 25, 32]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) #This applies the rotation to the complex numbers in xk_, then converts it back to real numbers [65, 8, 25, 16, 2] and flattens the last two dimensions [65, 8, 25, 32]
        return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_heads: int, dropout: float, pos_encoding_type: str):
        super().__init__()
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads ** -0.5
        self.pos_encoding_type = pos_encoding_type

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, freqs_cis=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        if self.pos_encoding_type == "absolute":
            dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale #Since we already added absolute position embedding in the input
        elif self.pos_encoding_type in [ "rope_2d_mixed", "rope_2d_axial"]:
            q, k = RoPEUtils.apply_rotary_emb(q, k, freqs_cis)
            dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale


        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class RoPETransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_heads: int, mlp_dim: int, dropout: float, pos_encoding_type: str):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, RoPEAttention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout, pos_encoding_type=pos_encoding_type))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, freqs_cis=None, pe_type=None):
        x_center = []
        for i, (attention, mlp) in enumerate(self.layers):
            if pe_type == "rope_2d_mixed":
                x = attention(x, freqs_cis=freqs_cis[i]) #for mixed_rope, freqs_cis is a list of freqs_cis for each Transformer layer 
            elif pe_type == "rope_2d_axial":
                x = attention(x, freqs_cis=freqs_cis) #for axial_rope, freqs_cis is a single tensor for all Transformer layers
            elif pe_type == "absolute":
                x = attention(x, freqs_cis=None) #for absolute position embedding, we don't need to pass freqs_cis
            x = mlp(x)
            index = x.shape[1] // 2
            x_center.append(x[:,index,:])
        return x, x_center
    
class HyperRopeViT(nn.Module):
    def __init__(self, params: Dict):
        super(HyperRopeViT, self).__init__()
        self.params = params
        net_params = params['net']
        data_params = params['data']

        self.num_classes = data_params.get("num_classes", 13)
        self.patch_size = data_params.get("patch_size", 5)
        self.spectral_size = data_params.get("spectral_size", 171)
        self.data_sign = data_params.get("data_sign", "Plastic")
        self.pca_components = data_params.get("pca", 0)

        self.depth = net_params.get("depth", 1)
        self.heads = net_params.get("heads", 8)
        self.mlp_intermediate_dim = net_params.get("mlp_intermediate_dim", 256)
        self.kernel = net_params.get('kernal', 3)
        self.padding = net_params.get('padding', 1)
        self.dropout_rate = net_params.get("dropout", 0)
        
        self.dim = net_params.get("dim", 64)
        self.conv2d_out = self.dim #Features as Embeddings for each pixel in the patch
        self.dim_heads = self.dim // self.heads
        
        self.conv2d_features = self._create_conv2d_features()
        self.dropout = nn.Dropout(0.1)
        
        
        
        self.pos_encoding_type = net_params.get("pos_encoding_type", "rope_2d_mixed")
        
        if self.pos_encoding_type == "absolute":
            self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_size * self.patch_size, self.dim))
            
        elif self.pos_encoding_type == "rope_2d_mixed":
            self.rope_theta = 10.0
            self.freqs = self._init_rope_freqs()
            self.register_coords_xy_buffers()
            
        elif self.pos_encoding_type == "rope_2d_axial":
            self.rope_theta = 100.0
            self.freqs_x = 1.0 / (self.rope_theta ** (torch.arange(0, self.dim_heads, 4)[: (self.dim_heads // 4)].float() / self.dim_heads))
            self.freqs_y = 1.0 / (self.rope_theta ** (torch.arange(0, self.dim_heads, 4)[: (self.dim_heads // 4)].float() / self.dim_heads))
            self.register_buffer('freqs_axial_x', self.freqs_x)
            self.register_buffer('freqs_axial_y', self.freqs_y)
            self.register_coords_xy_buffers()

        self.local_trans_pixel = RoPETransformer(
            dim=self.dim, depth=self.depth, heads=self.heads, 
            dim_heads=self.dim_heads, mlp_dim=self.mlp_intermediate_dim, dropout=self.dropout_rate,
            pos_encoding_type=self.pos_encoding_type
        )

        self.center_weight = nn.Parameter(torch.ones(self.depth, 1, 1) * 0.001)

        self.classifier_mlp = self._create_classifier_mlp()

    def _create_conv2d_features(self):
        '''
        if self.data_sign in ['IndianPine', 'Pavia', 'Houston']:
            return nn.Sequential(
                ZeroMaskedConv2d(
                    in_channels=self.spectral_size, 
                    out_channels=self.conv2d_out, 
                kernel_size=self.kernel, 
                padding=self.padding
            ),
            nn.BatchNorm2d(self.conv2d_out),
            nn.ReLU()
            )
        
        '''
        if self.pca_components > 0:
            return nn.Sequential(
                nn.Conv2d(in_channels=self.pca_components, out_channels=self.conv2d_out, 
                          kernel_size=(self.kernel, self.kernel), padding=(self.padding, self.padding)),
                nn.BatchNorm2d(self.conv2d_out),
                nn.ReLU(),
            )
        elif self.pca_components == 0:
            return nn.Sequential(
                    nn.Conv2d(in_channels=self.spectral_size, out_channels=self.conv2d_out, 
                          kernel_size=(self.kernel, self.kernel), padding=(self.padding, self.padding)),
                nn.BatchNorm2d(self.conv2d_out),
                nn.ReLU(),
            )

    def _init_rope_freqs(self):
        if self.pos_encoding_type == "rope_2d_mixed":
            freqs = []
            for _ in range(self.depth):
                freqs.append(
                    RoPEUtils.init_2d_freqs(dim=self.dim // self.heads, num_heads=self.heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, self.depth, -1)  #shape is [2, depth, num_heads * 4]
            return nn.Parameter(freqs.clone(), requires_grad=True)


    def register_coords_xy_buffers(self):
        t_x, t_y = RoPEUtils.init_coords_xy(end_x=self.patch_size, end_y=self.patch_size)
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)
        

    def _create_classifier_mlp(self):
        linear_dim = self.dim * 2
        return nn.Sequential(
            nn.Linear(self.dim, linear_dim),
            nn.BatchNorm1d(linear_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(linear_dim, self.num_classes),
        )
        
    def _compute_freqs_cis(self):
        if self.pos_encoding_type == "rope_2d_mixed":
            return RoPEUtils.compute_mixed_cis(self.freqs, self.freqs_t_x, self.freqs_t_y, num_heads=self.heads)
        elif self.pos_encoding_type == "rope_2d_axial":
            return RoPEUtils.compute_axial_cis(self.freqs_axial_x, self.freqs_axial_y, self.freqs_t_x, self.freqs_t_y)
    

    def _compute_pos_encoding(self, x):
        if self.pos_encoding_type == "absolute":
            return x + self.pos_embedding
        elif self.pos_encoding_type == "relative_bias":
            return x, self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size * self.patch_size, self.patch_size * self.patch_size, -1)
        elif self.pos_encoding_type == "rope_2d_mixed":
            return self._compute_freqs_cis()
        elif self.pos_encoding_type == "rope_2d_axial":
            return self._compute_freqs_cis()

    def encoder_block(self, x):
        b, s, w, h = x.shape
        x_pixel = self.conv2d_features(x)
        x_pixel = rearrange(x_pixel, 'b s w h -> b (w h) s')
        
        pos_encoding = self._compute_pos_encoding(x_pixel)
        if self.pos_encoding_type == "absolute":
            x_pixel = pos_encoding
            
        x_pixel = self.dropout(x_pixel)

        x_pixel, x_center_list = self.local_trans_pixel(x_pixel, freqs_cis=pos_encoding, pe_type=self.pos_encoding_type)
        x_center_tensor = torch.stack(x_center_list, dim=0)
        logit_pixel = torch.sum(x_center_tensor * self.center_weight, dim=0)
        #reduce_x = torch.mean(x_pixel, dim=1) This is if we take the mean of all the 25 pixels in patch instead of center pixel
        return logit_pixel



    def forward(self, x):
        '''
        x: (Batch, Spectral, Width, Height)
        
        '''
        logit_x = self.encoder_block(x)
        return self.classifier_mlp(logit_x)


class overall(nn.Module):
    def __init__(self,
                 patch_size,
                 in_chans,
                 num_classes,
                 depth=1,
                 mlp_dim=1,
                 dim=64,
                 heads=8,
                 kernel=3,
                 padding=1,
                 dropout=0.1,
                 pos_encoding_type="rope_2d_mixed",
                 pca=0,
                 data_sign="custom"):
        super().__init__()

        # 兼容两种写法：
        # 1) mlp_dim 是 ratio，比如 1/2/4
        # 2) mlp_dim 直接就是隐藏层维度，比如 128/256
        if isinstance(mlp_dim, (int, float)) and mlp_dim <= 8:
            mlp_hidden_dim = int(dim * mlp_dim)
        else:
            mlp_hidden_dim = int(mlp_dim)

        params = {
            "data": {
                "num_classes": num_classes,
                "patch_size": patch_size,
                "spectral_size": in_chans,
                "data_sign": data_sign,
                "pca": pca
            },
            "net": {
                "depth": depth,
                "heads": heads,
                "mlp_intermediate_dim": mlp_hidden_dim,
                "kernal": kernel,   # 这里保持和你原代码里的键名一致
                "padding": padding,
                "dropout": dropout,
                "dim": dim,
                "pos_encoding_type": pos_encoding_type
            }
        }

        self.backbone = HyperRopeViT(params)

    def forward(self, x):
        """
        兼容两种输入：
        1) [B, 1, C, P, P]   <- 你的新接口示例
        2) [B, C, P, P]      <- 原始 HyperRopeViT 接口
        """
        if x.dim() == 5:
            # [B, 1, C, P, P] -> [B, C, P, P]
            if x.size(1) != 1:
                raise ValueError(
                    f"Expected x.shape[1] == 1 for 5D input, but got {x.shape}"
                )
            x = x.squeeze(1)

        elif x.dim() != 4:
            raise ValueError(
                f"Input must be 4D [B,C,P,P] or 5D [B,1,C,P,P], but got {x.shape}"
            )

        return self.backbone(x)



def HyperRoPESST(dataset,
                 patch_size=9,
                 mlp_ratios=1,
                 depths_te=1):
    model = None

    if dataset == 'sa':
        model = overall(
            patch_size=patch_size,
            in_chans=204,
            num_classes=16,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'pu':
        model = overall(
            patch_size=patch_size,
            in_chans=103,
            num_classes=9,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'whulk':
        model = overall(
            patch_size=patch_size,
            in_chans=270,
            num_classes=9,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'hrl':
        model = overall(
            patch_size=patch_size,
            in_chans=176,
            num_classes=14,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'whuhh':
        model = overall(
            patch_size=patch_size,
            in_chans=270,
            num_classes=22,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'whuhc':
        model = overall(
            patch_size=patch_size,
            in_chans=274,
            num_classes=16,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'IP':
        model = overall(
            patch_size=patch_size,
            in_chans=200,
            num_classes=16,
            depth=depths_te,
            mlp_dim=mlp_ratios
        )
    elif dataset == 'BS':
        model = overall(
            patch_size=patch_size,
            in_chans=145,
            num_classes=14
        )
    elif dataset == 'HsU':
        model = overall(
            patch_size=patch_size,
            in_chans=144,
            num_classes=15
        )
    elif dataset == 'KSC':
        model = overall(
            patch_size=patch_size,
            in_chans=176,
            num_classes=13
        )
    elif dataset == 'pc':
        model = overall(
            patch_size=patch_size,
            in_chans=102,
            num_classes=9
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = torch.randn(size=(64, 1, 103, 9, 9)).to(device)
    net = HyperRoPESST(dataset='pu', patch_size=9)
    net = net.to(device)
    out = net(t)
    print("output shape:", out.shape)   # 预期 [64, 9]
    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)
        print(sum.trainable_params)