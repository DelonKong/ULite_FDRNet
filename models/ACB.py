# -*- coding: utf-8 -*-

import warnings
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import torch.nn as nn
import torch
import math
from torchinfo import summary

"""
Y. Cui, L. Zhu, C. Zhao, L. Wang, and S. Gao, 
“Lightweight Spectral–Spatial Feature Extraction Network Based on Domain Generalization for Cross-Scene Hyperspectral Image Classification,” 
IEEE Transactions on Geoscience and Remote Sensing, 
vol. 62, pp. 1–14, Aug. 2024, doi: 10.1109/TGRS.2024.3427375.
"""

class Mlp(nn.Module):
    def __init__(self, channel, mlp_dim):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(channel, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, channel)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


def multi_head_trans(input_feature, num_heads):
    new_x_shape = input_feature.size()[:-1] + (num_heads, input_feature.shape[-1] // num_heads)
    output_feature = input_feature.view(*new_x_shape)
    return output_feature.permute(0, 2, 1, 3)


class LinearAttention(nn.Module):
    def __init__(self, heads_num, channel, qkv_channel, pixel_num):
        super(LinearAttention, self).__init__()
        self.num_attention_heads = heads_num

        self.qk_channel, self.v_channel = qkv_channel

        self.query = nn.Linear(channel, self.qk_channel)
        self.key = nn.Linear(channel, self.qk_channel)
        self.value = nn.Linear(channel, self.v_channel)

        self.q_token = nn.Linear(self.qk_channel, self.qk_channel)
        self.attention_dropout = nn.Dropout(0.1)

        self.out = nn.Linear(self.v_channel, channel)
        self.proj_dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_feature):
        mixed_query_layer = self.query(input_feature)
        mixed_key_layer = self.key(input_feature)
        mixed_value_layer = self.value(input_feature)

        mixed_query_layer = self.q_token(mixed_query_layer)

        query_layer = multi_head_trans(mixed_query_layer, self.num_attention_heads)
        key_layer = multi_head_trans(mixed_key_layer, self.num_attention_heads)
        value_layer = multi_head_trans(mixed_value_layer, self.num_attention_heads)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(query_layer.shape[-1])
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.v_channel,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Bottleneck(nn.Module):
    multiple = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        self.mid_channel = int(out_channel/self.multiple)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.mid_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channel)
        self.conv2 = nn.Conv2d(in_channels=self.mid_channel, out_channels=self.mid_channel, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(self.mid_channel)
        self.conv3 = nn.Conv2d(in_channels=self.mid_channel, out_channels=out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, pixel_num, channel):
        super(Embeddings, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel))
        self.position_embeddings = nn.Parameter(torch.zeros(1, pixel_num + 1, channel))

        self.dropout = nn.Dropout(0.1)

    def forward(self, feature):
        B = feature.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feature_cls = torch.cat((cls_tokens, feature), dim=1)

        embeddings = feature_cls + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class ResidualLinearAttentionBlock(nn.Module):
    def __init__(self, heads_num, channel, mlp_dim, qkv_channel, pixel_num):
        super(ResidualLinearAttentionBlock, self).__init__()

        self.attention_norm = nn.LayerNorm(channel, eps=1e-6)
        self.attention = LinearAttention(heads_num, channel, qkv_channel, pixel_num)

        self.ffn_norm = nn.LayerNorm(channel, eps=1e-6)
        self.ffn = Mlp(channel, mlp_dim)

    def forward(self, input_feature):
        input_copy = input_feature
        norm_feature = self.attention_norm(input_feature)
        attention_feature = self.attention(norm_feature)
        residual_feature = attention_feature + input_copy

        residual_copy = residual_feature
        norm_feature = self.ffn_norm(residual_feature)
        ffn_feature = self.ffn(norm_feature)
        residual_feature = ffn_feature + residual_copy
        return residual_feature


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, heads_num, channel, pixel_num, qkv_channel, mlp_dim, embedding=True, out=False):
        super(TransformerEncoder, self).__init__()
        self.embedding = embedding
        self.out = out

        self.embeddings = Embeddings(pixel_num, channel)

        self.layer = nn.ModuleList()
        for _ in range(num_layers):
            if self.embedding:
                layer = ResidualLinearAttentionBlock(heads_num, channel, mlp_dim, qkv_channel, pixel_num+1)
            else:
                layer = ResidualLinearAttentionBlock(heads_num, channel, mlp_dim, qkv_channel, pixel_num)
            self.layer.append(copy.deepcopy(layer))
        self.encoder_norm = nn.LayerNorm(channel, eps=1e-6)

    def forward(self, input_cube):
        B, C, H, W = input_cube.shape
        features = input_cube.flatten(2)
        features = features.transpose(-1, -2)

        if self.embedding:
            features = self.embeddings(features)

        for layer_block in self.layer:
            features = layer_block(features)
        features = self.encoder_norm(features)

        if self.embedding:
            if self.out:
                features = features[:, 0, :]
            else:
                features = features[:, 1:, :]
                features = features.transpose(-1, -2).reshape(B, C, H, W)
        else:
            features = features.transpose(-1, -2).reshape(B, C, H, W)
        return features


class overall(nn.Module):
    def __init__(self, num_classes,
                 num_bands,
                 patch_size,
                 embed_dim=512,
                 channel1=32,
                 channel2=32,
                 channel3=32,
                 channel4=32,
                 ):
        super(overall, self).__init__()
        self.n_bands = num_bands
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.layer1 = Bottleneck(self.n_bands, channel1, 2)
        self.layer2 = Bottleneck(channel1, channel2, 1)
        self.layer3 = Bottleneck(channel2, channel3, 2)
        self.layer4 = Bottleneck(channel3, channel4, 1)
        # self.fc = nn.Linear(in_features=self._get_layer_size(), out_features=embed_dim, bias=False)
        self.classifier = nn.Linear(in_features=self._get_layer_size(), out_features=self.num_classes, bias=False)

        self.transformer = TransformerEncoder(num_layers=1, heads_num=2, channel=32, pixel_num=9, qkv_channel=[16, 32],
                                               mlp_dim=32, embedding=True, out=False)
        # self.att_classifier = nn.Linear(in_features=32, out_features=self.num_classes, bias=False)

    def forward(self, x):
        x = x.squeeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.transformer(out)

        out = out.reshape(out.shape[0], -1)

        # feature = self.fc(out)
        pred = self.classifier(out)

        # return pred, feature
        return pred

    def _get_layer_size(self):
        with torch.no_grad():
            x1 = torch.zeros((1, self.n_bands, self.patch_size, self.patch_size))
            out = self.layer1(x1)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.shape[0], -1)
            s = out.size()[1]
        return s


def ACB(dataset, patch_size, pca=False):

    if dataset == 'sa':
        n_bands=204
        num_classes=16
    elif dataset == 'pu':
        n_bands=103
        num_classes=9
    elif dataset == 'whulk':
        n_bands=270
        num_classes=9
    elif dataset == 'whuhh':
        n_bands=270
        num_classes=22
    elif dataset == 'hrl':
        n_bands=176
        num_classes=14
    elif dataset == 'IP':
        n_bands=200
        num_classes=16
    elif dataset == 'whuhc':
        n_bands=274
        num_classes=16
    elif dataset == 'BS':
        n_bands=145
        num_classes=14
    elif dataset == 'HsU':
        n_bands=144
        num_classes=15
    elif dataset == 'KSC':
        n_bands=176
        num_classes=13
    elif dataset == 'pc':
        n_bands=102
        num_classes=9
    else:
        warnings.warn(f"Unsupported dataset: {dataset}. Returning None model.")
        return None

    model = overall(num_classes, n_bands, patch_size)

    return model


if __name__ == '__main__':
    t = torch.randn(size=(64, 1, 204, 9, 9))
    dataset = 'sa'
    print("input shape:", t.shape)
    net = ACB(dataset=dataset, patch_size=9, pca=True)
    print("output shape:", net(t).shape)

    with torch.no_grad():
        sum = summary(net, input_size=(1, 1, t.shape[2], t.shape[-2], t.shape[-1]), verbose=0)
        print(sum)


    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # net.eval()
    # flops = FlopCountAnalysis(net, t)
    # print(flop_count_table(flops))
    #
    # from thop import profile, clever_format
    # flops, params = profile(net, inputs=(t,))
    # macs, params = clever_format([flops, params], "%.3f")
    # print(macs, params)



