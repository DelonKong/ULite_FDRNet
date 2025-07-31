# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import time

from torch import nn

from models.ULite_TAA import ULite_TAA
from models.get_model import get_model
from utils.dataset import load_mat_hsi

if __name__ == '__main__':
    device = torch.device("cuda:{}".format(0))
    dump_input = torch.randn(size=(1, 1, 40, 9, 9)).to(device)


    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='LMSS_NAS')
    # proposed: ULite_FDRNet
    # =============================== Non Lite:===============================
    # A2S2K(TGRS 2021), SSFTT(TGRS 2022), morphFormer(TGRS 2023), LRDTN(TGRS 2024), CSCANet(TIP 2025, pca=30), S2VNet(TGRS 2025)
    # =============================== lite:===============================
    # LS2CM(LGRS 2021), ELS2T(TGRS 2023), LMSS_NAS(TGRS 2023), CLOLN(TGRS 2024), ACB(TGRS 2024)
    # =============================== others:===============================
    # DFFN(TGRS 2018), GhostNet(LGRS 2019), HybridSN(LGRS 2020), GAHT(TGRS 2022), SpectalFormer(TGRS 2022),
    # BS2T(TGRS 2022), HybridFormer(TGRS 2023), AMF(TGRS 2023), DSNet(TGRS 2024), MHIAIFormer(JSTARS, 2024)
    # =============================== REPLACEMENT EXPERIMENTS:===============================
    # A2S2K_FDR, SSFTT_FDR, LRDTN_FDR(pca=100), CSCANet_FDR(pca=30)

    parser.add_argument("--dataset_name", type=str, default="whulk")  # pu sa whuhh whulk

    parser.add_argument("--num_run", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--is_ratio", type=bool, default=True)  # False True
    parser.add_argument("--ratio", type=float, default=0.02)  # percentage: for example
    parser.add_argument("--train_num_per_class", type=int, default=40)  # numbers per class: for example
    parser.add_argument("--test_num_per_class", type=int, default=15)  # numbers per class: for example
    parser.add_argument("--data_aug", type=bool, default=True)  # TrainSet data_aug

    parser.add_argument("--n_components", type=int, default=40)  # PCA

    parser.add_argument("--grouped_spectral_similarity", type=bool, default=False)  # grouped_spectral_similarity
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--bs", type=int, default=64)  # bs = batch size
    parser.add_argument("--patch_size", type=int, default=9)

    parser.add_argument("--out_channels_3d", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=2)

    # pca
    parser.add_argument("--embed_dim", type=int, default=40)

    parser.add_argument("--ratio_2d", type=int, default=2)
    parser.add_argument("--ratio_3d", type=int, default=2)

    parser.add_argument("--mlp_ratios", type=int, default=3)
    parser.add_argument("--num_extras", type=int, default=1)
    parser.add_argument("--depths_te", type=int, default=1)
    parser.add_argument("--Scale", type=int, default=5)

    parser.add_argument("--act_layer", type=str, default='gelu')
    opts = parser.parse_args()

    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir, opts.grouped_spectral_similarity,
                                     opts.threshold, opts.group_size, opts.n_components)

    model = get_model(model_name=opts.model,
                      in_chans=image.shape[-1],
                      dataset_name=opts.dataset_name,
                      patch_size=opts.patch_size,
                      out_channels_3d=opts.out_channels_3d,
                      num_heads=opts.num_heads,
                      threshold=opts.threshold,
                      group_size=opts.group_size,
                      embed_dim=opts.embed_dim,
                      mlp_ratios=opts.mlp_ratios,
                      act_layer=opts.act_layer,
                      num_extras=opts.num_extras,
                      depths_te=opts.depths_te,
                      ratio_2d=opts.ratio_2d,
                      ratio_3d=opts.ratio_3d,
                      topk=opts.topk,
                      mode='train',
                      Scale=opts.Scale
                      )
    model.eval()
    model.to(device)

    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_device = 'cuda', record_shapes=False, profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace('./model_profile.json')

    # 打开 Chrome/Edge 浏览器，在地址栏输入 chrome://tracing
    # 导入json文件
    # 操作：
    # 按键盘w, a, s, d键，可以对profiler的结果进行缩放和移动
