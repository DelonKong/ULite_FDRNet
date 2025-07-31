# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import sys

import numpy as np
import torch.nn as nn
import torch.utils.data
from torchinfo import summary
from thop import profile
from thop import clever_format
from tqdm import tqdm
from wandb import Settings

from loss import SMLoss, FocalLoss
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.gpu_info import gpu_info
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test, validation, save_checkpoint
import wandb
import yaml

wandb.login(key='a75b9250aebfa283a14c493104d5287dd998a153', relogin=True)

parser = argparse.ArgumentParser(description="run patch-based HSI classification with wandb")
parser.add_argument("--num_run", type=int, default=1)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--device", type=str, default="0")
parser.add_argument("--dataset_dir", type=str, default="../datasets")

opts = parser.parse_args()

device = torch.device("cuda:{}".format(opts.device))

# 'method': 'grid', 'random'
sweep_config = {
    # 'name': 'patchSize',
    'method': 'grid'
}
metric = {
    'name': 'best_acc', 'goal': 'maximize'
}
sweep_config['metric'] = metric

sweep_config['parameters'] = {}

# 'lr': {'value': 0.001},
# 固定不变的超参
sweep_config['parameters'].update({
    # 'dataset_name': {'value': "whulk"},
    # 'model': {'value': "ULite_TAA"},

    'bs': {'value': 64},
    # 'ratio': {'value': 0.02},
    # 'ratio': {'value': 0.01},

    'n_components': {'value': 0},
    # 'n_components': {'value': 40},
    # 'embed_dim': {'value': 56},

    'patch_size': {'value': 9},

    'out_channels_3d': {'value': 4},
    'num_heads': {'value': 2},

    'ratio_2d': {'value': 2},
    'ratio_3d': {'value': 2},

    'num_extras': {'value': 1},
    'depths_te': {'value': 1},

    'mlp_ratios': {'value': 3},
    'act_layer': {'value': 'gelu'},
    'Scale': {'value': 5},

    'grouped_spectral_similarity': {'value': False},
    'threshold': {'value': 0},
    'group_size': {'value': 0},
    'topk': {'value': 0}
})

# =============================== Non Lite:===============================
# A2S2K(TGRS 2021), SSFTT(TGRS 2022), morphFormer(TGRS 2023), LRDTN(TGRS 2024), CSCANet(TIP 2025, pca=30), S2VNet(TGRS 2025)
# =============================== lite:===============================
# LS2CM(LGRS 2021), ELS2T(TGRS 2023), LMSS_NAS(TGRS 2023), CLOLN(TGRS 2024), ACB(TGRS 2024)
# =============================== others:===============================
# DFFN(TGRS 2018), GhostNet(LGRS 2019), HybridSN(LGRS 2020), GAHT(TGRS 2022), SpectalFormer(TGRS 2022),
# BS2T(TGRS 2022), HybridFormer(TGRS 2023), AMF(TGRS 2023), DSNet(TGRS 2024), MHIAIFormer(JSTARS, 2024)
# "A2S2K", "SSFTT", "morphFormer", "LRDTN", "LS2CM", "ELS2T", "LMSS_NAS", "CLOLN", "ACB"
# CSCANet
# "DFFN", "A2S2K", "HybridFormer", "SpectalFormer", "GAHT", "GhostNet", "SSFTT", "morphFormer", "LRDTN", "CLOLN", "LS2CM", "ELS2T", "LMSS_NAS"
# 离散型分布超参
sweep_config['parameters'].update({
    'model': {
        'values': ["A2S2K", "SSFTT", "morphFormer", "LRDTN", "LS2CM", "ELS2T", "LMSS_NAS", "CLOLN", "ACB"]
    },

    'dataset_name': {
        'values': ['pu', 'sa', 'whuhh', 'whulk']
    },

    # 'n_components': {
    #     'values': [16, 24, 32, 48, 64]
    # },
    # 'embed_dim': {
    #     'values': [24, 32, 48, 64, 96, 128]
    # },
    # 'n_components': {
    #     'values': [10, 15, 20, 25, 30, 35, 40, 45, 50]
    # },
    # 'embed_dim': {
    #     'values': [16, 24, 32, 40, 48, 56, 64, 72, 80]
    # },
    # 'n_components': {
    #     'values': [10, 15, 20, 25, 30, 35]
    # },
    # 'embed_dim': {
    #     'values': [48, 56]
    # },

    # 'patch_size': {
    #     'values': [5, 7, 9, 11, 13, 15]
    # },

    # 'out_channels_3d': {
    #     'values': [4, 6, 8, 10, 12]
    # },
    # 'num_heads': {
    #     'values': [2, 4, 8, 12, 16]
    # },

    # 'Scale': {
    #     'values': [1, 3, 5, 7, 9]
    # },

    # 'mlp_ratios': {
    #     'values': [1, 2, 3, 4, 5, 6, 7, 8]
    # },

    # 'num_extras': {
    #     'values': [1, 2, 3, 4, 5]
    # },
    # 'depths_te': {
    #     'values': [1, 2, 3, 4, 5]
    # },

    'ratio': {
        'values': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    },

    # 'ratio_2d': {
    #     'values': [2, 4, 8, 12, 16]
    # },
    # 'ratio_3d': {
    #     'values': [2, 4, 8, 12, 16]
    # },

    # 'topk': {
    #     'values': [1, 2, 3, 4, 5, 6]
    # },
    #
    # 'group_size': {
    #     'values': [4, 8, 16, 32]
    # },
    # 'threshold': {
    #     'values': [0.6, 0.7, 0.8, 0.9]
    # },
})

# # 连续型分布超参
# sweep_config['parameters'].update({
#     'lr': {
#         'distribution': 'log_uniform_values',
#         'min': 0.0001,
#         'max': 0.1
#     },
# })



def train_wandb(config):
    # Print the current hyperparameter values
    current_hyperparams = config
    print("Current Hyperparameters:")
    for key, value in current_hyperparams.items():
        print(f"{key}: {value}")

    # if config.topk > config.out_channels_3d:
    #     print(f"Skipping combination: out_channels_3d={config.out_channels_3d}, topk={config.topk}")
    #     wandb.finish()  # Cleanly end this run in wandb
    #     sys.exit()  # Exit the current grid search iteration

    if config.dataset_name == 'whulk':
        embed_dim = 16
    elif config.dataset_name == 'whuhh':
        embed_dim = 64
    else:
        embed_dim = 40

    if config.dataset_name == 'pu':
        num_heads = 4
    else:
        num_heads = config.num_heads

    # load data
    image, gt, labels = load_mat_hsi(config.dataset_name, opts.dataset_dir, config.grouped_spectral_similarity,
                                     config.threshold, config.group_size, config.n_components)

    num_classes = len(labels)
    num_bands = image.shape[-1]
    wandb.log({'num_bands': num_bands})
    num_pixels = image.shape[0] * image.shape[1]
    print("{} dataset: The numbands is: {}, num_classes is: {}, h: {}, w:{}, num_pixels is: {}"
          .format(config.dataset_name, num_bands, num_classes, image.shape[0], image.shape[1], num_pixels))

    # get train_gt, val_gt and test_gt
    # trainval_gt, test_gt = sample_gt(gt, ratio, 202307)  # 1-ratio*2的测试集，default=0.9
    trainval_gt, test_gt = sample_gt(gt, config.ratio, 202307)
    train_gt, val_gt = sample_gt(trainval_gt, 0.5, 202307)  # 1%的训练集
    del trainval_gt

    train_set = HSIDataset(image, train_gt, patch_size=config.patch_size, data_aug=True)
    val_set = HSIDataset(image, val_gt, patch_size=config.patch_size, data_aug=False)

    train_loader = torch.utils.data.DataLoader(train_set, config.bs, drop_last=False, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, config.bs, drop_last=False, shuffle=False)

    model = get_model(model_name=config.model,
                      in_chans=image.shape[-1],
                      dataset_name=config.dataset_name,
                      patch_size=config.patch_size,
                      out_channels_3d=config.out_channels_3d,
                      # num_heads=config.num_heads,
                      num_heads=num_heads,
                      threshold=config.threshold,
                      group_size=config.group_size,
                      # embed_dim=config.embed_dim,
                      embed_dim=embed_dim,
                      mlp_ratios=config.mlp_ratios,
                      act_layer=config.act_layer,
                      num_extras=config.num_extras,
                      depths_te=config.depths_te,
                      ratio_2d=config.ratio_2d,
                      ratio_3d=config.ratio_3d,
                      topk=config.topk,
                      Scale=config.Scale
                      )

    model = model.to(device)

    # =================计算模型macs, params
    input = torch.randn(1, 1, num_bands, config.patch_size, config.patch_size)
    input = input.to(device)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print("Model Macs: {}, Params: {}".format(macs, params))
    wandb.log({'Macs': macs, 'Params': params})

    # optimizer = torch.optim.__dict__[config.optim_type](params=model.parameters(), lr=config.lr)
    optimizer, scheduler = load_scheduler(config.model, model)
    criterion = nn.CrossEntropyLoss()

    # where to save checkpoint model
    model_dir = "../checkpoints/" + config.model + '/wandb_test'

    best_acc = -0.1
    losses = []

    for e in tqdm(range(1, opts.epoch + 1), desc="training the network"):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if e % 10 == 0 or e == 1:
            mean_losses = np.mean(losses)
            train_info = "train at epoch {}/{}, loss={:.6f}"
            train_info = train_info.format(e, opts.epoch, mean_losses)
            tqdm.write(train_info)
            losses = []
        else:
            losses = []

        val_acc = validation(model, val_loader, device)

        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(model, is_best, model_dir, epoch=e, acc=best_acc)
        wandb.log({'train_loss': loss.item(), 'val_acc': val_acc, 'best_acc': best_acc}, step=e)

        if e == opts.epoch:
            # test the model
            probabilities = test(model, model_dir, image, config.patch_size, num_classes, device, config.bs)

            prediction = np.argmax(probabilities, axis=-1)

            # computing metrics
            run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
            show_results(run_results, label_values=labels)
            wandb.log({'OA': run_results["Accuracy"], 'AA': run_results['AA'], 'Kappa': run_results["Kappa"]})

            del model, train_set, train_loader, val_set, val_loader
        # wandb.finish()


def main():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 如果你有代理
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    wandb.init(
        project='ULite_FDR_percentage',
        name=nowtime,
        save_code=False,
        settings=Settings(init_timeout=300)
    )# , mode="offline")

    train_wandb(wandb.config)
    # wandb.sync()
    # wandb.finish()


# # 中断后继续运行：
# # 加载 sweep_resume.yaml 文件中的 sweep 配置
# sweep_id = "patch_size_IP"
# resume_path = wandb.restore("sweep_resume.yaml", run_path="qdu/MyHSIC_3/"+sweep_id, replace=True)
# sweep_config = wandb.sweep.load_config(resume_path)

print(sweep_config)
sweep_id = wandb.sweep(sweep_config, project='ULite_FDR_percentage')
wandb.agent(sweep_id, function=main)
