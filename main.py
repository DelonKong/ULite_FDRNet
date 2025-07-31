import argparse
import time
from datetime import timedelta
import numpy as np
import torch.nn as nn
import torch.utils.data
from einops import rearrange
from sklearn.decomposition import PCA
from torchinfo import summary
from thop import profile
from thop import clever_format
from fvcore.nn import FlopCountAnalysis, flop_count_table
from models.DSNet.train_DSNet import train_DSNet, test_DSNet
from models.S2VNet.train_S2VNet import train_S2VNet, test_S2VNet
from models.groupedSpectralSimilarity import grouped_spectral_similarity
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.dataset2 import sample_gt_fixed
from utils.gpu_info import gpu_info
from utils.utils import split_info_print, metrics, show_results, create_logger
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test
import torch

if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='A2S2K')
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


    # =========================================
    # * data params
    parser.add_argument("--dataset_name", type=str, default="whulk")  # pu sa whuhh whulk
    # pc pu IP whuhc whuhh whulk hrl sa BS HsU KSC
    # =========================================
    # 1.5%: 0.03
    # 1.25%: 0.025
    # 1%： 0.02
    # 0.75%: 0.015
    # 0.50%： 0.01
    # 0.25%： 0.005

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

    # =========================================
    # * model params
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

    # others
    parser.add_argument("--topk", type=int, default=3)

    # =========================================
    # * Finetuning params
    parser.add_argument('--finetune', default=None,
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # =========================================
    # * others
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")

    opts = parser.parse_args()

    gpu_info()


    log_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/logs'
    logger = create_logger(log_dir)
    logger.info("Running model: {}".format(opts.model))
    if opts.device == "0":
        device = torch.device("cuda:{}".format(opts.device))
        # torch.cuda.set_device(0)
        logger.info("experiments will run on GPU device {}".format(opts.device))
    else:
        device = torch.device("cpu")
        logger.info("experiments will run on CPU")
    logger.info("Dataset name = {}".format(opts.dataset_name))
    logger.info("total epoch = {}".format(opts.epoch))
    logger.info("total runs = {}\n".format(opts.num_run))
    if opts.is_ratio:
        logger.info("{} for training and validation and {} testing".format(opts.ratio, 1 - opts.ratio))
    else:
        logger.info("{} for training and {} for validation".format(opts.train_num_per_class, opts.test_num_per_class))
    logger.info("data_aug = {}\n".format(opts.data_aug))

    # print parameters
    logger.info("batch size = {}".format(opts.bs))
    logger.info("patch_size = {}\n".format(opts.patch_size))

   
    if opts.model == "ULite_FDRNet":
        if opts.dataset_name == "pu":
            opts.embed_dim = 40
            opts.num_heads = 4
        elif opts.dataset_name == "sa":
            opts.embed_dim = 40
        elif opts.dataset_name == "whuhh":
            opts.embed_dim = 64
        elif opts.dataset_name == "whulk":
            opts.embed_dim = 16
        logger.info("n_components = {}\n".format(opts.n_components))
        logger.info("embed_dim = {}".format(opts.embed_dim))
        logger.info("out_channels_3d = {}\n".format(opts.out_channels_3d))
        logger.info("num_heads = {}".format(opts.num_heads))
        logger.info("mlp_ratios = {}".format(opts.mlp_ratios))
        logger.info("act_layer = {}\n".format(opts.act_layer))
    elif opts.model == "SSFTT":
        logger.info("n_components = {}\n".format(opts.n_components))
    elif opts.model == "CSCANet" or opts.model == "CSCANet_FDR":
        opts.n_components = 30
        logger.info("n_components = {}\n".format(opts.n_components))
    else:
        opts.n_components = 0
        opts.grouped_spectral_similarity = False


    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir, opts.grouped_spectral_similarity,
                                     opts.threshold, opts.group_size, opts.n_components)

    num_classes = len(labels)
    num_bands = image.shape[-1]
    height = image.shape[0]
    width = image.shape[1]
    num_pixels = image.shape[0] * image.shape[1]
    logger.info("{} dataset: The numbands is: {}, num_classes is: {}, h: {}, w:{}, num_pixels is: {}"
                .format(opts.dataset_name, num_bands, num_classes, image.shape[0], image.shape[1], num_pixels))

    # random seeds
    seeds = [202201, 202202, 202203, 202204, 202205]

    # empty list to storing results
    results = []

    total_trainTime = 0
    total_testTime = 0
    for run in range(opts.num_run):
        logger.info("\n# ==================================================================== #")

        np.random.seed(seeds[run])
        logger.info("run {} / {}".format(run + 1, opts.num_run))

        if opts.is_ratio:
            # ================================================================
            # get train_gt, val_gt and test_gt with ratio per class
            trainval_gt, test_gt = sample_gt(gt, opts.ratio, seeds[run])  # 1-opts.ratio的测试集，default=0.9
            train_gt, val_gt = sample_gt(trainval_gt, 0.5, seeds[run])  # 1%的训练集
            del trainval_gt

            train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=opts.data_aug)
            val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)

            train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=False, shuffle=False)
        else:
            # ================================================================
            # get train_gt, val_gt and test_gt with fixed numbers per dataset
            train_gt, val_gt, test_gt = sample_gt_fixed(gt, opts.train_num_per_class, opts.test_num_per_class)

            train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=opts.data_aug)
            val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)
            test_set = HSIDataset(image, test_gt, patch_size=opts.patch_size, data_aug=False)

            train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=False, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_set, opts.bs, drop_last=False, shuffle=False)
            # ================================================================

        # load model and loss
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

        model = model.to(device)
        # model = model.cpu()
        # model = model.cuda()

        if run == 0:
            train_class_num, val_class_num, test_class_num = split_info_print(train_gt, val_gt, test_gt, labels)
            # 打印表头
            logger.info("{:<10} {:<10} {:<10} {:<10}".format("class", "train", "val", "test"))

            # 打印每个类别的数量
            for i in range(len(labels)):
                logger.info("{:<10} {:<10} {:<10} {:<10}".format(
                    labels[i], train_class_num[i], val_class_num[i], test_class_num[i]
                ))

            print("network information:")
            with torch.no_grad():
                sum = summary(model, input_size=(1, 1, image.shape[-1], opts.patch_size, opts.patch_size), verbose=0)
                logger.info(sum)
                """
                打印出中间层张量的摘要信息: show_input=True, show_output=True
                显示模型的输出的内存使用情况： input_data=True
                """
            # # =================计算模型macs, params
            input = torch.randn(1, 1, image.shape[-1], opts.patch_size, opts.patch_size).to(device)
            # input = input.to(device)
            # flops, params = profile(model, inputs=(input,))
            # macs, params = clever_format([flops, params], "%.3f")
            # logger.info("Model Macs: {}, Params: {}".format(macs, params))
            #
            # model.eval()
            # flops = FlopCountAnalysis(model, input)
            # logger.info(flop_count_table(flops))

            from fvcore.nn import FlopCountAnalysis, flop_count_table
            model.eval()
            flops = FlopCountAnalysis(model, input)
            logger.info(flop_count_table(flops))

        optimizer, scheduler = load_scheduler(opts.model, model)

        criterion = nn.CrossEntropyLoss()

        # where to save checkpoint model
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run)

        # ===========================================================================================
        # load checkpoins if finetuning
        if opts.finetune:
            checkpoint = torch.load(opts.finetune, map_location='cpu')

            print("Load ckpt from %s" % opts.finetune)
            checkpoint_model = None
            for model_key in opts.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # # pop unnecessary key
            # checkpoint_model.pop('SpectralEmbedding.masked_embed', None)
            # checkpoint_model.pop('SpatialEmbedding.masked_embed', None)

            # utils.load_state_dict(model, checkpoint_model, prefix=opts.model_prefix)
            # model.load_state_dict(checkpoint_model, strict=False)
            model.extractor.load_state_dict(checkpoint_model['extractor'], strict=False)
            model.vit.load_state_dict(checkpoint_model['student'], strict=False)
            # model.vit.load_state_dict(checkpoint_model['teacher'], strict=False)
            # for param in model.extractor.parameters():
            #     param.requires_grad = False
            for param in model.vit.parameters():
                param.requires_grad = False
        # ===========================================================================================

        try:
            print("\n###################\n")
            print(next(model.parameters()).device)
            start = time.perf_counter()
            if opts.model == 'S2VNet':
                train_S2VNet(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
            elif opts.model == 'DSNet':
                train_DSNet(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
            else:
                train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
            end = time.perf_counter()
            runTime = end - start
            total_trainTime += runTime
            runTime_ms = runTime * 1000
            formatted_time = str(timedelta(seconds=runTime))
            logger.info("\n Training time: {} s".format(runTime))
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')

        logger.info("# ==================================================================== #")
        # test the model
        start = time.perf_counter()
        if opts.model == 'S2VNet':
            probabilities = test_S2VNet(model, model_dir, image, opts.patch_size, num_classes, device, batch_size=opts.bs)
        elif opts.model == 'DSNet':
            probabilities = test_DSNet(model, model_dir, image, opts.patch_size, num_classes, device, batch_size=opts.bs)
        else:
            probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device, batch_size=opts.bs)
        end = time.perf_counter()
        runTime = end - start
        total_testTime += runTime
        logger.info("Inference time: {} s \n".format(runTime))

        prediction = np.argmax(probabilities, axis=-1)

        # computing metrics
        run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
        results.append(run_results)
        text = show_results(run_results, label_values=labels)
        logger.info(text)
        logger.info("# ==================================================================== #")

        del model, train_set, train_loader, val_set, val_loader

    if opts.num_run > 1:
        text = show_results(results, label_values=labels, agregated=True)
        logger.info("# Agregated results: ================================================= #")
        logger.info(text)
        logger.info("# ==================================================================== #")

    runTime = total_trainTime
    runTime_ms = runTime * 1000
    formatted_time = str(timedelta(seconds=runTime))
    logger.info("\n# =========================== ")
    logger.info("Total {} Training time: {}".format(opts.num_run, formatted_time))
    logger.info("Total {} Training time: {} s".format(opts.num_run, runTime))
    logger.info("Total {} Training time: {} ms".format(opts.num_run, runTime_ms))
    logger.info("# =========================== \n")

    runTime = total_testTime
    runTime_ms = runTime * 1000
    formatted_time = str(timedelta(seconds=runTime))
    logger.info("\n# =========================== ")
    logger.info("Total {} Inference time: {}".format(opts.num_run, formatted_time))
    logger.info("Total {} Inference time: {} s".format(opts.num_run, runTime))
    logger.info("Total {} Inference time: {} ms".format(opts.num_run, runTime_ms))
    logger.info("# =========================== \n")
