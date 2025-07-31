import os
import time
import torch
import argparse
import seaborn as sns
import numpy as np

from models.DSNet.train_DSNet import test_DSNet
from models.S2VNet.train_S2VNet import test_S2VNet
from utils.dataset import load_mat_hsi
from models.get_model import get_model
from train import test
from utils.utils import metrics, show_results, create_logger
import imageio


def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSI classification evaluation")
    # =========================================
    parser.add_argument("--model", type=str, default='LRDTN')
    parser.add_argument("--weights", type=str, default="./checkpoints/LRDTN/whulk/3")
    # =============================== Non Lite:===============================
    # A2S2K(TGRS 2021), SSFTT(TGRS 2022), morphFormer(TGRS 2023), LRDTN(TGRS 2024), CSCANet(TIP 2025, pca=30), S2VNet(TGRS 2025)
    # =============================== lite:===============================
    # LS2CM(LGRS 2021), ELS2T(TGRS 2023), LMSS_NAS(TGRS 2023), CLOLN(TGRS 2024), ACB(TGRS 2024)

    # * data params
    parser.add_argument("--dataset_name", type=str, default="whulk")
    # pc pu IP whuhc whuhh whulk hrl sa BS HsU KSC

    parser.add_argument("--n_components", type=int, default=30)  # PCA

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

    parser.add_argument("--act_layer", type=str, default='gelu')

    # others
    parser.add_argument("--topk", type=int, default=3)

    # =========================================
    # * others
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--outputs", type=str, default="./results")
    opts = parser.parse_args()

    outfile = os.path.join(opts.outputs, opts.dataset_name,  opts.model)
    os.makedirs(outfile, exist_ok=True)
    logger = create_logger(outfile)

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
    elif opts.model == "CSCANet":
        opts.n_components = 30
        logger.info("n_components = {}\n".format(opts.n_components))
    else:
        opts.n_components = 0


    device = torch.device("cuda:{}".format(opts.device))
    # device = torch.device("cpu")

    print("dataset: {}".format(opts.dataset_name))
    print("patch size: {}".format(opts.patch_size))
    print("model: {}".format(opts.model))

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

    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes + 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))

    # load model and weights
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
                      mode='inference'
                      )

    logger.info('loading weights from %s' % opts.weights + '/model_best.pth')
    model = model.to(device)
    # model.load_state_dict(torch.load(os.path.join(opts.weights, 'model_best.pth')), strict=False)
    # model.eval()

    # testing model: metric for the whole HSI, including train, val, and test
    logger.info("# ==================================================================== #")
    # test the model
    start = time.perf_counter()
    if opts.model == 'S2VNet':
        probabilities = test_S2VNet(model, opts.weights, image, opts.patch_size, num_classes, device, batch_size=opts.bs)
    elif opts.model == 'DSNet':
        probabilities = test_DSNet(model, opts.weights, image, opts.patch_size, num_classes, device, batch_size=opts.bs)
    else:
        probabilities = test(model, opts.weights, image, opts.patch_size, num_classes, device, batch_size=opts.bs)
    prediction = np.argmax(probabilities, axis=-1)
    end = time.perf_counter()
    runTime = end - start
    logger.info("Inference time: {} s \n".format(runTime))


    run_results = metrics(prediction, gt, n_classes=num_classes)

    prediction[gt < 0] = -1

    # color results
    colored_pred = color_results(prediction+1, palette)

    outfile = os.path.join(opts.outputs, opts.dataset_name,  opts.model)
    os.makedirs(outfile, exist_ok=True)

    imageio.imsave(os.path.join(outfile, opts.dataset_name+'_' + opts.model + '_out.png'), colored_pred)  # or png

    res = show_results(run_results, label_values=labels)
    logger.info(res)

    if opts.model == "ULite_TAA":
        colored_gt = color_results(gt + 1, palette)
        imageio.imsave(os.path.join(outfile, opts.dataset_name + '_gt.png'), colored_gt)  # eps or png

    del model
