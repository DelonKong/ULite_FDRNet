import argparse

from scipy import io
import os
import scipy.io as sio
import scipy.io
import numpy as np
import tifffile
from spectral.io.envi import save_image

def load_mat_hsi(dataset_name, dataset_dir):
    """ load HSI.mat dataset """
    # available sets
    available_sets = [
        'sa',
        'pu',
        'whulk',
        'hrl',
        'whuhh',
        'whuhc',
        'IP',
        'BS',
        'HsU',
        'KSC',
        'pc'
    ]
    assert dataset_name in available_sets, "dataset should be one of" + ' ' + str(available_sets)

    image = None
    gt = None
    labels = None

    if dataset_name == 'sa':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_corrected.mat"))
        image = image['salinas_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Salinas_gt.mat"))
        gt = gt['salinas_gt']
        labels = [
            "Undefined",
            "Brocoli_green_weeds_1",
            "Brocoli_green_weeds_2",
            "Fallow",
            "Fallow_rough_plow",
            "Fallow_smooth",
            "Stubble",
            "Celery",
            "Grapes_untrained",
            "Soil_vinyard_develop",
            "Corn_senesced_green_weeds",
            "Lettuce_romaine_4wk",
            "Lettuce_romaine_5wk",
            "Lettuce_romaine_6wk",
            "Lettuce_romaine_7wk",
            "Vinyard_untrained",
            "Vinyard_vertical_trellis",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'pu':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU.mat"))
        image = image['paviaU']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "PaviaU_gt.mat"))
        gt = gt['paviaU_gt']
        labels = [
            "Undefined",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Painted metal sheets",
            "Bare Soil",
            "Bitumen",
            "Self-Blocking Bricks",
            "Shadows",
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whulk':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou.mat"))
        image = image['WHU_Hi_LongKou']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_LongKou_gt.mat"))
        gt = gt['WHU_Hi_LongKou_gt']
        labels = [
            'Undefined',
            'Corn',
            'Cotton',
            'Sesame',
            'Broad-leaf soybean',
            'Narrow-leaf soybean',
            'Rice',
            'Water',
            'Roads and houses',
            'Mixed weed',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'hrl':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Loukia.mat"))
        image = image['loukia']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Loukia_GT.mat"))
        gt = gt['loukia_gt']
        labels = [
            'Undefined',
            'Dense Urban Fabric',
            'Mineral Extraction Sites',
            'Non Irrigated Arable Land',
            'Fruit Trees',
            'Olive Groves',
            'Broad-leaved Forest',
            'Coniferous Forest',
            'Mixed Forest',
            'Dense Sclerophyllous Vegetation',
            'Sparce Sclerophyllous Vegetation',
            'Sparcely Vegetated Areas',
            'Rocks and Sand',
            'Water',
            'Coastal Water'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whuhh':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu.mat"))
        image = image['WHU_Hi_HongHu']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HongHu_gt.mat"))
        gt = gt['WHU_Hi_HongHu_gt']
        labels = [
            'Undefined',
            'Red roof',
            'Road',
            'Bare soil',
            'Cotton',
            'Cotton firewood',
            'Rape',
            'Chinese cabbage',
            'Pakchoi',
            'Cabbage',
            'Tuber mustard',
            'Brassica parachinensis',
            'Brassica chinensis',
            'Small Brassica chinensis',
            'Lactuca sativa',
            'Celtuce',
            'Film covered lettuce',
            'Romaine lettuce',
            'Carrot',
            'White radish',
            'Garlic sprout',
            'Broad bean',
            'Tree'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'whuhc':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HanChuan.mat"))
        image = image['WHU_Hi_HanChuan']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "WHU_Hi_HanChuan_gt.mat"))
        gt = gt['WHU_Hi_HanChuan_gt']
        labels = [
            'Undefined',
            'Strawberry',
            'Cowpea',
            'Soybean',
            'Sorghum',
            'Water spinach',
            'Watermelon',
            'Greens',
            'Trees',
            'Grass',
            'Red roof',
            'Gray roof',
            'Plastic',
            'Bare soil',
            'Road',
            'Bright object',
            'Water',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'IP':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_corrected.mat"))
        image = image['indian_pines_corrected']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Indian_pines_gt.mat"))
        gt = gt['indian_pines_gt']
        labels = [
            'Undefined',
            'Alfalfa',
            'Corn-notill',
            'Corn-mintill',
            'Corn',
            'Grass-pasture',
            'Grass-trees',
            'Grass-pasture-mowed',
            'Hay-windrowed',
            'Oats',
            'Soybean-notill',
            'Soybean-mintill',
            'Soybean-clean',
            'Wheat',
            'Woods',
            'Buildings-Grass-Tress-Drives',
            'Stone-Stell-Towerss',
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'BS':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Botswana.mat"))
        image = image['Botswana']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Botswana_gt.mat"))
        gt = gt['Botswana_gt']
        labels = [
            'Undefined',
            'Water',
            'Hippo grass',
            'Floodplain grasses1',
            'Floodplain grasses2',
            'Reeds1',
            'Riparian',
            'Firescar2',
            'Island interior',
            'Acacia woodlands',
            'Acacia shrublands',
            'Acacia grasslands',
            'Short mopane',
            'Mixed mopane',
            'Exposed soils'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0
    elif dataset_name == 'HsU':
        # Houston University
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "houston.mat"))
        image = image['hsi']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "houston_gt_sum.mat"))
        gt = gt['a']
        labels = [
            'Undefined',
            'Healthy Grass',
            'Stressed Grass',
            'Syntheic Grass',
            'Trees',
            'Soil',
            'Water',
            'Residential',
            'Commercial',
            'Road',
            'Highway',
            'Railway',
            'Parking Lot 1',
            'Parking Lot 2',
            'Tennis Court',
            'Running Track'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'KSC':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "KSC.mat"))
        image = image['KSC']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "KSC_gt.mat"))
        gt = gt['KSC_gt']
        labels = [
            'Undefined',
            'Scrub',
            'Willow swamp',
            'CP hammock',
            'Slash pine',
            'Oak/Broadleaf',
            'Hardwood',
            'Swamp',
            'Graminoid marsh',
            'Saprtina marsh',
            'Cattail marsh',
            'Salt marsh',
            'Mud flats',
            'Water'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    elif dataset_name == 'pc':
        image = io.loadmat(os.path.join(dataset_dir, dataset_name, "Pavia.mat"))
        image = image['pavia']
        gt = io.loadmat(os.path.join(dataset_dir, dataset_name, "Pavia_gt.mat"))
        gt = gt['pavia_gt']
        labels = [
            'Undefined',
            'Water',
            'Trees',
            'Asphalt',
            'Self-Blocking Bricks',
            'Bitumen',
            'Tiles',
            'Shadows',
            'Meadows',
            'Bare Soil'
        ]
        rgb_bands = [0, 1, 2]  # to be edited
        undefined_label_index = 0

    # after getting image and ground truth (gt), let us do data preprocessing!
    # step1 filter nan values out
    nan_mask = np.isnan(image.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print("warning: nan values found in dataset {}, using 0 replace them".format(dataset_name))
        image[nan_mask] = 0
        gt[nan_mask] = 0

    # step2 normalise the HSI data (method from SSAN, TGRS 2020)
    # image = np.asarray(image, dtype=np.float32)
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # mean_by_c = np.mean(image, axis=(0, 1))
    # for c in range(image.shape[-1]):
    #     image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # step3 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step4 remove undefined label
    labels = labels[1:]

    return image, gt, labels

if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="get HSI 3D Cube")
    parser.add_argument("--dataset_name", type=str, default="IP")
    opts = parser.parse_args()

    save_dir = r"F:\KDL\pycharmProjects\MyHSIC2\datasets" + "/" + opts.dataset_name
    save_path = opts.dataset_name + ".tif"
    save_path = os.path.join(save_dir, save_path)


    image, gt, labels = load_mat_hsi(opts.dataset_name, "../datasets")

    # 获取数据维度
    num_rows, num_cols, num_bands = image.shape

    # # 将数据转换为 ENVI 能识别的格式：波段作为第一个维度
    # tif_data = np.moveaxis(image, -1, 0)  # 将最后一个维度(波段)移到最前面
    # print(f"转换后的数据形状: {tif_data.shape}")

    tif_data = image

    # 保存为 TIFF 文件
    # tifffile.imwrite(save_path, tif_data.astype(np.uint16), planarconfig="separate")  # 根据实际数据调整类型

    tifffile.imwrite(
        save_path,
        tif_data,
        metadata={
            'Description': 'ENVI TIFF format',
            'ImageWidth': num_cols,
            'ImageLength': num_rows,
            'SamplesPerPixel': num_bands,
        },
        planarconfig='contig'
    )


    print(f"数据已保存为 TIFF 文件: {save_path}")

    # 加载保存的 TIFF 文件并检查形状
    loaded_tif_data = tifffile.imread(save_path)
    print(f"Loaded TIFF data shape: {loaded_tif_data.shape}")

