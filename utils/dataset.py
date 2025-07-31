import argparse
from collections import Counter
from collections import defaultdict
import random
from einops import rearrange
from scipy import io
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

"""
按每个类别固定比例划分训练集、测试集、验证集。
"""

def load_mat_hsi(dataset_name, dataset_dir, spectral_similarity=False,
                 threshold=0.8, group_size=8, n_components=0, norm=True, mean=True):
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

    if norm:
        # step2 normalise the HSI data (method from SSAN, TGRS 2020)
        # 全局归一化，会使整个图像的数据分布压缩到[0, 1]
        image = np.asarray(image, dtype=np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    if mean:
        # 均值去中心化， 会使每个通道的分布均值为0
        mean_by_c = np.mean(image, axis=(0, 1))
        for c in range(image.shape[-1]):
            image[:, :, c] = image[:, :, c] - mean_by_c[c]

    # step3 set undefined index 0 to -1, so class index starts from 0
    gt = gt.astype('int') - 1

    # step4 remove undefined label
    labels = labels[1:]

    num_bands = image.shape[-1]
    height = image.shape[0]
    width = image.shape[1]
    # =======================PCA=========================
    if n_components != 0 and spectral_similarity==False:
        hsi_reshaped = image.reshape(-1, num_bands)
        # hsi_mean = np.mean(hsi_reshaped, axis=0)
        # hsi_std = np.std(hsi_reshaped, axis=0)
        # hsi_normalized = (hsi_reshaped - hsi_mean) / hsi_std
        pca = PCA(n_components=n_components)
        hsi_pca = pca.fit_transform(hsi_reshaped)
        # [height, width, n_components]
        image = hsi_pca.reshape(height, width, n_components)


    return image, gt, labels


def sample_gt(gt, percentage, seed):
    """
    Splitting train and test dataset
    :param gt: 2d int array, -1 for undefined or not selected, index starts at 0
    :param percentage: for example, 0.1 for 10%, 0.02 for 2%, 0.5 for 50%
    :param seed: random seed
    :return: train_gt, test_gt
    """

    # 比如，gt =
    # [[0 0]
    #  [1 1]]

    # np.where 中只有条件信息condition，返回行位置信息和列位置信息
    # 比如，(array([0, 0, 1, 1], dtype=int64), array([0, 1, 0, 1], dtype=int64))
    indices = np.where(gt >= 0)

    # X表示位置信息，比如 [(0, 0), (0, 1), (1, 0), (1, 1)]
    X = list(zip(*indices))

    # 将符合位置信息的真值标签展开
    # 比如，[0 0 1 1]
    y = gt[indices].ravel()

    train_gt = np.full_like(gt, fill_value=-1)
    test_gt = np.full_like(gt, fill_value=-1)

    train_indices, test_indices = train_test_split(
        X,
        train_size=percentage,
        random_state=seed,
        stratify=y  # 保持测试集与整个数据集里result的数据分类比例一致。
    )

    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]

    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    """
    train_gt:
    如果该位置在测试集中，则标签值为-1；
    如果该位置在训练集中，则标签值等于输入数组gt在该位置上的标签值。
    """
    return train_gt, test_gt



class HSIDataset(torch.utils.data.Dataset):
    def __init__(self, image, gt, patch_size, data_aug=False):
        """
        :param image: 3d float np array of HSI, image
        :param gt: train_gt or val_gt or test_gt
        :param patch_size: 7 or 9 or 11 ...
        :param data_aug: whether to use data augment, default is True
        """
        super().__init__()
        self.data_aug = data_aug
        self.patch_size = patch_size

        # 为了避免因为卷积运算导致输出图像缩小和图像边缘信息丢失，常常采用图像边缘填充技术
        self.ps = self.patch_size // 2  # padding size
        self.data = np.pad(image, ((self.ps, self.ps), (self.ps, self.ps), (0, 0)), mode='reflect')
        self.label = np.pad(gt, ((self.ps, self.ps), (self.ps, self.ps)), mode='reflect')

        mask = np.ones_like(self.label)
        mask[self.label < 0] = 0
        x_pos, y_pos = np.nonzero(mask)

        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)
                                 if self.ps <= x < image.shape[0] + self.ps
                                 and self.ps <= y < image.shape[1] + self.ps])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def hsi_augment(self, data):
        # e.g. (7 7 200) data = numpy array float32
        do_augment = np.random.random()
        if do_augment > 0.5:
            prob = np.random.random()
            if 0 <= prob <= 0.2:
                data = np.fliplr(data)
            elif 0.2 < prob <= 0.4:
                data = np.flipud(data)
            elif 0.4 < prob <= 0.6:
                data = np.rot90(data, k=1)
            elif 0.6 < prob <= 0.8:
                data = np.rot90(data, k=2)
            elif 0.8 < prob <= 1.0:
                data = np.rot90(data, k=3)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # 中心像素的位置
        x, y = self.indices[i]
        # patch的起止位置
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]

        # 以中心像素作标签
        label = self.label[x, y]

        if self.data_aug:
            # Perform data augmentation (only on 2D patches)
            data = self.hsi_augment(data)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Add a fourth dimension for 3D CNN
        data = data.unsqueeze(0)

        return data, label


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run HSI datasets loading...")
    parser.add_argument("--dataset_name", type=str, default="pc")
    # pu IP whuhc whuhh whulk hrl sa HsU KSC BS

    # pc dataset: The numbands is: 102, num_classes is: 9, h: 1096, w:715, num_pixels is: 783640, spatial resolution: 1.3 m
    # pu dataset: The numbands is: 103, num_classes is: 9, h: 610, w:340, num_pixels is: 207400, spatial resolution: 1.3 m
    # IP dataset: The numbands is: 200, num_classes is: 16, h: 145, w:145, num_pixels is: 21025, spatial resolution: 20 m
    # whuhc dataset: The numbands is: 274, num_classes is: 16, h: 1217, w:303, num_pixels is: 368751, spatial resolution: 0.109 m
    # whuhh dataset: The numbands is: 270, num_classes is: 22, h: 940, w:475, num_pixels is: 446500, spatial resolution: 0.043
    # whulk dataset: The numbands is: 270, num_classes is: 9, h: 550, w:400, num_pixels is: 220000, spatial resolution: 0.463 m
    # hrl dataset: The numbands is: 176, num_classes is: 14, h: 249, w:945, num_pixels is: 235305, spatial resolution:
    # sa dataset: The numbands is: 204, num_classes is: 16, h: 512, w:217, num_pixels is: 111104, spatial resolution: 3.7 m
    # HsU dataset: The numbands is: 144, num_classes is: 15, h: 349, w:1905, num_pixels is: 664845, spatial resolution: 2.5 m
    # KSC dataset: The numbands is: 176, num_classes is: 13, h: 512, w:614, num_pixels is: 314368, spatial resolution: 18 m
    # BS dataset: The numbands is: 145, num_classes is: 14, h: 1476, w:256, num_pixels is: 377856, spatial resolution: 30 m

    parser.add_argument("--dataset_dir", type=str, default="../datasets")

    opts = parser.parse_args()

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands = image.shape[-1]
    num_pixels = image.shape[0] * image.shape[1]
    print("{} dataset: The numbands is: {}, num_classes is: {}, h: {}, w:{}, num_pixels is: {}"
          .format(opts.dataset_name, num_bands, num_classes, image.shape[0], image.shape[1], num_pixels))

