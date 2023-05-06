import pickle
import os
import torch.utils.data as torch_data
import numpy as np
from torch import from_numpy
from PIL import Image
import dataset.transform as T
import dataset.transform_cv2 as Tcv
import cv2


eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
map_classes = {
    7: "road",  # 1
    8: "sidewalk",  # 2
    9: "parking",
    10: "rail truck",
    11: "building",  # 3
    12: "wall",  # 4
    13: "fence",  # 5
    14: "guard_rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",  # 6
    18: "pole_group",
    19: "light",  # 7
    20: "sign",  # 8
    21: "vegetation",  # 9
    22: "terrain",  # 10
    23: "sky",  # 11
    24: "person",  # 12
    25: "rider",  # 13
    26: "car",  # 14
    27: "truck",  # 15
    28: "bus",  # 16
    29: "caravan",
    30: "trailer",
    31: "train",  # 17
    32: "motorcycle",  # 18
    33: "bicycle"  # 19
}

IMAGES_DIR = os.path.join('..', 'data', 'cityscapes', 'data', 'leftImg8bit')
TARGET_DIR = os.path.join('..', 'data', 'cityscapes', 'data', 'gtFine')


class Cityscapes(torch_data.Dataset):

    def __init__(self, data, transform=None, target_transform=None, test_transform=None, cl19=False,
                 test_bisenetv2=False, double=False, quadruple=False, use_cv2_transform=False, dom_gen=None,
                 split_name='heterogeneous'):

        self.use_cv2_transform = use_cv2_transform
        self.images = data
        self.true_len = len(self.images['x'])
        self.transform = transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.test_bisenetv2 = test_bisenetv2
        self.double = double
        self.quadruple = quadruple
        self.dom_gen = dom_gen
        self.split_name = split_name

        if cl19 and target_transform is None:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
            self.target_transform = lambda x: from_numpy(mapping[x])

    def cfsi(self, index):

        ampl_dir = os.path.join('..', 'data', 'cityscapes', 'data', self.split_name, 'bank_A')
        img = Image.open(os.path.join(IMAGES_DIR, self.images['x'][index])).convert('RGB')
        target = cv2.imread(os.path.join(TARGET_DIR, self.images['y'][index]), 0)
        # exchange info with other domain
        amplitudes = os.listdir(ampl_dir)
        amp_n = np.load(os.path.join(ampl_dir, np.random.choice(amplitudes)))

        img_np = np.asarray(img, np.float32)
        fft_img_np = np.fft.fft2(img_np, axes=(0, 1))
        amp_k, pha_k = np.abs(fft_img_np), np.angle(fft_img_np)

        a_local = np.fft.fftshift(amp_k, axes=(0, 1))
        a_trg = np.fft.fftshift(amp_n, axes=(0, 1))
        alpha = 0.05
        h, w, _ = a_local.shape
        h_l = np.floor(h * alpha).astype(int)
        w_l = np.floor(w * alpha).astype(int)
        # center
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)

        h1 = c_h - h_l
        h2 = c_h + h_l + 1
        w1 = c_w - w_l
        w2 = c_w + w_l + 1
        lamb = np.random.uniform(size=1)
        a_local[h1:h2, w1:w2, :] = a_local[h1:h2, w1:w2, :] * (1 - lamb) + a_trg[h1:h2, w1:w2, :] * lamb
        a_local = np.fft.ifftshift(a_local, axes=(0, 1))

        fft_local_ = a_local * np.exp(1j * pha_k)
        local_in_trg = np.fft.ifft2(fft_local_, axes=(0, 1))
        local_in_trg = np.abs(local_in_trg)
        # np.clip(local_in_trg / 255, 0, 1)
        img = local_in_trg.astype(np.uint8)
        return img, target

    def lab(self, index):

        target = cv2.imread(os.path.join(TARGET_DIR, self.images['y'][index]), 0)
        lab_dir = os.path.join('..', 'data', 'cityscapes', 'data', self.split_name, 'bank_lab')
        # exchange info with other domain
        mean_std_list = os.listdir(lab_dir)
        choice = np.random.choice(mean_std_list)
        infile = open(os.path.join(lab_dir, choice), 'rb')
        dict_ms = pickle.load(infile)
        mean_t = dict_ms['mean']
        std_t = dict_ms['std']
        infile.close()

        img = cv2.imread(os.path.join(IMAGES_DIR, self.images['x'][index]))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        mean_s = np.mean(img_lab, axis=(0, 1))
        std_s = np.std(img_lab, axis=(0, 1))

        img_new = (img_lab - mean_s) * std_t / std_s + mean_t

        img_new = np.clip(img_new, 0, 255)
        img_new = img_new.astype(np.uint8)

        img = cv2.cvtColor(img_new, cv2.COLOR_Lab2RGB)

        return img, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the label of segmentation.
        """
        original_index = index

        if original_index >= self.true_len and (self.double or self.quadruple):
            index %= self.true_len

        if self.dom_gen == 'cfsi' and (index % 2 == 0) and not self.test_bisenetv2:
            img, target = self.cfsi(index)
        elif self.dom_gen == 'lab' and (index % 2 == 0) and not self.test_bisenetv2:
            img, target = self.lab(index)
        else:
            if self.use_cv2_transform:
                img = cv2.imread(os.path.join(IMAGES_DIR, self.images['x'][index]))[:, :, ::-1]
                target = cv2.imread(os.path.join(TARGET_DIR, self.images['y'][index]), 0)
            else:
                img = Image.open(os.path.join(IMAGES_DIR, self.images['x'][index]))
                target = Image.open(os.path.join(TARGET_DIR, self.images['y'][index]))

        if (self.double and original_index >= self.true_len) or (
                self.quadruple and original_index >= 2 * self.true_len):
            if self.use_cv2_transform:
                img, target = Tcv.RandomHorizontalFlip(1)(img, target)
            else:
                img, target = T.RandomHorizontalFlip(1)(img, target)

        original_img = None

        if self.transform is not None:
            if self.test_bisenetv2:
                original_img = img.copy()
                img = self.test_transform(img)
            else:
                img, target = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.test_bisenetv2:
            if self.use_cv2_transform:
                return (self.test_transform.transforms[-1](original_img), img), target
            return (T.Compose(self.test_transform.transforms[-2:])(original_img), img), target  # jump the resize

        return img, target

    def __len__(self):

        if self.double:
            return 2 * self.true_len
        if self.quadruple:
            return 4 * self.true_len
        return self.true_len
