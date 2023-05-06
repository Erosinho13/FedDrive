import os
import numpy as np
from torch.utils import data
from PIL import Image
from torch import from_numpy
import dataset.transform as T
import cv2
import pickle

IMAGES_DIR = os.path.join('..', 'data', 'idda', 'data', 'IDDAsmall')
eval_classes = list(range(1, 24))
class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]


class IDDADataset(data.Dataset):

    def __init__(self, data, transform=None, target_transform=None, test_transform=None, test_bisenetv2=True, mean=0,
                 crop_size=None, remap=False, ignore_index=255, dom_gen=None, use_cv2_transform=False,
                 setting_type=None, split_type=None, user=None):

        self.images = data
        self.true_len = len(self.images['x'])
        self.transform = transform
        self.test_transform = test_transform
        self.test_bisenetv2 = test_bisenetv2
        self.setting_type = setting_type
        self.split_type = split_type

        self.crop_size = crop_size
        self.ignore_index = ignore_index
        self.mean = mean
        self.remap = remap
        self.dom_gen = dom_gen
        self.use_cv2_transform = use_cv2_transform
        self.return_domain = False
        self.return_path = False
        self.user = user
        self.subclient_by_img = None

        if self.remap:  # 16 classes
            classes = class_eval
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[i] = cl
            self.target_transform = lambda x: from_numpy(mapping[x])
        else:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
            self.target_transform = lambda x: from_numpy(mapping[x])

    def __len__(self):
        return self.true_len

    def cfsi(self, index):
        ampl_dir = os.path.join('..', 'data', 'idda', 'data', self.split_type, self.setting_type, 'bank_A')
        img = Image.open(os.path.join(IMAGES_DIR, self.images['x'][index])).convert('RGB')
        target = Image.open(os.path.join(IMAGES_DIR, self.images['y'][index]))
        # exchange info with other domain
        amplitudes = os.listdir(ampl_dir)
        parts = self.images['x'][index].split('/')[0].split('_')
        ids = parts[0] + '_' + parts[1]
        amplitudes_filt = [a for a in amplitudes if not a.startswith(ids)]
        amp_n = np.load(os.path.join(ampl_dir, np.random.choice(amplitudes_filt)))

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

        local_in_trg = local_in_trg.astype(np.uint8)

        img = Image.fromarray(local_in_trg)
        if self.crop_size is not None:
            img = img.resize(self.crop_size, Image.BICUBIC)
            target = target.resize(self.crop_size, Image.NEAREST)

        return img, target

    def lab(self, index):
        target = Image.open(os.path.join(IMAGES_DIR, self.images['y'][index]))
        lab_dir = os.path.join('..', 'data', 'idda', 'data', self.split_type, self.setting_type, 'bank_lab')
        # exchange info with other domain
        mean_std_list = os.listdir(lab_dir)
        parts = self.images['x'][index].split('/')[0].split('_')
        ids = parts[0] + '_' + parts[1]
        mean_std_filt = [a for a in mean_std_list if not a.startswith(ids)]
        choice = np.random.choice(mean_std_filt)
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

        img_rgb = cv2.cvtColor(img_new, cv2.COLOR_Lab2RGB)

        img = Image.fromarray(img_rgb.astype(np.uint8))
        if self.crop_size is not None:
            img = img.resize(self.crop_size, Image.BICUBIC)
            target = target.resize(self.crop_size, Image.NEAREST)

        return img, target

    def __getitem__(self, index):

        if self.dom_gen == 'cfsi' and (index % 2 == 0) and not self.test_bisenetv2:
            img, target = self.cfsi(index)
        elif self.dom_gen == 'lab' and (index % 2 == 0) and not self.test_bisenetv2:
            img, target = self.lab(index)
        else:
            if self.use_cv2_transform:
                img = cv2.imread(os.path.join(IMAGES_DIR, self.images['x'][index]))[:, :, ::-1]
                target = cv2.imread(os.path.join(IMAGES_DIR, self.images['y'][index]), 0)
                if self.crop_size is not None:
                    img = cv2.resize(img, self.crop_size, cv2.INTER_CUBIC)
                    target = cv2.resize(target, self.crop_size, cv2.INTER_NEAREST)
            else:
                img = Image.open(os.path.join(IMAGES_DIR, self.images['x'][index]))
                target = Image.open(os.path.join(IMAGES_DIR, self.images['y'][index]))
                if self.crop_size is not None:
                    img = img.resize(self.crop_size, Image.BICUBIC)
                    target = target.resize(self.crop_size, Image.NEAREST)

        if self.transform is not None:
            if self.test_bisenetv2:
                original_img = img.copy()
                img = self.test_transform(img)
            else:
                img, target = self.transform(img, target)
                original_img = None
        else:
            original_img = None

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.test_bisenetv2:

            if self.use_cv2_transform:

                if self.return_domain:
                    return (self.test_transform.transforms[-1](original_img), img), target, \
                           self.images['x'][index].split('/')[0]
                if self.return_path:
                    return (self.test_transform.transforms[-1](original_img), img), target, \
                           self.images['x'][index]
                return (self.test_transform.transforms[-1](original_img), img), target

            if self.return_domain:
                return (T.Compose(self.test_transform.transforms[-2:])(original_img), img), target, \
                       self.images['x'][index].split('/')[0]
            if self.return_path:
                return (T.Compose(self.test_transform.transforms[-2:])(original_img), img), target, \
                       self.images['x'][index]
            return (T.Compose(self.test_transform.transforms[-2:])(original_img), img), target  # jump the resize

        if self.return_domain:
            return img, target, self.images['x'][index].split('/')[0]
        if self.return_path:
            return img, target, self.images['x'][index]
        return img, target
