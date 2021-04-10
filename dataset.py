import torch
import numpy as np
import json
import torchvision
import cv2
import PIL
import time
import os


class CocoStuff3Dataset(torch.utils.data.Dataset):

    def __init__(self, config, purpose):
        # create cool dictionary
        with open(config.filenames) as f:
            self.data = json.load(f)
        self.jitter_tf = torchvision.transforms.ColorJitter(brightness=config.jitter_brightness,
                                                            contrast=config.jitter_contrast,
                                                            saturation=config.jitter_saturation,
                                                            hue=config.jitter_hue)
        self.flip_p = config.flip_p
        self.purpose = purpose
        self.random_crop = config.random_crop

    def __len__(self):
        return len(self.data)

    def _prepare_train(self, img, label):
        start = time.time()

        if self.random_crop:
            x = np.random.randint(img.shape[1] - 128)
            y = np.random.randint(img.shape[0] - 128)
        else:
            x = img.shape[1] / 2 - 64
            y = img.shape[0] / 2 - 64
        img = img[int(y):int(y + 128), int(x):int(x + 128)]
        label = label[int(y):int(y + 128), int(x):int(x + 128)]

        _, mask_img1 = _filter_label(label)

        mask_img1 = torch.from_numpy(mask_img1.astype(np.uint8))

        # create image pair and transform
        img = PIL.Image.fromarray(img.astype(np.uint8))
        img_pair = self.jitter_tf(img)
        img = np.array(img)
        img_pair = np.array(img_pair)
        img = grey_image(img)
        img_pair = grey_image(img_pair)
        img = img.astype(np.float32) / 255.
        img_pair = img_pair.astype(np.float32) / 255.
        # image to gpu
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)
        img_pair = torch.from_numpy(img_pair).permute(2, 0, 1).to(torch.float32)

        # flip
        flip = False
        if np.random.rand() <= self.flip_p:
            img_pair = torch.flip(img_pair, dims=[2])
            flip = True
        # print(time.time() - start)
        return img, img_pair, flip, mask_img1

    def _prepare_test(self, img, label):
        x = img.shape[1] / 2 - 64
        y = img.shape[0] / 2 - 64

        img = img[int(y):int(y + 128), int(x):int(x + 128)]
        label = label[int(y):int(y + 128), int(x):int(x + 128)]

        img = grey_image(img)
        img = img.astype(np.float32) / 255.
        # image to gpu
        img = torch.from_numpy(img).permute(2, 0, 1)

        label, mask = _filter_label(label)

        return img, torch.from_numpy(label), torch.from_numpy(mask.astype(np.uint8))

    def __getitem__(self, id):
        image_path = self.data[id]["file"]
        if not os.path.exists(image_path):
            print(image_path)
        label_path = get_label_path(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.uint32)
        img = img.astype(np.float32)
        label = label.astype(np.int32)

        img = cv2.resize(img, dsize=None, fx=2 / 3, fy=2 / 3,
                         interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=None, fx=2 / 3,
                           fy=2 / 3,
                           interpolation=cv2.INTER_NEAREST)

        if self.purpose == "train":
            return self._prepare_train(img, label)
        elif self.purpose == "test":
            return self._prepare_test(img, label)
        else:
            raise NotImplementedError("Type is not train or test.")


def _filter_label(label):
    # print(label)
    new_label_map = -1 * np.ones(label.shape, dtype=label.dtype)
    sky_labels = [105, 156]
    ground_labels = [110, 124, 125, 135, 139, 143, 144, 146, 148, 153, 158]
    plant_labels = [93, 96, 118, 123, 128, 133, 141, 162, 168]
    new_label_map[np.isin(label, sky_labels)] = 0
    new_label_map[np.isin(label, ground_labels)] = 1
    new_label_map[np.isin(label, plant_labels)] = 2

    mask = (new_label_map >= 0)

    return new_label_map, mask


def grey_image(img):
    return np.concatenate([img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(img.shape[0], img.shape[1], 1)], axis=2)


def sobel(imgs):
    grey_imgs = imgs[:, 3, :, :].unsqueeze(1)
    rgb_imgs = imgs[:, :3, :, :]

    sobelxweights = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    convx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convx.weight = torch.nn.Parameter(
        torch.Tensor(sobelxweights).cuda().float().unsqueeze(0).unsqueeze(0))
    dx = convx(torch.autograd.Variable(grey_imgs)).data

    sobelyweights = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convy.weight = torch.nn.Parameter(
        torch.from_numpy(sobelyweights).cuda().float().unsqueeze(0).unsqueeze(0))
    dy = convy(torch.autograd.Variable(grey_imgs)).data

    return torch.cat([rgb_imgs, dx, dy], dim=1)

def get_label_path(img_path):
    return img_path.replace("train2017", "traingt2017").replace("val2017", "valgt2017").replace(".jpg", ".png")