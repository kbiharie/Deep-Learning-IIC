import torch
import numpy as np
import json
import torchvision
import cv2
import PIL
import time

class CocoStuff3Dataset(torch.utils.data.Dataset):

    def __init__(self, config):
        # create cool dictionary
        with open(config.filenames) as f:
            self.data = json.load(f)
        self.jitter_tf = torchvision.transforms.ColorJitter(brightness=config.jitter_brightness,
                                                            contrast=config.jitter_contrast,
                                                            saturation=config.jitter_saturation,
                                                            hue=config.jitter_hue)
        self.flip_p = config.flip_p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        start = time.time()
        image_path = self.data[id]["file"]
        img1 = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        cv2.resize(img1, dsize=None, fx=2/3, fy=2/3,
                   interpolation=cv2.INTER_LINEAR)
        x = np.random.randint(img1.shape[1] - 128)
        y = np.random.randint(img1.shape[0] - 128)
        img1 = img1[int(y):int(y + 128), int(x):int(x+128)]
        # create image pair and transform
        img1 = PIL.Image.fromarray(img1.astype(np.uint8))
        img2 = self.jitter_tf(img1)
        img1 = np.array(img1)
        img2 = np.array(img2)
        img1 = grey_image(img1)
        img2 = grey_image(img2)
        img1 = img1.astype(np.float32) / 255.
        img2 = img2.astype(np.float32) / 255.
        # image to gpu
        img1 = torch.from_numpy(img1).permute(2, 0, 1).to(torch.float32)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).to(torch.float32)

        # flip
        flip = False
        if np.random.rand() <= self.flip_p:
            img2 = torch.flip(img2, dims=[2])
            flip = True
        # print(time.time() - start)
        return img1, img2, flip

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