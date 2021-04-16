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

    def prepare_train(self, img, label):
        # Random crop image
        if self.random_crop:
            x = np.random.randint(img.shape[1] - 128) if img.shape[1] > 128 else 0
            y = np.random.randint(img.shape[0] - 128) if img.shape[0] > 128 else 0
        # Center crop image
        else:
            x = img.shape[1] / 2 - 64
            y = img.shape[0] / 2 - 64

        img = img[int(y):int(y + 128), int(x):int(x + 128)]
        label = label[int(y):int(y + 128), int(x):int(x + 128)]
        _, mask_img1 = filter_label(label)
        mask_img1 = torch.from_numpy(mask_img1.astype(np.uint8))

        # Create second image
        img = PIL.Image.fromarray(img.astype(np.uint8))

        # Jitter img2
        img2 = self.jitter_tf(img)
        img = np.array(img)
        img2 = np.array(img2)

        # Add grey layer to images
        img = grey_image(img)
        img2 = grey_image(img2)
        img = img.astype(np.float32) / 255.
        img2 = img2.astype(np.float32) / 255.

        # Create tensors
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)
        img2 = torch.from_numpy(img2).permute(2, 0, 1).to(torch.float32)

        # Flip second image
        flip = False
        if np.random.rand() <= self.flip_p:
            img2 = torch.flip(img2, dims=[2])
            flip = True

        return img, img2, flip, mask_img1

    def prepare_test(self, img, label):
        x = img.shape[1] / 2 - 64
        y = img.shape[0] / 2 - 64

        # Center crop
        img = img[int(y):int(y + 128), int(x):int(x + 128)]
        label = label[int(y):int(y + 128), int(x):int(x + 128)]

        img = grey_image(img)
        img = img.astype(np.float32) / 255.

        # Create tensor
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Collapse classes in label
        label, mask = filter_label(label)

        return img, torch.from_numpy(label), torch.from_numpy(mask.astype(np.uint8))

    def __getitem__(self, id):

        image_path = self.data[id]["file"]

        # If something goes wrong we want to know what caused it
        if not os.path.exists(image_path):
            print(image_path)

        # Generate the label path from image_path
        label_path = get_label_path(image_path)

        # Open the images
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.uint32)
        img = img.astype(np.float32)
        label = label.astype(np.int32)

        # Scale the images
        img = cv2.resize(img, dsize=None, fx=2 / 3, fy=2 / 3,
                         interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, dsize=None, fx=2 / 3,
                           fy=2 / 3,
                           interpolation=cv2.INTER_NEAREST)

        assert img.shape[1] >= 128
        assert img.shape[0] >= 128

        if self.purpose == "train":
            output = self.prepare_train(img, label)
        elif self.purpose == "test":
            output = self.prepare_test(img, label)
        else:
            raise NotImplementedError("Type is not train or test.")
        return output


def filter_label(label):
    # Default class is -1
    new_label_map = -1 * np.ones(label.shape, dtype=label.dtype)

    # Categories taken from the annotation file
    sky_labels = [105, 156]
    ground_labels = [110, 124, 125, 135, 139, 143, 144, 146, 148, 153, 158]
    plant_labels = [93, 96, 118, 123, 128, 133, 141, 162, 168]

    # Collapse classes from label
    new_label_map[np.isin(label, sky_labels)] = 0
    new_label_map[np.isin(label, ground_labels)] = 1
    new_label_map[np.isin(label, plant_labels)] = 2

    mask = (new_label_map >= 0)

    return new_label_map, mask


def grey_image(img):
    # Add a grey channel
    return np.concatenate([img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(img.shape[0], img.shape[1], 1)], axis=2)


def sobel(imgs):
    # Take grey channel
    grey_imgs = imgs[:, 3, :, :].unsqueeze(1)

    # Remove grey channel
    rgb_imgs = imgs[:, :3, :, :]

    # Kernel for vertical edges
    sobelxweights = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    convx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convx.weight = torch.nn.Parameter(
        torch.Tensor(sobelxweights).cuda().float().unsqueeze(0).unsqueeze(0))
    dx = convx(torch.autograd.Variable(grey_imgs)).data

    # Kernel for horizontal edges
    sobelyweights = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    convy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convy.weight = torch.nn.Parameter(
        torch.from_numpy(sobelyweights).cuda().float().unsqueeze(0).unsqueeze(0))
    dy = convy(torch.autograd.Variable(grey_imgs)).data

    # Append sobel channels
    return torch.cat([rgb_imgs, dx, dy], dim=1)


def get_label_path(img_path):
    # Label name is same as image name, but folder and extension are different
    return img_path.replace("train2017", "traingt2017").replace("val2017", "valgt2017").replace(".jpg", ".png")