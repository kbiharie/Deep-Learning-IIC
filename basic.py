import cv2
import torch
import numpy as np
import torch.utils.data
import torchvision.transforms
import torch.nn.functional
import json
import torch.optim
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        start = time.time()
        image_path = self.data[id]["file"]
        img1 = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.uint8)
        x = img1.shape[1] / 2 - 128 / 2
        y = img1.shape[0] / 2 - 128 / 2
        img1 = img1[int(y):int(y + 128), int(x):int(x+128)]
        # create image pair and transform

        # image to gpu
        img1 = torch.from_numpy(img1).permute(2, 0, 1).cuda()
        img2 = self.jitter_tf(img1)
        print("loading image took", str(time.time() - start))
        return img1, img2

def create_model():
    # Set parameters
    config = type('config', (object,), {})()
    config.dataloader_batch_sz = 32
    config.shuffle = True
    config.filenames = "../datasets/filenamescoco.json"
    config.jitter_brightness = 0.4
    config.jitter_contrast = 0.4
    config.jitter_saturation = 0.4
    config.jitter_hue = 0.125
    # Create dataset
    dataset = CocoStuff3Dataset(config)

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=config.shuffle,
                                                   num_workers=0,
                                                   drop_last=False)

    epochs = 5
    # For every epoch
    for epoch in range(epochs):
        start = time.time()
        # For every batch
        for step, (img1, img2) in enumerate(train_dataloader):
            if step == 20:
                break
            print(step)
            print("epoch took", time.time() - start, "s")
