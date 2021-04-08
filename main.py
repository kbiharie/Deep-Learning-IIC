import json
import os
import cv2
import torch
import numpy as np
import torch.utils.data
import torchvision.transforms
import torch.nn.functional
import prep_data
import json
import sys
import torch.optim
import os.path
import time
import PIL.Image


def transform_single_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.uint8)
    image = torch.from_numpy(image).cuda().permute(2, 0, 1)
    img2 = torch.flip(image, dims=[2]).permute(1, 2, 0)
    img2 = np.array(img2.cpu())
    window_name = 'image'
    cv2.imshow(window_name, img2)
    cv2.waitKey(0)


def create_model():
    # Set parameters
    config = type('config', (object,), {})()
    # TODO: maybe precroppings allows for larger batch sizes?
    config.dataloader_batch_sz = 32
    config.shuffle = True
    config.filenames = "../datasets/filenamescoco.json"
    config.jitter_brightness = 0.4
    config.jitter_contrast = 0.4
    config.jitter_saturation = 0.4
    config.jitter_hue = 0.125
    config.flip_p = 0.5

    # Create train_imgs

    # Create dataset
    dataset = CocoStuff3Dataset(config)

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=config.shuffle,
                                                   num_workers=4,
                                                   drop_last=False)

    net = IICNet()
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    optimizer = torch.optim.Adam(net.module.parameters(), lr=0.1)

    epochs = 5

    all_losses = []

    # For every epoch
    for epoch in range(epochs):
        total_loss = 0
        total_loss_no_lamb = 0
        # For every batch
        for step, (img1, img2, flip) in enumerate(train_dataloader):
            if step == 3:
                break
            print(step)
            img1 = img1.cuda()
            img2 = img2.cuda()

            img1 = sobel(img1)
            img2 = sobel(img2)

            net.module.zero_grad()
            n_imgs = img1.shape[0]
            x1_outs = net(img1)
            x2_outs = net(img2)

            # TODO: is this the same dimension?
            for i in range(x2_outs.shape[0]):
                if flip[i]:
                    x2_outs[i] = torch.flip(x2_outs[i], dims=[1])

            # imgout = x1_outs.permute(0, 2, 3, 1)
            # imgout = imgout.numpy()
            # imgout = imgout.cpu().detach().numpy()
            # imgout = imgout * 255
            # imgout = imgout.astype(dtype="uint8")

            avg_loss_batch = None
            avg_loss_no_lamb_batch = None

            loss, loss_no_lamb = loss_fn(x1_outs, x2_outs)
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_loss_no_lamb += loss_no_lamb
            start = time.time()

            # for i in range(n_imgs):
            #     window_name = 'image ' + str(i)
            #     cv2.imshow(window_name, imgout[i])
            #     cv2.waitKey(0)
        all_losses.append(total_loss)

        print(total_loss.item())

    # torch.save(net.state_dict(), "../models/model.pth")
    # img1, img2 = dataset.__getitem__(0)
    # train_dataloader.
    # imgs = torch.zeros(1, 3, 64, 64).to(torch.float32).cuda()
    # imgs[0, :, :, :] = img1
    # net = IICNet()
    # net.cuda()
    # net = torch.nn.DataParallel(net)
    # x_outs = net(imgs)
    #
    # imgout = x_outs.permute(0,2,3,1)
    # imgout = imgout.cpu().detach().numpy()
    # imgout = imgout[0] * 255
    # img1, img2 = img1.permute(1, 2, 0), img2.permute(1, 2, 0)
    # img1, img2 = np.array(img1.cpu()), np.array(img2.cpu())
    #
    # imgout = imgout.astype(dtype="uint8")
    # imgout = cv2.copyMakeBorder(imgout, 0, 34, 0, 34, cv2.BORDER_CONSTANT, value=[255,255,255])
    #
    # images = np.concatenate((img1, img2, imgout), axis=1)
    # window_name = 'image'
    # cv2.imshow(window_name, images)
    # cv2.waitKey(0)

def test():
    net = IICNet()
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load("../models/model.pt"))
    net.eval()

    # Set parameters
    config = type('config', (object,), {})()
    # TODO: maybe precroppings allows for larger batch sizes?
    config.dataloader_batch_sz = 32
    config.shuffle = True
    config.filenames = "../datasets/filenamescoco.json"
    config.jitter_brightness = 0.4
    config.jitter_contrast = 0.4
    config.jitter_saturation = 0.4
    config.jitter_hue = 0.125

    # Create train_imgs

    # Create dataset
    dataset = CocoStuff3Dataset(config)

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=config.shuffle,
                                                   num_workers=0,
                                                   drop_last=False)

    for step, (img1, img2) in enumerate(train_dataloader):
        x_outs = net(img1.to(torch.float32))

        imgout = x_outs.permute(0, 2, 3, 1)
        imgout = imgout.cpu().detach().numpy()
        imgout = imgout[0] * 255
        # img1, img2 = img1.permute(1, 2, 0), img2.permute(1, 2, 0)
        # img1, img2 = np.array(img1.cpu()), np.array(img2.cpu())

        imgout = imgout.astype(dtype="uint8")

        # images = np.concatenate((img1, img2, imgout), axis=1)
        window_name = 'image'
        cv2.imshow(window_name, imgout)
        cv2.waitKey(0)
        break

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
        # x = img1.shape[1] / 2 - 128 / 2
        # y = img1.shape[0] / 2 - 128 / 2
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


class IICNet(torch.nn.Module):
    def __init__(self):
        super(IICNet, self).__init__()
        self.in_channels = 5
        self.pad = 1
        self.conv_size = 3
        self.out_channels = 3
        self.features = self._make_layers()
        self.track_running_stats = False

    #TODO: batchnorm stuff
    def _make_layers(self, batch_norm=True):
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=self.in_channels, out_channels=64,
                                      kernel_size=self.conv_size, stride=1,
                                      padding=self.pad, dilation=1, bias=False))
        layers.append(torch.nn.BatchNorm2d(64, track_running_stats=False))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(in_channels=64, out_channels=128,
                                      kernel_size=self.conv_size, stride=1,
                                      padding=self.pad, dilation=1, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(torch.nn.Conv2d(in_channels=128, out_channels=256,
                                      kernel_size=self.conv_size, stride=1,
                                      padding=self.pad, dilation=1, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(in_channels=256, out_channels=256,
                                      kernel_size=self.conv_size, stride=1,
                                      padding=self.pad, dilation=1, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(in_channels=256, out_channels=512,
                                      kernel_size=self.conv_size, stride=1,
                                      padding=self.pad, dilation=2, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(in_channels=512, out_channels=512,
                                      kernel_size=self.conv_size, stride=1,
                                      padding=self.pad, dilation=2, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Sequential(torch.nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=1,
                                      stride=1, dilation=1, padding=1, bias=False),
                      torch.nn.Softmax2d()))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)


def loss_fn(x1_outs, x2_outs, all_affine2_to_1=None,
                          all_mask_img1=None, lamb=1.0,
                          half_T_side_dense=0,
                          half_T_side_sparse_min=0,
                          half_T_side_sparse_max=0):
    #TODO: perform inverse affine transformation
    x2_outs_inv = x2_outs

    x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()
    x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()

    p_i_j = torch.nn.functional.conv2d(x1_outs, weight=x2_outs_inv,
                                       padding=(half_T_side_dense, half_T_side_dense))
    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)

    current_norm = float(p_i_j.sum())
    p_i_j = p_i_j / current_norm
    p_i_j = (p_i_j + p_i_j.t()) / 2.

    p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)
    p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)

    EPS = sys.float_info.epsilon

    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS

    loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                      lamb * torch.log(p_j_mat))).sum()

    loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                      torch.log(p_j_mat))).sum()

    return loss, loss_no_lamb

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


if __name__ == "__main__":
    # with open("../annotations/stuff_train2017.json") as f:
    #     annotations = json.load(f)
    # ids = cocostuff_ids(annotations)
    # ids = cocostuff3_ids(annotations)
    # print(len(ids))
    # cocostuff_clean(ids, annotations, "../datasets/train2017")

    # transform_single_image("../datasets/val2017/000000001532.jpg")
    # create_model()
    # prep_data.cocostuff3_write_filenames()
    create_model()
    # test()
    # prep_data.cocostuff_crop()
    # prep_data.cocostuff_clean_with_json()
