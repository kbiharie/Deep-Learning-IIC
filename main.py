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
from model import *
from configuration import *
from dataset import *


def transform_single_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.uint8)
    image = torch.from_numpy(image).cuda().permute(2, 0, 1)
    img2 = torch.flip(image, dims=[2]).permute(1, 2, 0)
    img2 = np.array(img2.cpu())
    window_name = 'image'
    cv2.imshow(window_name, img2)
    cv2.waitKey(0)


def create_model(model_name):
    # Set parameters
    config = create_config()

    # Create train_imgs
    # Create dataset
    dataset = CocoStuff3Dataset(config)

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=config.shuffle,
                                                   num_workers=4,
                                                   drop_last=False)

    net = IICNet(config)
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
        torch.save(net.state_dict(), "../datasets/models/" + model_name + "_epoch" + str(epoch) + ".pth")

        print(total_loss.item())

    torch.save(net.state_dict(), "../datasets/models/" + model_name + ".pth")

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
    net = IICNet(create_config())
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


def evaluate(model_name):
    print("evaluating")


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
    create_model("coco3")
    # test()
    # prep_data.cocostuff_crop()
    # prep_data.cocostuff_clean_with_json(True)
