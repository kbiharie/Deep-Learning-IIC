import json
import os
import cv2
import torch
import numpy as np
import torch.utils.data
import torch.nn.functional
import prep_data
import json
import sys
import torch.optim
import os.path
import time
import PIL.Image
from display import *
from model import *
from configuration import *
from dataset import *
from evaluate import evaluate


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
    config = create_config()

    # Create train_imgs
    # Create dataset
    dataset = CocoStuff3Dataset(config, "train")

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=config.shuffle,
                                                   num_workers=config.num_workers,
                                                   drop_last=False)

    net = IICNet(config)
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    optimizer = torch.optim.Adam(net.module.parameters(), lr=0.0001)

    epochs = 30
    all_losses = []

    log_file = time.strftime("../datasetscopy/logs/%Y_%m_%d-%H_%M_%S_log.json")

    log = []
    with open(log_file, "w") as w:
        json.dump(log, w)

    # For every epoch
    for epoch in range(epochs):
        print("epoch", epoch)
        total_loss = 0
        total_loss_no_lamb = 0
        start_time = time.time()
        epoch_model_path = "../datasetscopy/models/" + config.model_name + "_epoch_" + str(epoch) + ".pth"
        if os.path.exists(epoch_model_path) and config.existing_model:
            net.load_state_dict(torch.load(epoch_model_path))
            optimizer = torch.optim.Adam(net.module.parameters(), lr=0.0001)
            continue
        # For every batch
        batch_time = time.time()
        for step, (img1, img2, flip, mask) in enumerate(train_dataloader):
            print("batch", step - 1, "took", time.time() - batch_time)
            batch_time = time.time()
            img1 = img1.cuda()
            img2 = img2.cuda()
            mask = mask.cuda()

            img1 = sobel(img1)
            img2 = sobel(img2)

            net.module.zero_grad()
            n_imgs = img1.shape[0]
            x1_outs = net(img1)
            x2_outs = net(img2)

            del img1
            del img2


            # TODO: is this the same dimension?
            for i in range(x2_outs.shape[0]):
                if flip[i]:
                    x2_outs[i] = torch.flip(x2_outs[i], dims=[1])

            avg_loss_batch = None
            avg_loss_no_lamb_batch = None

            loss, loss_no_lamb = loss_fn(x1_outs, x2_outs, all_mask_img1=mask)

            del x1_outs
            del x2_outs

            loss.backward()
            optimizer.step()
            total_loss += loss
            total_loss_no_lamb += loss_no_lamb
        to_log = {"type": "epoch", "loss": total_loss.item(), "epoch": epoch, "duration": time.time() - start_time,
                  "finished": time.strftime("%Y_%m_%d-%H_%M_%S")}
        log.append(to_log)
        all_losses.append(total_loss)
        torch.save(net.state_dict(), epoch_model_path)
        with open(log_file, "r") as f:
            old_log = json.load(f)
        old_log.append(to_log)
        with open(log_file, "w") as w:
            json.dump(old_log, w)

        print(total_loss.item())

    torch.save(net.state_dict(), "../datasetscopy/models/" + config.model_name + ".pth")


def loss_fn(x1_outs, x2_outs, all_mask_img1=None, lamb=1.0):
    # TODO: perform inverse affine transformation
    half_T_side_dense = 10

    x2_outs_inv = x2_outs

    bn, k, h, w = x1_outs.shape
    all_mask_img1 = all_mask_img1.view(bn, 1, h, w)
    x1_outs = x1_outs * all_mask_img1
    x2_outs_inv = x2_outs_inv * all_mask_img1

    x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()
    x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()

    p_i_j = torch.nn.functional.conv2d(x1_outs, weight=x2_outs_inv,
                                       padding=(half_T_side_dense, half_T_side_dense))

    # UNCOLLAPSED
    p_i_j = p_i_j.permute(2,3,0,1)
    p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
    p_i_j = (p_i_j + p_i_j.permute(0,1,3,2)) / 2.

    p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1,1,k,1)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1,1,1,k)

    EPS = sys.float_info.epsilon

    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS

    T_side_dense = half_T_side_dense * 2 + 1

    loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                      lamb * torch.log(p_j_mat))).sum() / (T_side_dense * T_side_dense)

    loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                              torch.log(p_j_mat))).sum() / (T_side_dense * T_side_dense)

    return loss, loss_no_lamb


def display_image():
    config = create_config()
    dataset = CocoStuff3Dataset(config, "train")
    for i in range(10, len(dataset)):
        img1, img2, flip, mask = dataset.__getitem__(i)
        display_output_image_and_output(img1, mask)


if __name__ == "__main__":
    # with open("../annotations/stuff_train2017.json") as f:
    #     annotations = json.load(f)
    # ids = cocostuff_ids(annotations)
    # ids = cocostuff3_ids(annotations)
    # print(len(ids))
    # cocostuff_clean(ids, annotations, "../datasets/train2017")

    # transform_single_image("../datasets/val2017/000000001532.jpg")
    create_model()
    # prep_data.cocostuff3_write_filenames()
    # dataset = CocoStuff3Dataset(create_config(), "train")
    # prep_data.cocostuff_crop()
    # prep_data.cocostuff_clean_with_json(True)
    # display_image()
    # evaluate()
    # test_loss()
