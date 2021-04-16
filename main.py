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


def train():
    # Create configuration
    config = create_config()

    # Create dataset and dataloader
    dataset = CocoStuff3Dataset(config, "train")
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=config.shuffle,
                                                   num_workers=config.num_workers,
                                                   drop_last=False)

    # Create network
    net = IICNet(config)
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    # Create optimizer
    optimizer = torch.optim.Adam(net.module.parameters(), lr=0.0001)

    epochs = 30

    # Create empty log file
    log_file = time.strftime(config.dataset_path + "logs/%Y_%m_%d-%H_%M_%S_log.json")
    log = []
    with open(log_file, "w") as w:
        json.dump(log, w)

    heads = 2 if config.overclustering else 1

    # For every epoch
    for epoch in range(epochs):
        epoch_model_path = config.dataset_path + "models/" + config.model_name + "_epoch_" + str(epoch) + ".pth"
        optimizer_path = config.dataset_path + "models/" + config.model_name + "_epoch_" + str(epoch) + ".pt"
        print("epoch", epoch)

        # Load existing epoch
        if os.path.exists(epoch_model_path) and config.existing_model:
            net.load_state_dict(torch.load(epoch_model_path))
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path))
            continue
        # For every head
        for head in range(heads):
            print("epoch", epoch)
            print("head", head)

            if head == 0:
                headstr = "A"
            elif head == 1:
                headstr = "B"

            # Load existing head
            epoch_head_path = config.dataset_path + "models/" + config.model_name + "_epoch_" + str(epoch) + headstr + ".pth"
            optimizer_head_path = config.dataset_path + "models/" + config.model_name + "_epoch_" + str(epoch) + headstr + ".pt"
            if os.path.exists(epoch_head_path) and config.existing_model:
                net.load_state_dict(torch.load(epoch_head_path))
                if os.path.exists(optimizer_head_path):
                    optimizer.load_state_dict(torch.load(optimizer_head_path))
                continue

            total_loss = 0
            start_time = time.time()

            # For every batch
            batch_time = time.time()
            for step, (img1, img2, flip, mask) in enumerate(train_dataloader):
                print("batch", step - 1, "took", time.time() - batch_time)
                batch_time = time.time()

                # Images to GPU
                img1 = img1.cuda()
                img2 = img2.cuda()
                mask = mask.cuda()

                # Sobel filters
                img1 = sobel(img1)
                img2 = sobel(img2)

                # Feed images to network
                net.module.zero_grad()
                x1_outs = net(img1, head)
                x2_outs = net(img2, head)

                # Flip outputs back if needed
                for i in range(x2_outs.shape[0]):
                    if flip[i]:
                        x2_outs[i] = torch.flip(x2_outs[i], dims=[1])

                # Calculate loss
                loss = loss_fn(x1_outs, x2_outs, all_mask_img1=mask)

                # Perform backward and optimizer.step
                loss.backward()
                optimizer.step()
                total_loss += loss

            # Write epoch and log to file
            to_log = {"type": "epoch_" + str(head), "loss": total_loss.item(), "epoch": epoch, "duration": time.time() - start_time,
                      "finished": time.strftime("%Y_%m_%d-%H_%M_%S")}
            log.append(to_log)
            with open(log_file, "r") as f:
                old_log = json.load(f)
            old_log.append(to_log)
            with open(log_file, "w") as w:
                json.dump(old_log, w)
            print(total_loss.item())
            if heads == 1:
                continue
            torch.save(net.state_dict(), epoch_head_path)
            torch.save(optimizer.state_dict(), optimizer_head_path)
        torch.save(net.state_dict(), epoch_model_path)
        torch.save(optimizer.state_dict(), optimizer_path)

    torch.save(net.state_dict(), config.dataset_path + "models/" + config.model_name + ".pth")
    torch.save(optimizer.state_dict(), config.dataset_path + "models/" + config.model_name + ".pt")


def loss_fn(x1_outs, x2_outs, all_mask_img1=None, lamb=1.0):
    half_T_side_dense = 10

    bn, k, h, w = x1_outs.shape
    all_mask_img1 = all_mask_img1.view(bn, 1, h, w)

    # Apply mask to both images
    x1_outs = x1_outs * all_mask_img1
    x2_outs = x2_outs * all_mask_img1

    # Permute channels to perform convolution
    x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()
    x2_outs = x2_outs.permute(1, 0, 2, 3).contiguous()

    # Perform convolution
    p_i_j = torch.nn.functional.conv2d(x1_outs, weight=x2_outs,
                                       padding=(half_T_side_dense, half_T_side_dense))

    # Permute channels back
    p_i_j = p_i_j.permute(2,3,0,1)
    p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
    p_i_j = (p_i_j + p_i_j.permute(0,1,3,2)) / 2.

    p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1,1,k,1)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1,1,1,k)

    # Delete small values
    EPS = sys.float_info.epsilon
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS

    T_side_dense = half_T_side_dense * 2 + 1

    # Calculate loss
    return (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                      lamb * torch.log(p_j_mat))).sum() / (T_side_dense * T_side_dense)


def display_image():
    config = create_config()
    dataset = CocoStuff3Dataset(config, "train")
    # Display output from network
    for i in range(len(dataset)):
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
    # train()
    # prep_data.cocostuff3_write_filenames()
    # dataset = CocoStuff3Dataset(create_config(), "train")
    # prep_data.cocostuff_crop()
    # prep_data.cocostuff_clean_with_json(True)
    # display_image()
    evaluate()
    # test_loss()
