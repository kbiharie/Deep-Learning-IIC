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
from display import *
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


def create_model():
    # Set parameters
    config = create_config()

    dataset = CocoStuff3Dataset(config, "train")

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

    epochs = 10

    log_file = time.strftime("../datasets/logs/%Y_%m_%d-%H_%M_%S_log.json")

    log = []
    with open(log_file, "w") as w:
        json.dump(log, w)

    heads = 2 if config.overclustering else 1

    # For every epoch
    for epoch in range(epochs):
        epoch_model_path = "../datasets/models/" + config.model_name + "_epoch_" + str(epoch) + ".pth"
        if os.path.exists(epoch_model_path) and config.existing_model:
            net.load_state_dict(torch.load(epoch_model_path))
            optimizer = torch.optim.Adam(net.module.parameters(), lr=0.0001)
            continue
        for head in range(heads):
            print("epoch", epoch)
            print("head", head)
            total_loss = 0
            total_loss_no_lamb = 0
            start_time = time.time()

            # For every batch
            batch_time = time.time()
            for step, (img1, img2, flip, mask) in enumerate(train_dataloader):
                print("batch", step - 1, "took", time.time() - batch_time)
                batch_time = time.time()
                if step == 5:
                    break
                img1 = img1.cuda()
                img2 = img2.cuda()
                mask = mask.cuda()

                img1 = sobel(img1)
                img2 = sobel(img2)

                net.module.zero_grad()
                x1_outs = net(img1, head)
                x2_outs = net(img2, head)

                del img1
                del img2

                for i in range(x2_outs.shape[0]):
                    if flip[i]:
                        x2_outs[i] = torch.flip(x2_outs[i], dims=[1])

                loss, loss_no_lamb = loss_fn(x1_outs, x2_outs, all_mask_img1=mask)

                del x1_outs
                del x2_outs

                loss.backward()
                optimizer.step()
                total_loss += loss
                total_loss_no_lamb += loss_no_lamb
            to_log = {"type": "epoch_" + str(head), "loss": total_loss.item(), "epoch": epoch, "duration": time.time() - start_time,
                      "finished": time.strftime("%Y_%m_%d-%H_%M_%S")}
            log.append(to_log)
            with open(log_file, "r") as f:
                old_log = json.load(f)
            old_log.append(to_log)
            with open(log_file, "w") as w:
                json.dump(old_log, w)
            print(total_loss.item())
        torch.save(net.state_dict(), epoch_model_path)

    torch.save(net.state_dict(), "../datasets/models/" + config.model_name + ".pth")


def evaluate():
    print("evaluating")

    config = create_config()

    net = IICNet(config)
    # net.cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load("../datasets/models/" + config.model_name + ".pth"))

    mapping_assignment_dataloader = torch.utils.data.DataLoader(CocoStuff3Dataset(config, "test"),
                                                   batch_size=config.dataloader_batch_sz,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   drop_last=False)
    match, test_acc = eval(config,
                           net,
                           mapping_assignment_dataloader)

    print(test_acc)


def eval(config, net, mapping_assignment_dataloader):
    torch.cuda.empty_cache()
    net.eval()
    test_accs = 0
    samples = 0
    matches = []

    start_time = time.time()

    seq = {}

    for bnumber, curr_batch in enumerate(mapping_assignment_dataloader):
        if bnumber % 20 == 0:
            print(bnumber, time.time() - start_time)
        match, test_acc, batch_samples = _get_assignment_data_matches(net,
                                                                      (bnumber, curr_batch),
                                                                      config,
                                                                      segmentation_data_method)
        test_accs += test_acc * batch_samples
        samples += batch_samples
        matches.append(match)
        matchstr = "".join([str(match[x]) for x in match])
        if matchstr not in seq:
            seq[matchstr] = 0
        seq[matchstr] += 1
        if bnumber == 100:
            break

    net.train()
    torch.cuda.empty_cache()
    print(seq)
    return matches, test_accs / samples


def _get_assignment_data_matches(net, curr_batch, config, segmentation_data_method):
    predictions_batch, labels_batch = segmentation_data_method(config, net, curr_batch)

    num_test = labels_batch.shape[0]
    num_samples = num_test

    match = _original_match(predictions_batch, labels_batch, config.output_k, config.gt_k)
    match = {0:1, 1:2, 2:0}
    found = torch.zeros(config.output_k)
    reordered_preds = torch.zeros(num_samples,
                                  dtype=predictions_batch.dtype).cuda()

    for pred_i in match:
        target_i = match[pred_i]
        reordered_preds[predictions_batch == pred_i] = target_i
        found[pred_i] = 1

    acc = int((reordered_preds == labels_batch).sum()) / float(reordered_preds.shape[0])

    return match, acc, reordered_preds.shape[0]


def segmentation_data_method(config, net, curr_batch):
    batch_number, batch = curr_batch
    imgs, labels, mask = batch

    imgs = imgs.cuda()
    imgs = sobel(imgs)

    with torch.no_grad():
        x_outs = net(imgs)

    batch_pred = torch.argmax(x_outs, dim=1)

    # vectorise stuff view(-1)
    predictions_batch = batch_pred.view(-1).cuda()
    labels_batch = labels.view(-1).cuda()
    mask_batch = mask.view(-1).cuda()

    predictions_batch = predictions_batch.masked_select(mask=mask_batch)
    labels_batch = labels_batch.masked_select(mask=mask_batch)

    return predictions_batch, labels_batch


def _original_match(predictions_batch, labels_batch, preds_k, labels_k):
    # map each output channel to the best matching ground truth (many to one)

    out_to_gts = {}
    out_to_gts_scores = {}
    for out_c in range(preds_k):
        for gt_c in range(labels_k):
            # the amount of out_c at all the gt_c samples
            tp_score = int(((predictions_batch == out_c) * (labels_batch == gt_c)).sum())
            if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_scores[out_c] = tp_score
    return out_to_gts


def loss_fn(x1_outs, x2_outs, all_affine2_to_1=None,
            all_mask_img1=None, lamb=1.0,
            half_T_side_dense=0,
            half_T_side_sparse_min=0,
            half_T_side_sparse_max=0):
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

    # COLLAPSED
    # p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)
    #
    # current_norm = float(p_i_j.sum())
    # p_i_j = p_i_j / current_norm
    # p_i_j = (p_i_j + p_i_j.t()) / 2.
    #
    # p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)
    # p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)

    # UNCOLLAPSED
    p_i_j = p_i_j.permute(2,3,0,1)
    p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
    p_i_j = (p_i_j + p_i_j.permute(0,1,3,2)) / 2.

    p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1,1,k,1)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1,1,1,k)

    # CONTINUE
    EPS = sys.float_info.epsilon

    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS

    T_side_dense = half_T_side_dense * 2 + 1

    # Removed minus in front of p_i_j!!
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


def test_loss():
    test = "../images/test2.jpg"
    test_gt = "../images/test2.png"

    img = cv2.imread(test, cv2.IMREAD_COLOR).astype(np.uint8)
    label = cv2.imread(test_gt, cv2.IMREAD_GRAYSCALE).astype(np.uint32)
    img = img.astype(np.float32)
    label = label.astype(np.int32)

    img = cv2.resize(img, dsize=None, fx=2 / 3, fy=2 / 3,
                     interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, dsize=None, fx=2 / 3,
                       fy=2 / 3,
                       interpolation=cv2.INTER_NEAREST)

    x = img.shape[1] / 2 - 64
    y = img.shape[0] / 2 - 64

    img = img[int(y):int(y + 128), int(x):int(x + 128)]
    label = label[int(y):int(y + 128), int(x):int(x + 128)]
    img = img.astype(np.float32) / 255.
    grey = grey_image(img)
    grey = torch.from_numpy(grey).permute(2, 0, 1)
    img = torch.from_numpy(img).permute(2, 0, 1)
    label, mask = filter_label(label)

    label, mask = torch.from_numpy(label), torch.from_numpy(mask.astype(np.uint8))

    input = grey.cuda()
    flipped_input = torch.flip(input, dims=[1])

    inputs = torch.zeros([1, 4, input.shape[1], input.shape[2]]).cuda()
    inputs[0] = input
    config = create_config()
    model_path = "../datasets/models/" + "coco3" + "_epoch_1.pth"

    net = IICNet(config)
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    output1 = net(sobel(inputs))
    inputs[0] = flipped_input
    output2 = torch.flip(net(sobel(inputs)), dims=[1])

    out_display = torch.zeros([3, label.shape[0], label.shape[1]])

    out_display[0, label == 0] = 1
    out_display[1, label == 1] = 1
    out_display[2, label == 2] = 1

    imgs1 = torch.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
    imgs2 = torch.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
    masks = torch.zeros([1, mask.shape[0], mask.shape[1]])
    imgs1[0] = img
    imgs2[0] = out_display
    masks[0] = mask

    print(loss_fn(imgs2, imgs2, all_mask_img1=masks)[0])
    print(loss_fn(output1, output2, all_mask_img1=masks.cuda())[0])

    out_display[0,:,:] = 1
    out_display[1,:,:] = 0
    out_display[2,:,:] = 0
    imgs2[0] = out_display
    # print(loss_fn(imgs1, imgs2, all_mask_img1=masks)[0])

    out_display = out_display * mask
    out_display = out_display.permute(1, 2, 0)
    out_display = out_display.numpy()

    in_display = img.permute(1,2,0)

    masked_display = img * mask
    masked_display = masked_display.permute(1, 2, 0)
    masked_display = masked_display.numpy()

    print(in_display.shape, masked_display.shape, out_display.shape)

    display = np.concatenate((in_display, masked_display, out_display), axis=1)

    cv2.imshow("window", display)
    cv2.waitKey(0)


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
