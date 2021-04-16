import torch
import time
from itertools import permutations

from configuration import create_config
from model import IICNet
from dataset import CocoStuff3Dataset, sobel


def evaluate():
    print("evaluating")

    config = create_config()

    # Create network
    net = IICNet(config)
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load("../datasets/models/" + config.model_name + ".pth"))

    dataloader = torch.utils.data.DataLoader(CocoStuff3Dataset(config, "test"),
                                                                batch_size=config.dataloader_batch_sz,
                                                                shuffle=False,
                                                                num_workers=config.num_workers,
                                                                drop_last=False)

    torch.cuda.empty_cache()
    net.eval()
    test_accs = 0
    samples = 0
    matches = []
    start_time = time.time()
    seq = {}

    # For every batch
    for bnumber, curr_batch in enumerate(dataloader):
        if bnumber % 20 == 0:
            print(bnumber, time.time() - start_time)

        # Retrieve match and accuracy for batch
        match, test_acc, batch_samples = get_assignment_data_matches(net,
                                                                     (bnumber, curr_batch),
                                                                     config)
        test_accs += test_acc * batch_samples
        samples += batch_samples
        matches.append(match)

        # Create match frequency dictionary
        matchstr = "".join([str(match[x]) for x in match])
        if matchstr not in seq:
            seq[matchstr] = 0
        seq[matchstr] += 1

    net.train()
    torch.cuda.empty_cache()

    # Print frequency of all matches
    print(seq)

    # Calculate weighted average accuracy
    test_accs = test_accs / samples
    print(test_accs)


def get_assignment_data_matches(net, curr_batch, config):
    # Retrieve class outputs
    predictions_batch, labels_batch = segmentation_data_method(net, curr_batch)

    num_test = labels_batch.shape[0]
    num_samples = num_test

    # Retrieve match
    match = original_match(predictions_batch, labels_batch, config.out_channels_a, config.out_channels_a)
    found = torch.zeros(config.out_channels_a)
    reordered_preds = torch.zeros(num_samples,
                                  dtype=predictions_batch.dtype).cuda()

    # Replace class in image with ground truth class
    for pred_i in match:
        target_i = match[pred_i]
        reordered_preds[predictions_batch == pred_i] = target_i
        found[pred_i] = 1

    # Calculate accuracy
    acc = int((reordered_preds == labels_batch).sum()) / float(reordered_preds.shape[0])

    return match, acc, reordered_preds.shape[0]


def segmentation_data_method(net, curr_batch):
    batch_number, batch = curr_batch
    imgs, labels, mask = batch
    imgs = imgs.cuda()
    imgs = sobel(imgs)

    with torch.no_grad():
        x_outs = net(imgs)

    # Retrieve best classes
    batch_pred = torch.argmax(x_outs, dim=1)

    # Images to gpu
    predictions_batch = batch_pred.view(-1).cuda()
    labels_batch = labels.view(-1).cuda()
    mask_batch = mask.view(-1).cuda()

    # Only keep masked parts
    predictions_batch = predictions_batch.masked_select(mask=mask_batch)
    labels_batch = labels_batch.masked_select(mask=mask_batch)

    return predictions_batch, labels_batch


def original_match(predictions_batch, labels_batch, preds_k, labels_k, distinct=False):

    # Map each output channel to the best matching ground truth
    out_to_gts = {}
    out_to_gts_best_scores = {}
    out_to_gts_scores = {out_c: {} for out_c in range(preds_k)}
    for out_c in range(preds_k):
        for gt_c in range(labels_k):
            # Calculate score
            tp_score = int(((predictions_batch == out_c) * (labels_batch == gt_c)).sum())
            # Store best score
            if (out_c not in out_to_gts) or (tp_score > out_to_gts_best_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_best_scores[out_c] = tp_score
            out_to_gts_scores[out_c][gt_c] = tp_score

    if not distinct:
        return out_to_gts

    # If distinct, do one to one matching
    perms = set(permutations(range(preds_k)))
    best_score = -1
    for perm in perms:
        tp_score = 0
        for out_c in range(preds_k):
            tp_score += out_to_gts_scores[out_c][perm[out_c]]
        if tp_score >= best_score:
            best_score = tp_score
            for out_c in range(preds_k):
                out_to_gts[out_c] = perm[out_c]

    return out_to_gts
