import torch
import time

from configuration import create_config
from model import IICNet
from dataset import CocoStuff3Dataset, sobel


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
                                                                num_workers=config.num_workers,
                                                                drop_last=False)
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
        match, test_acc, batch_samples = get_assignment_data_matches(net,
                                                                     (bnumber, curr_batch),
                                                                     config)
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
    test_accs = test_accs / samples

    print(test_accs)


def get_assignment_data_matches(net, curr_batch, config):
    predictions_batch, labels_batch = segmentation_data_method(config, net, curr_batch)

    num_test = labels_batch.shape[0]
    num_samples = num_test

    match = original_match(predictions_batch, labels_batch, config.output_k, config.gt_k)
    # match = {0: 1, 1: 0, 2: 2}
    found = torch.zeros(config.output_k)
    reordered_preds = torch.zeros(num_samples,
                                  dtype=predictions_batch.dtype).cuda()

    for pred_i in match:
        target_i = match[pred_i]
        reordered_preds[predictions_batch == pred_i] = target_i
        found[pred_i] = 1

    acc = int((reordered_preds == labels_batch).sum()) / float(reordered_preds.shape[0])

    return match, acc, reordered_preds.shape[0]


def segmentation_data_method(net, curr_batch):
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


def original_match(predictions_batch, labels_batch, preds_k, labels_k):
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
