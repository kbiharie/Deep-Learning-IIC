from model import *
from configuration import *
from dataset import *


def display_dataset_image(image, mask):
    if image.shape[0] == 4:
        image = image[:3, :, :]
    masked = image * mask.view(1, image.shape[1], image.shape[2])
    image = image.permute(1, 2, 0)
    masked = masked.permute(1, 2, 0)
    return image, masked


def display_output_image_and_output(image, mask):
    in_display, masked_display = display_dataset_image(image, mask)

    config = create_config()
    model_path = "../datasets/models/coco3.pth"

    net = IICNet(config)
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    imgs = np.zeros((1, image.shape[0], image.shape[1], image.shape[2]))
    imgs = torch.tensor(imgs).cuda().to(torch.float32)
    imgs[0] = image
    imgs = sobel(imgs)

    mask = mask.cuda()

    out_display = net(imgs)
    out_display = out_display[0]
    out_display = out_display * mask
    out_display = out_display.permute(1, 2, 0)
    out_display = out_display.cpu().detach().numpy()

    # TODO: take arg max of out_display

    display = np.concatenate((in_display, masked_display, out_display), axis=1)

    cv2.imshow("window", display)
    cv2.waitKey(0)