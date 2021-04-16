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
    model_path = "../datasets/models/" + config.model_name +".pth"

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

    rgb = torch.zeros([out_display.shape[0], out_display.shape[1], out_display.shape[2]])

    out_display_max = torch.argmax(out_display, dim=2)
    out_display = out_display.cpu().detach().numpy()

    rgb = rgb.permute(2,0,1)
    rgb[0][out_display_max == 0] = 1
    rgb[1][out_display_max == 1] = 1
    rgb[2][out_display_max == 2] = 1
    mask = mask.cpu().detach()
    rgb = rgb * mask
    rgb = rgb.permute(1, 2, 0)

    display = np.concatenate((in_display, out_display, rgb), axis=1)

    cv2.imshow("window", display)
    cv2.waitKey(0)
