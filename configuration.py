
def create_config():

    config = type('config', (object,), {})()
    config.dataloader_batch_sz = 32
    config.input_sz = 128
    config.shuffle = True
    config.filenames = "../datasets/filenamescocofew.json"

    config.existing_model = True
    config.num_workers = 2
    config.model_name = "coco3_oc_epoch_0"

    config.jitter_brightness = 0.4
    config.jitter_contrast = 0.4
    config.jitter_saturation = 0.4
    config.jitter_hue = 0.125
    config.flip_p = 0.5

    # RGB + 2 sobel
    config.in_channels = 5
    config.pad = 1
    config.conv_size = 3

    # Out head A
    config.out_channels_a = 3
    # Out head B
    config.out_channels_b = 15

    config.random_crop = True
    config.overclustering = False

    config.dataset_path = "../datasets/"

    return config
