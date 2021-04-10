

def create_config():

    config = type('config', (object,), {})()
    config.dataloader_batch_sz = 32
    config.input_sz = 128
    config.output_k = 3
    config.gt_k = 3
    config.shuffle = True
    config.filenames = "../datasets/filenamescoco.json"
    config.existing_model = False
    config.model_name = "coco3"

    config.jitter_brightness = 0.4
    config.jitter_contrast = 0.4
    config.jitter_saturation = 0.4
    config.jitter_hue = 0.125
    config.flip_p = 0.5

    # Model
    config.in_channels = 5
    config.pad = 1
    config.conv_size = 3
    config.out_channels = 3

    config.random_crop = True

    return config