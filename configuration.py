

def create_config():

    config = type('config', (object,), {})()
    config.dataloader_batch_sz = 32
    config.shuffle = True
    config.filenames = "../datasets/filenamescoco.json"
    config.existing_model = True

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

    return config