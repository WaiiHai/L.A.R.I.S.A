import model as nn
models = {
    'UNet': nn.build_unet,
    'DeepLabV3Plus': nn.build_deeplabv3plus,
    'UNet3Plus': nn.build_unet3plus,
    'EfficientNet': nn.build_efficientnet_unet
}

paths = {
    'UNet': 'logs/2024-05-29_14-49-45_UNET/model.weights.h5',
    'DeepLabV3Plus': 'logs/2024-05-14_18-38-19_DeepLabV3Plus/model.weights.h5',
    'UNet3Plus': 'logs/2024-05-21_17-49-55_UNET_3PLUS/model.weights.h5',
    'EfficientNet': 'logs/2024-05-21_23-40-41_EfficientNetB0_UNET/model.weights.h5'
}

if __name__ == '__main__':
    from utils import init_run_option
    (name, path, output) = init_run_option()
    if name not in  models: name = 'DeepLabV3Plus'

    model = models[name]((256, 256, 3))
    model.load_weights(paths[name])
        
    from videoeditor import VideoEditor
    video_editor = VideoEditor(path, output)(model)