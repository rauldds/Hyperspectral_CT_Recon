from src.DETCTCNN.model.Unet3DMC import Unet3DMC
from src.DETCTCNN.model.Unet2DMC import Unet2DMC


def get_model(input_channels=2,with_1conv=True, use_bn=False, depth=3,basic_out_channel=64, n_labels=7, data_dimensions=2, skip_connections=True):
    if data_dimensions == 3:
        model = Unet3DMC(input_channels=input_channels, 
                        with_1conv=with_1conv,
                        use_bn=use_bn,
                        depth=depth,
                        basic_out_channel=basic_out_channel,
                        n_labels=n_labels
        )
    else:
        model = Unet2DMC(input_channels=input_channels, 
                        with_1conv=with_1conv,
                        use_bn=use_bn,
                        depth=depth,
                        basic_out_channel=basic_out_channel,
                        n_labels=n_labels,
                        skip_connections=skip_connections
        )
    return model
