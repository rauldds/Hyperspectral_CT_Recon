hparams = {
    'lr': 0.001,
    'epochs': 1,
    'batch_size': 4,
    'log_interval': 2,
    'early_stopping_patience': 10,
    'initial_epochs_buffer': 50,
    'dataset_path': "/Users/davidg/Projects/Hyperspectral_CT_Recon/MUSIC2D_HDF5",
}

hparams_LogReg = {
    'lr': 0.001,
    'epochs': 50,
    'batch_size': 256,
    'log_interval': 500,
    'early_stopping_patience': 10,
    'num_black_division_factor': 16,
    'initial_epochs_buffer': 2,
    "num_black_division_factor": 4,
    'dataset_path_2d': "/Users/davidg/Projects/Hyperspectral_CT_Recon/MUSIC2D_HDF5",
    'dataset_path_3d': "/Users/davidg/Projects/Hyperspectral_CT_Recon/MUSIC3D_HDF5",
}
