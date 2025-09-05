import torch.optim as optim


def load_scheduler(model_name, model):
    optimizer, scheduler = None, None
    
    if model_name == 'm3ddcnn':
        optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.01)
        scheduler = None

    elif model_name == '3DCNN':
        # MaxEpoch in the paper is unknown, so 300 is set as MaxEpoch
        # and paper said: for each (Max Epoch / 3) iteration, the learning rate is divided by 10
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'rssan':
        optimizer = optim.RMSprop(model.parameters(), lr=0.0003, weight_decay=0.0, momentum=0.0)
        scheduler = None

    elif model_name == 'ablstm':
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1)

    elif model_name == 'DFFN':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'HybridSN':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    elif model_name == 'AMFAN':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    elif model_name == 'SpectalFormer':
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120, 150, 180, 210, 240, 270], gamma=0.9)

    elif model_name == 'SSFTT' or model_name == 'SSFT_FDR':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    elif model_name == 'GAHT':
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        scheduler = None

    elif model_name == 'BS2T':
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = None

    elif model_name == 'morphFormer':
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    elif model_name == 'MHIAIFormer':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        scheduler = None

    elif model_name == 'LMSS-NAS':
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100 // 5, gamma=0.5)

    elif model_name == 'LRDTN' or model_name == 'LRDTN_FDR':
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = None

    elif model_name == 'ELS2T':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None

    elif model_name == 'HybridFormer':
        optimizer = optim.SGD(model.parameters(), lr=0.0001)
        scheduler = None

    elif model_name == 'A2S2K':
        scheduler = None
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    elif model_name == 'AMF':
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-7, last_epoch=-1)

    elif model_name == 'CLOLN':
        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)

    elif model_name == 'LS2CM':
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)

    elif model_name == 'GhostNet':
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)

    elif model_name == 'CSCANet' or model_name == 'CSCANet_FDR':
        scheduler = None
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

    elif model_name == 'DSNet':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100//10, gamma=0.9)

    elif model_name == 'ACB':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = None

    elif model_name == 'ULite_TAA':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        scheduler = None


    # ==================================================================
    # Mamba models
    # CenterMamba, HyperMamba, IGroupSSMamba, MambaHSI, 3DSSMamba(SSMamba3D)
    # ==================================================================
    elif model_name == 'CenterMamba':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif model_name == 'HyperMamba':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif model_name == 'IGroupSSMamba':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None
    elif model_name == 'MambaHSI':
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        scheduler = None
    elif model_name == '3DSSMamba':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = None



    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        scheduler = None

    return optimizer, scheduler


