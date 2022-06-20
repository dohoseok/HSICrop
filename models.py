import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import LI, LIU, HU, HAMIDA, M3DDCNN, LEE, CNN_HSI, VSCNN, SSRN, SSTN, UNet, HSIUNet, TwoCNN
from config import NUM_CHANNELS, NUM_CLASSES

def get_model(**kwargs):
    name = kwargs["model"]
    print(name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.ones(NUM_CLASSES)
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "hamida":
        patch_size = kwargs.setdefault("patch_size", 5)
        model = HAMIDA(NUM_CHANNELS, NUM_CLASSES, patch_size=patch_size)
        lr = kwargs.setdefault("lr", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault("batch_size", 6400)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("expand_dims", 0)
    elif name == "M3DDCNN":
        kwargs.setdefault("patch_size", 7)
        model = M3DDCNN(NUM_CHANNELS, NUM_CLASSES, patch_size=kwargs["patch_size"])
        model.to(device)
        lr = kwargs.setdefault("lr", 0.01)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        kwargs.setdefault("batch_size", 1280)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("expand_dims", 0)
    elif name == "hu":
        kwargs.setdefault("patch_size", 1)
        model = HU(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 6400)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "lee":
        kwargs.setdefault("patch_size", 5)
        model = LEE(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 6400)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("expand_dims", 0)
        kwargs.setdefault("center_pixel", False)
    elif name == "li":
        patch_size = kwargs.setdefault("patch_size", 5)
        model = LI(NUM_CHANNELS, NUM_CLASSES, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault("lr", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        epoch = kwargs.setdefault("epoch", 200)
        kwargs.setdefault("batch_size", 6400)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("expand_dims", 0)
    elif name == "liu":
        patch_size = kwargs.setdefault("patch_size", 9)
        model = LIU(NUM_CHANNELS, NUM_CLASSES, patch_size)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 1280)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("expand_dims", 0)
    elif name == "CNN_HSI":
        patch_size = kwargs.setdefault("patch_size", 5)
        model = CNN_HSI(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        epoch = kwargs.setdefault("epoch", 10)
        kwargs.setdefault("batch_size", 1280)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("scheduler", None)
    elif name == "TwoCNN":
        kwargs.setdefault("patch_size", 21)
        model = TwoCNN(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.1)
        epoch = kwargs.setdefault("epoch", 10)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=0.8)
        kwargs.setdefault("scheduler", scheduler)
        criterion = nn.CrossEntropyLoss()
        kwargs.setdefault("expand_dims", 0)  
    elif name =="VSCNN":
        patch_size = kwargs.setdefault("patch_size", 13)
        model = VSCNN(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.001)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        epoch = kwargs.setdefault("epoch", 10)
        kwargs.setdefault("batch_size", 1280)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epoch//2, gamma=0.8)
        kwargs.setdefault("scheduler", scheduler)
        kwargs.setdefault("expand_dims", 0)
    elif name == "SSRN":
        patch_size = kwargs.setdefault("patch_size", 9)
        model = SSRN(NUM_CHANNELS)
        lr = kwargs.setdefault("lr", 0.002)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 1280)
        epoch = kwargs.setdefault("epoch", 20)
        kwargs.setdefault("scheduler", None)
    elif name == "SSTN":
        patch_size = kwargs.setdefault("patch_size", 9)
        model = SSTN(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.002)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 1280)
        epoch = kwargs.setdefault("epoch", 10)
        kwargs.setdefault("scheduler", None)
    elif name == "UNet":
        patch_size = kwargs.setdefault("patch_size", 128)
        model =UNet(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 64)
        epoch = kwargs.setdefault("epoch", 1000)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch, last_epoch=-1)
        kwargs.setdefault("scheduler", scheduler)
        kwargs.setdefault("center_pixel", False)
    elif name == "HSIUNet":
        patch_size = kwargs.setdefault("patch_size", 128)
        model =HSIUNet(NUM_CHANNELS, NUM_CLASSES)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 64)
        epoch = kwargs.setdefault("epoch", 1000)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch, last_epoch=-1)
        kwargs.setdefault("scheduler", scheduler)
        kwargs.setdefault("center_pixel", False)
    else:
        raise KeyError("{} model is unknown.".format(name))

    kwargs.setdefault("batch_size", 1280)
    epoch = kwargs.setdefault("epoch", 40)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    kwargs.setdefault("expand_dims", None)
    kwargs.setdefault("center_pixel", True)

    return model, optimizer, criterion, kwargs