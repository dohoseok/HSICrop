import os
import argparse
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim

from test import test
from HSICropDataset import HSICropData
from models import get_model
from utils import save_plt_fig
from config import *


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="HSIUNet",
        help="Model to train. Available:\n"
            "hamida, M3DDCNN, hu, lee, li, liu, CNN_HSI, TwoCNN, VSCNN, SSRN, SSTN, UNet, HSIUNet")
    parser.add_argument('--epoch', metavar='E', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', metavar='B', type=int, help='Batch size')
    parser.add_argument('--lr', metavar='LR', type=float, help='Learning rate')
    parser.add_argument("--gpu_id", type=str, default=GPUS, help="GPU ID")

    return parser.parse_args()


def validate(model, data_loader, device, criterion):
    model.eval()
    accuracy, total, evalLoss = 0.0, 0.0, []
    for (image, target) in data_loader:
        with torch.no_grad():
            image = image.to(device)
            target = target.to(device)
            
            if type(criterion) is tuple:
                output, rec = model(image)
                loss  = criterion[0](output, target) + model.module.aux_loss_weight * criterion[1](rec, image)
            else:
                output = model(image)
                loss = criterion(output, target)
            evalLoss.append(loss.item())

            output = torch.argmax(output, dim=1)
            for out, gt in zip(output.view(-1), target.view(-1)):
                accuracy += out.item() == gt.item()
                total += 1
    return accuracy/total, np.mean(evalLoss)


def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    hyperparams = vars(args)
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    model, optimizer, criterion, hyperparams = get_model(**hyperparams)
    print(hyperparams)


    today = time.localtime()
    folder_path = '{:02d}{:02d}{:02d}_{:02d}{:02d}'.format(today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour, today.tm_min)
    weight_path_prefix = "./weights"
    weight_save_path = os.path.join(weight_path_prefix, hyperparams['model'], folder_path)
    os.makedirs(weight_save_path, exist_ok=True)

    print(model)
    print(folder_path)

    model = torch.nn.DataParallel(model)
    model.to(device)

    train_dataset = HSICropData("train", DATA_PATH, **hyperparams)
    val_dataset = HSICropData("test", DATA_PATH, **hyperparams)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            num_workers = NUM_WORKERS
            
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            num_workers = NUM_WORKERS
        )

    EPOCHS = hyperparams['epoch']
    scheduler = hyperparams['scheduler']
    if EPOCHS > 500:
        val_intv = 10
    elif EPOCHS > 100:
        val_intv = 5
    else:
        val_intv = 1

    epoch_list = []
    loss_list = []
    acc_list = []
    val_acc = 0.1

    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        model.train()
        
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for (image, target) in train_loader:
                image = image.to(device)
                target = target.to(device)

                if type(criterion) is tuple:
                    output, rec = model(image)
                    loss  = criterion[0](output, target) + model.module.aux_loss_weight * criterion[1](rec, image)
                else:
                    output = model(image)
                    loss = criterion(output, target)

                optimizer.zero_grad()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                loss.backward()
                if "UNet" in hyperparams['model']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

                pbar.update(image.shape[0])
            
            if (epoch + 1) % val_intv == 0:
                val_acc, val_loss = validate(model, val_loader, device, criterion)
                print(f"Epoch : {epoch + 1}, validate accuracy : {val_acc:.4f}, validate loss : {val_loss:.4f}")
                epoch_list.append(epoch+1)
                loss_list.append(val_loss)
                acc_list.append(val_acc)

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(-val_acc)
            elif scheduler is not None:
                scheduler.step()

            weight_save_file = os.path.join(weight_save_path, 'latest.pth')
            torch.save(model.module.state_dict(), weight_save_file)

    save_plt_fig(epoch_list, loss_list, 'epoch', 'loss', weight_save_path)
    save_plt_fig(epoch_list, acc_list, 'epoch', 'Accuracy', weight_save_path)

    return weight_save_file

if __name__ == "__main__":
    args = get_args()

    weight_save_file = train(args)
    print("weight_save_file", weight_save_file)

    test(weight_save_file, args)