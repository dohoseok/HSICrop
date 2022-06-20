import os
import argparse
import numpy as np

import torch

from HSICropDataset import HSICropData
from models import get_model
from utils import metrics, save_results, save_visible
from config import *

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="HSIUNet",
        help="Model to test. Available:\n"
            "hamida, M3DDCNN, hu, lee, li, liu, CNN_HSI, TwoCNN, VSCNN, SSRN, SSTN, UNet, HSIUNet")
    parser.add_argument("--gpu_id", type=str, default=GPUS, help="GPU ID")
    parser.add_argument('--weight', type=str, required=True, help='Load weight from a .pth file')

    return parser.parse_args()


def test(weight_file, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hyperparams = vars(args)
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    model, _, _, hyperparams = get_model(**hyperparams)

    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(weight_file, map_location=device))

    test_dataset = HSICropData("test", DATA_PATH, **hyperparams)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle = False,
            num_workers = NUM_WORKERS
        )

    predictions = None
    gts = None
    logits_list = None

    for data, target in test_loader:
        data = data.to(device)
        logits = model(data)
        if hyperparams['center_pixel'] is True:
            if type(logits) is tuple:
                output, _ = logits
                pred = torch.argmax(output, dim=1).cpu().detach().numpy().flatten()
            else:
                pred = torch.argmax(logits, dim=1).cpu().detach().numpy().flatten()

            if predictions is None:
                predictions = pred
            else:
                predictions = np.append(predictions, pred, axis=0)
            target = target.numpy().flatten()

        else:
            logits = np.transpose(logits.cpu().detach().numpy(), (0, 2, 3, 1))
            if logits_list is None:
                logits_list = logits
            else:
                logits_list = np.append(logits_list, logits, axis=0)

        if gts is None:
            gts = target
        else:
            gts = np.append(gts, target, axis=0)

    if hyperparams['center_pixel'] is False:
        patch_size = hyperparams['patch_size']
        NUM_IMG = len(test_dataset) // ((129-patch_size) * (129-patch_size))
        probs = np.zeros((NUM_IMG, 128, 128, NUM_CLASSES))
        gts_reshape = np.zeros((NUM_IMG, 128, 128))
        logits = logits_list.reshape((NUM_IMG, 129 - patch_size, 129 - patch_size, patch_size, patch_size, NUM_CLASSES))
        gts = gts.reshape((NUM_IMG, 129 - patch_size, 129 - patch_size, patch_size, patch_size))
        for b in range(NUM_IMG):
            for x in range(129 - patch_size):
                for y in range(129 - patch_size):
                    probs[b, x:x+patch_size, y:y+patch_size, :] += logits[b, x, y]
                    gts_reshape[b, x:x+patch_size, y:y+patch_size] = gts[b, x, y]

        probs = np.argmax(probs, -1)
        predictions = probs.flatten()
        gts = gts_reshape.flatten()

    run_results = metrics(
        predictions,
        gts,
        n_classes=NUM_CLASSES,
    )


    weight_path = os.path.split(weight_file)[0]
    result_path = os.path.join(weight_path, "result")
    
    save_visible(predictions, result_path, os.path.join(DATA_PATH, "test.txt"))

    result_filename = os.path.join(weight_path, f"{os.path.basename(weight_path)}.xlsx")
    save_results(run_results, result_filename)


if __name__ == "__main__":
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    test(args.weight, args)