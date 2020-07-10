import os
import shutil

import numpy as np
import scipy.io as io
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.data3dunet import MyPatientFor3DUNet

from models.model3dunet import UNet3D
from models.UResNet import ResUnet as ResidualUNet3D

from resources.config import get_configs
from trainer import utils
from losses.losses import DiceAccuracy
from models.downup import DownUp


config = get_configs()



step = 'val'


def predict(model, loader, device):
    """
    Return prediction masks by applying the model on the given dataset

    config:
        model (Unet3D): trained 3D UNet model used for prediction
        dataset (torch.utils.data.Dataset): input dataset
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         probability_maps (numpy array): prediction masks for given dataset
    """
    threhold = 0.5
    print('Running prediction on {} patches...'.format(len(loader[step])))
    accuracy_criterion = DiceAccuracy(shrehold=threhold)
    val_accuracy = utils.RunningAcc()

    model.eval()
    with torch.no_grad():
        sampel_num = 0
        for t in (loader[step]):
            # forward pass
            data, target, name = t
            data, target = data.to(device), target.to(device)
            probs = model(data)
            sampel_num += 1

            A, B, I = accuracy_criterion(probs.detach().cpu().numpy(), target.detach().cpu().numpy())
            val_accuracy.update(A, B, I)

            _save_mat(probs, name)

        print('sum_dice:',val_accuracy.sum_dice,'mean_dice:',val_accuracy.mean_dice/val_accuracy.count)


def _save_mat(probs,name):
    probs = probs.cpu().numpy()
    probs = np.squeeze(probs)
    # probs = np.transpose(probs,(0,2,1))  Y
    # probs = np.transpose(probs, (2, 0, 1))  Z
    probs = np.transpose(probs, (0, 1, 2))
    io.savemat("./results/X/test/{}.mat".format(name[0].split('_')[0]), {'data': probs})


def _get_loaders(in_channels=1, out_channels=1, label_type='float32'):
    val_dataset = MyPatientFor3DUNet(in_channels=in_channels, out_channels=out_channels, step='val', data_type=label_type)
    train_dataset = MyPatientFor3DUNet(in_channels=in_channels, out_channels=out_channels, step='train', data_type=label_type)
    return {
        'val': DataLoader(val_dataset, batch_size=480, shuffle=False),
        'train': DataLoader(train_dataset, batch_size=480, shuffle=False)
    }


def main():
    if config.model == 'UResNet':
        # model = DownUp(init_channel_number=config.init_channel_number)
        # model = UNet3D(in_channels, out_channels,
        #                init_channel_number=config.init_channel_number,
        #                final_sigmoid=final_sigmoid,
        #                interpolate=interpolate,
        #                conv_layer_order=layer_order)
        model = ResidualUNet3D(config.in_channels)
    print('Loading model from {}...'.format(config.model_path))
    utils.load_checkpoint(config.model_path, model)

    print('Loading datasets...')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    # Get data loaders. If 'bce' or 'dice' loss is used, convert labels to float
    if config.loss in ['bce', 'gdl', 'dice']:
        label_dtype = 'float32'
    elif config.loss in ['ce', 'wce']:
        label_dtype = 'long'
    loader = _get_loaders(in_channels=config.in_channels, out_channels=config.out_channels, label_type=label_dtype)

    predict(model, loader, device)


if __name__ == '__main__':
    main()

