import os
import shutil
import numpy as np
import scipy.io as io
from models.FRALUNet import FRLUnet
from models.FResUNet1 import FResUnet
from models.psp.pspnet import PSPNet
from models.Lpsp.Lpspnet import LPSPNet
from models.LFPSP.pspnet import FPSPNet
from models.deeplab_resnet import DeepLabv3_plus
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.RPLUnet import RPLUnet
from datasets.data3dunet import MyPatientFor3DUNet
from models.unet.unet_model import UNet
from models.model3dunet import UNet3D
from models.RPLFUnet import RPLFUnet
from models.UResNet import ResUnet as ResidualUNet3D
from models.Res_Unet import ResUnet
from models.ResLFPUNet import FLPResUnet
from resources.config import get_configs
from losses.losses import get_loss_criterion, DiceAccuracy

from trainer.trainerunet3d import UNet3DTrainer
from trainer.utils import get_logger

config = get_configs()




def _create_optimizer(config, model):
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _get_loaders(in_channels=1, out_channels=1, label_type='float32'):
    if config.model == 'UResNet':
        train_dataset = MyPatientFor3DUNet(in_channels=in_channels, out_channels=out_channels, step='train',
                                           data_type=label_type)
        val_dataset = MyPatientFor3DUNet(in_channels=in_channels, out_channels=out_channels, step='val',
                                         data_type=label_type)
        return {
            'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        }


def _para_():
    print("数据集路径: {}".format(config.dataset_path))

    print("batch_size: {}".format(config.batch_size))
    print("初始学习率: {}".format(config.learning_rate))
    print("weight_decay: {}".format(config.weight_decay))
    print("继续训练: {}".format(config.resume))
    print("模型保存路径: {}".format(config.checkpoint_dir))

    print("损失权重: {}".format(config.loss_weight))
    print("学习率调整间隔epoch: {}".format(config.patience))
    print("最大epoch: {}".format(config.epochs))
    print("模型初始特征图数量: {}".format(config.init_channel_number))

    print("损失函数: {}".format(config.loss))
    print("输出通道: {}".format(config.out_channels))
    print("sigmoid还是softmax: {}".format(config.final_sigmoid))
    print("测试模型路径: {}".format(config.model_path))

def main():
    path = config.dataset_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create loss criterion
    # if config.loss_weight is not None:
    #     weight = weight.split(':')
    #     loss_weight = torch.tensor((float(weight[0]), float(weight[1])))
    #     loss_weight = loss_weight.to(device)
    loss_weight = None
    loss_criterion = get_loss_criterion(config.loss, loss_weight, config.ignore_index)

    accuracy_criterion = DiceAccuracy(shrehold=0.5)

    # Get data loaders. If 'bce' or 'dice' loss is used, convert labels to float
    if config.loss in ['bce', 'gdl', 'dice']:
        label_dtype = 'float32'
    elif config.loss in ['ce', 'wce', 'focal', 'pce']:
        label_dtype = 'long'

    if config.model == 'UResNet':
        logger = get_logger('UNet3DTrainer')
        # model = DownUp(init_channel_number=config.init_channel_number)
        # model = UNet3D(config.in_channels, config.out_channels,
        #                init_channel_number=config.init_channel_number,
        #                conv_layer_order=config.layer_order,
        #                interpolate=config.interpolate,
        #                final_sigmoid=config.final_sigmoid)
        # model = F3Net()
        # model = ResidualUNet3D(config.in_channels,)
        # model = FPSPNet(config.in_channels)
        # model = UNet(1,1)
        # model = FLPResUnet(1)
        # model = ResUnet(1)
        # model = DeepLabv3_plus(nInputChannels=1, n_classes=1, os=16, pretrained=False, _print=True)
        # model = RPLUnet(1)
        # model = PSPNet(1)
        model = RPLFUnet(1)
        # model = FRLUnet(1)
        print(model)
        model = model.to(device)
        loaders = _get_loaders(in_channels=config.in_channels, out_channels=config.out_channels, label_type=label_dtype)


    # Log the number of learnable parameters
    # logger.info('Number of learnable params {}'.format(get_number_of_learnable_parameters(model)))

    # Create the optimizer
    optimizer = _create_optimizer(config, model)
    log_and_vaild = len(loaders['train'])
    _para_()
    print("训练集数量： ", len(loaders['train']))
    print("测试集数量： ", len(loaders['val']))
    if config.pertrain_path is not None:
        if os.path.exists("./checkpoint/logs/"):
            shutil.rmtree("./checkpoint/logs/")
        os.makedirs("./checkpoint/logs/")
        logger.info("from pertrain model: {}".format(config.pertrain_path))
        trainer = UNet3DTrainer.from_pertrain(config.pertrain_path, model, optimizer,
                                              loss_criterion,
                                              accuracy_criterion,
                                              device,
                                              loaders,
                                              config.checkpoint_dir,
                                              max_num_epochs=config.epochs,
                                              max_num_iterations=config.iters,
                                              max_patience=config.patience,
                                              patience=config.patience,
                                              validate_after_iters=log_and_vaild,
                                              log_after_iters=log_and_vaild,
                                              logger=logger)
    else:
        if config.resume:
            logger.info("from last model: {}".format(config.checkpoint_dir))
            trainer = UNet3DTrainer.from_checkpoint(config.resume, model, optimizer,
                                                    loss_criterion,
                                                    accuracy_criterion,
                                                    loaders,
                                                    validate_after_iters=log_and_vaild,
                                                    log_after_iters=log_and_vaild,
                                                    logger=logger)
        else:
            if os.path.exists("./checkpoint/logs/"):
                shutil.rmtree("./checkpoint/logs/")
            os.makedirs("./checkpoint/logs/")
            logger.info("new train.")
            trainer = UNet3DTrainer(model, optimizer,
                                    loss_criterion,
                                    accuracy_criterion,
                                    device, loaders, config.checkpoint_dir,
                                    max_num_epochs=config.epochs,
                                    max_num_iterations=config.iters,
                                    max_patience=config.patience,
                                    patience=config.patience,
                                    validate_after_iters=log_and_vaild,
                                    log_after_iters=log_and_vaild,
                                    logger=logger)

    trainer.fit()


if __name__ == '__main__':

    main()
