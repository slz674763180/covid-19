import logging
import os
import cv2
import numpy as np
import torch
# from tensorboardX import SummaryWriter
import torch.nn as nn
from trainer import utils
import torch.nn.functional as F



class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        loss_criterion (callable): loss function
        accuracy_criterion (callable): used to compute training/validation accuracy (such as Dice or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        max_patience (int): number of validation runs with no improvement
            after which the training will be stopped
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        best_val_accuracy (float): best validation accuracy so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, loss_criterion_1,
                 accuracy_criterion,
                 device, loaders, checkpoint_dir,
                 max_num_epochs=200, max_num_iterations=1e5, max_patience=100, patience=20,
                 validate_after_iters=100, log_after_iters=100,
                 best_val_accuracy=float('-inf'),
                 num_iterations=0, num_epoch=0, logger=None):
        if logger is None:
            self.logger = utils.get_logger('UNet3DTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.logger.info("Sending the model to '{}'".format(device))
        self.model = model.to(device)
        self.logger.debug(model)

        self.optimizer = optimizer
        self.loss_criterion_1 = loss_criterion_1
        self.accuracy_criterion = accuracy_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.best_val_accuracy = best_val_accuracy
        # self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        # used for early stopping
        self.max_patience = max_patience
        self.patience = patience
        # for param_group in self.optimizer.param_groups:
        #     print(param_group['lr'])

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, loss_criterion_1,
                        accuracy_criterion, loaders,
                        validate_after_iters,
                        log_after_iters,
                        logger=None):
        logger.info("Loading checkpoint '{}'...".format(checkpoint_path))
        # 修改此处的state可以使网络重新训练
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        # state = utils.load_checkpoint(checkpoint_path, model)
        logger.info(
            "Checkpoint loaded. Epoch: {}. Best val accuracy: {:.5f}. Num_iterations: {}".format(
                state['epoch'], state['best_val_accuracy'], state['num_iterations']
            ))
        # state['num_iterations'] = 0
        # state['epoch'] = 0
        # state['max_num_epochs'] = 200
        state['patience'] = 50
        state['max_patience'] = 50
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, loss_criterion_1, accuracy_criterion, torch.device(state['device']), loaders,
                   checkpoint_dir,
                   best_val_accuracy=state['best_val_accuracy'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   max_patience=state['max_patience'],
                   patience=state['patience'],
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   logger=logger)

    @classmethod
    def from_pertrain(cls, pertrain_path, model, optimizer, loss_criterion_1,
                      accuracy_criterion,
                      device, loaders, checkpoint_dir,
                      max_num_epochs=200, max_num_iterations=1e5, max_patience=20, patience=100,
                      validate_after_iters=100, log_after_iters=100,
                      best_val_accuracy=float('-inf'),
                      num_iterations=0, num_epoch=0, logger=None):
        # 修改此处的state可以使网络重新训练
        # state = utils.load_checkpoint(pertrain_path, model, optimizer)
        state = utils.load_checkpoint(pertrain_path, model)
        # checkpoint_dir = os.path.split(checkpoint_dir)
        return cls(model, optimizer, loss_criterion_1, accuracy_criterion,
                   device, loaders, checkpoint_dir,
                   best_val_accuracy=best_val_accuracy,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=int(max_num_iterations),
                   max_patience=max_patience,
                   patience=patience,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   logger=logger)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1
            # if self.num_epoch % 100 == 80:
            #     self._adjust_learning_rate()

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """

        # val_accuracy = self.validate(self.loaders['val'])

        train_losses = utils.RunningAverage()
        train_accuracy = utils.RunningAcc()

        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            # self.logger.info(
            #     'Training iteration {}. Batch {}. Epoch [{}/{}]'.format(
            #         self.num_iterations, i, self.num_epoch, self.max_num_epochs - 1
            #     ))

            if len(t) == 2:
                input, target = t
                input, target = input.to(self.device), target.to(self.device)
            if len(t) == 3:
                input, target, target_ = t
                input, target, target_ = input.to(self.device), target.to(self.device), target_.to(self.device)
            if len(t) == 6:
                input, target, aug1_in, aug1_ta, aug2_in, aug2_ta = t
                input = torch.cat([input, aug1_in], 0)
                target = torch.cat([target, aug1_ta], 0)
                input = torch.cat([input, aug2_in], 0)
                target = torch.cat([target, aug2_ta], 0)
                input, target = input.to(self.device), target.to(self.device)

            output, loss = self._forward_pass(input, target, target_)

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.num_iterations += 1

            # output = self.model.final_activation(output)
            A, B, I = self.accuracy_criterion(output.detach().cpu().numpy(), target.detach().cpu().numpy())
            train_losses.update(loss.item(), input.size(0))
            train_accuracy.update(A, B, I)

            if self.num_iterations % self.log_after_iters == 0:
                # log stats, params and images
                self.logger.info(
                    'Training stats. Loss: {:.4f}. sum_dice: {:.4f}  mean_dice: {:.4f}'.format(
                        train_losses.avg, train_accuracy.sum_dice, train_accuracy.mean_dice / train_accuracy.count))

                # self._log_params()

            if self.num_iterations % self.validate_after_iters == 0:
                # evaluate on validation set
                val_accuracy = self.validate(self.loaders['val'])

                # remember best validation metric
                is_best = self._is_best_val_accuracy(val_accuracy)

                # save checkpoint
                self._save_checkpoint(is_best)

                if self._check_early_stopping(is_best):
                    self.logger.info(
                        'Validation accuracy did not improve for the last {} validation runs. Early stopping...'.format(
                            self.max_patience
                        ))
                    return True

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    'Maximum number of iterations {} exceeded. Finishing training...'.format(
                        self.max_num_iterations
                    ))
                return True

        return False

    def validate(self, val_loader):
        self.logger.info('epoch: {}, Validating...'.format(self.num_epoch))

        val_losses = utils.RunningAverage()
        val_accuracy = utils.RunningAcc()
        self.model.eval()
        try:
            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    # self.logger.info('Validation iteration {}'.format(i))
                    if len(t) == 2:
                        input, target = t
                        input, target = input.to(self.device), target.to(self.device)
                    if len(t) == 3:
                        input, target, target_ = t
                        input, target, target_ = input.to(self.device), target.to(self.device), target_.to(self.device)
                    if len(t) == 4:
                        input, target, name = t
                        input, target = input.to(self.device), target.to(self.device)

                    output, loss = self._forward_pass(input, target, target_)

                    # img = output.detach().cpu().numpy()
                    # img = np.squeeze(img)
                    # img = np.rint(img)
                    # batch = np.shape(img)[0]
                    # for i in range(batch):
                    #     img1 = img[i, :, :]
                    #     cv2.imwrite('/home/slz/PycharmProjects/Res_Unet/result/Y/' + name[i], img1)

                    A, B, I = self.accuracy_criterion(
                        output.detach().cpu().numpy(),
                        target.detach().cpu().numpy())

                    val_losses.update(loss.item(), input.size(0))
                    val_accuracy.update(A, B, I)

                self._log_stats('val', val_losses.avg, val_accuracy.sum_dice,
                                val_accuracy.mean_dice / val_accuracy.count)
                self.logger.info(
                    'Validation finished. Loss: {:.8f}. sum_dice: {:.4f}, mean_dice:{:.4f}'.format(
                        val_losses.avg, val_accuracy.sum_dice, val_accuracy.mean_dice / val_accuracy.count))
                self.logger.info('Validation finished. BestAccuracy: {}'.format(self.best_val_accuracy))
                return val_accuracy.mean_dice / val_accuracy.count
        finally:
            self.model.train()

    def _forward_pass(self, input, target, target_):
        # output = self.model(input)
        # loss = self.loss_criterion_1(output, target)

        down_x, output = self.model(input)

        n = len(down_x)
        loss = 0
        loss += self.loss_criterion_1(down_x[0], target_) * 0.5
        loss += self.loss_criterion_1(down_x[1], target_) * 0.5
        for i in range(2, n):
            loss += self.loss_criterion_1(down_x[i], target) * 0.4
        #     # sum += abs(self.model.w[i]*0.4)
        # # loss += self.loss_criterion_1(down_x[4], target) * abs(2-sum)
        # # sum += abs(self.model.w[i])
        # # loss /= sum
        loss += self.loss_criterion_1(output, target)*2

        # loss = []
        # loss.append(self.loss_criterion_1(output, target))
        #
        # for i in range(n):
        #     loss.append(self.loss_criterion_1(down_x[i],target))
        # a = torch.Tensor(loss).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
        # losses = conv(a)
        # weight = conv.weight
        # losses = torch.mean(losses) / weight.sum()
        # print(weight)

        return output, loss

    def _check_early_stopping(self, best_model_found):
        """
        Check patience and adjust the learning rate if necessary.
        :param best_model_found: is current model the best one according to validation criterion
        :return: True if the training should be terminated, False otherwise
        """
        # if self.num_epoch % 100 == 99:
        #     self._adjust_learning_rate()
        if best_model_found:
            self.patience = self.max_patience
            print(self.patience)
        else:
            self.patience -= 1
            if self.patience <= 0:
                # early stop the training
                return True
            # adjust learning rate when reaching half of the max_patience
            if self.patience == self.max_patience // 2:
                self._adjust_learning_rate()
            print(self.patience)
            # self.patience = self.max_patience
        return False

    def _adjust_learning_rate(self, decay_rate=0.1):
        """Sets the learning rate to the initial LR decayed by 'decay_rate'"""

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        old_lr = get_lr(self.optimizer)
        assert old_lr > 0
        new_lr = decay_rate * old_lr
        self.logger.info('Changing learning rate from {} to {}'.format(old_lr, new_lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _is_best_val_accuracy(self, val_accuracy):
        is_best = val_accuracy > self.best_val_accuracy
        if is_best:
            self.logger.info(
                'Saving new best validation accuracy: {}'.format(val_accuracy))
        self.best_val_accuracy = max(val_accuracy, self.best_val_accuracy)
        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'max_patience': self.max_patience,
            'patience': self.patience
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_stats(self, phase, loss_avg, sum_dice, mean_dice):
        tag_value = {
            '{}_loss_avg'.format(phase): loss_avg,
            '{}_accuracy_avg'.format(phase): sum_dice,
            '{}_hit_avg'.format(phase): mean_dice,
        }

        # for tag, value in tag_value.items():
        #     self.writer.add_scalar(tag, value, self.num_iterations)

    # def _log_params(self):
    #     self.logger.info('Logging model parameters and gradients')
    #     for name, value in self.model.named_parameters():
    #         self.writer.add_histogram(name, value.data.cpu().numpy(),
    #                                   self.num_iterations)
    #         self.writer.add_histogram(name + '/grad',
    #                                   value.grad.data.cpu().numpy(),
    #                                   self.num_iterations)
