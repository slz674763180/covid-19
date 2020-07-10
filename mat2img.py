import numpy as np
import scipy.io as io
import os
import cv2

path = '/home/hhg/SLZ/copy/dataset/3DPET/'
s = 'X'


def train():
    pet_list = os.listdir(path + 'train/pet')
    for name in pet_list:
        pet = io.loadmat(path + 'train/pet/' + name)['data']
        mask = io.loadmat(path + 'train/mask/' + name)['data']
        pet = pet[1:-1, :, :]
        mask = mask[1:-1, :, :]
        m = np.shape(pet)[2]
        if m < 480:
            pet_temp = np.zeros((48, 96, 480), dtype=np.float32)
            mask_temp = np.zeros((48, 96, 480), dtype=np.float32)
            pet_temp[:, :, :m] = pet
            pet = pet_temp
            mask_temp[:, :, :m] = mask
            mask = mask_temp
        if m > 480:
            pet = pet[:, :, :480]
            mask = mask[:, :, :480]

        if s == 'Y':
            pet = np.transpose(pet, [2, 1, 0])
            mask = np.transpose(mask, [2, 1, 0])
        if s == 'Z':
            pet = np.transpose(pet, [2, 0, 1])
            mask = np.transpose(mask, [2, 0, 1])

        n = np.shape(pet)[2]
        for i in range(n):
            cv2.imwrite(
                '/home/hhg/SLZ/copy/dataset/2DPET/' + s + '/train/pet/' + name.split('.')[0] + '_' + str(i) + '.png',
                pet[:, :, i])
            cv2.imwrite(
                '/home/hhg/SLZ/copy/dataset/2DPET/' + s + '/train/mask/' + name.split('.')[0] + '_' + str(i) + '.png',
                mask[:, :, i])


def val():
    pet_list = os.listdir(path + 'val/pet')
    for name in pet_list:
        pet = io.loadmat(path + 'val/pet/' + name)['data']
        mask = io.loadmat(path + 'val/mask/' + name)['data']

        pet = pet[1:-1, :, :]
        mask = mask[1:-1, :, :]
        m = np.shape(pet)[2]
        if m < 480:
            pet_temp = np.zeros((48, 96, 480), dtype=np.float32)
            mask_temp = np.zeros((48, 96, 480), dtype=np.float32)
            pet_temp[:, :, :m] = pet
            pet = pet_temp
            mask_temp[:, :, :m] = mask
            mask = mask_temp
        if m > 480:
            pet = pet[:, :, :480]
            mask = mask[:, :, :480]

        if s == 'Y':
            pet = np.transpose(pet, [2, 1, 0])
            mask = np.transpose(mask, [2, 1, 0])
        if s == 'Z':
            pet = np.transpose(pet, [2, 0, 1])
            mask = np.transpose(mask, [2, 0, 1])
        n = np.shape(pet)[2]
        for i in range(n):
            cv2.imwrite(
                '/home/hhg/SLZ/copy/dataset/2DPET/' + s + '/val/pet/' + name.split('.')[0] + '_' + str(i) + '.png',
                pet[:, :, i])
            cv2.imwrite(
                '/home/hhg/SLZ/copy/dataset/2DPET/' + s + '/val/mask/' + name.split('.')[0] + '_' + str(i) + '.png',
                mask[:, :, i])


if __name__ == '__main__':
    train()
    val()
