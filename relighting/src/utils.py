__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def compute_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss

def to_image(image):
    """ converts to PIL image """
    return Image.fromarray((outOfGamutClipping(image) * 255).astype(np.uint8))


def imshow(img, result=None, target=None, gt=None, title=None):
    """ displays image """
    if target is not None and gt is None:
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].set_title('target')
        ax[1].imshow(target)
        ax[1].axis('off')
        ax[2].set_title('result')
        ax[2].imshow(result)
        ax[2].axis('off')
    elif target is not None and gt is not None:
        fig, ax = plt.subplots(1, 4)
        ax[0].set_title('input')
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].set_title('target')
        ax[1].imshow(target)
        ax[1].axis('off')
        ax[2].set_title('result')
        ax[2].imshow(result)
        ax[2].axis('off')
        ax[3].set_title('gt')
        ax[3].imshow(gt)
        ax[3].axis('off')

    elif target is None and gt is not None:
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].set_title('result')
        ax[1].imshow(result)
        ax[1].axis('off')
        ax[2].set_title('gt')
        ax[2].imshow(gt)
        ax[2].axis('off')

    elif gt is None:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('input')
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[1].set_title('result')
        ax[1].imshow(result)
        ax[1].axis('off')


    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()
