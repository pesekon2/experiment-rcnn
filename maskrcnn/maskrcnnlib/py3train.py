#!/usr/bin/env python
#
############################################################################
#
# MODULE:	    py3train
# AUTHOR(S):	Ondrej Pesek <pesej.ondrek@gmail.com>
# PURPOSE:	    A python3 script called to train your Mask R-CNN network
# COPYRIGHT:	(C) 2017 Ondrej Pesek and the GRASS Development Team
#
#		This program is free software under the GNU General
#		Public License (>=v2). Read the file COPYING that
#		comes with GRASS for details.
#
#############################################################################


import os
import sys
import time
import glob
import cv2
import numpy as np
# from PIL import Image
from random import shuffle
import skimage

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from config import ModelConfig
import utils
import model as modellib

from sys import exit

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Dataset
############################################################

class Dataset(utils.Dataset):
    def load(self, classes, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        self.classes = {'BG': 0}
        for i in range(1, len(classes) + 1):
            self.add_class('ondra', i, classes[i - 1])
            self.classes.update({classes[i - 1]: i})

        for path in subset:
            for image in glob.iglob(os.path.join(path, '*.jpg')):
                self.add_image('ondra', image_id = os.path.split(path)[1], path=image)


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        a = glob.glob(os.path.join(os.path.split(info['path'])[0], '*.png'))
        maskImage = skimage.io.imread(a[0])
        mask = np.zeros([maskImage.shape[0], maskImage.shape[1], 1])
        maskAppend = np.zeros([maskImage.shape[0], maskImage.shape[1], 1])
        class_ids = np.array([self.classes[a[0].split('-')[-2]]])
        mask[:, :, 0] = maskImage

        for i in range(1, len(a)):
            np.append(class_ids, self.classes[a[i].split('-')[-2]])
            maskAppend[:, :, 0] = skimage.io.imread(a[i])
            np.concatenate((mask, maskAppend), 2)

        return mask, class_ids

############################################################
#  Training
############################################################


def train(dataset, modelPath, classes, logs=DEFAULT_LOGS_DIR, epochs=200,
          steps_per_epoch=3000, ROIsPerImage=64):

    print("Model: ", modelPath)
    print("Dataset: ", dataset)
    print("Logs: ", logs)
    print(steps_per_epoch, type(steps_per_epoch))

    # Configurations
    # TODO: Make parameters from these given in Config
    config = ModelConfig(name='ondra', imagesPerGPU=1, GPUcount=1, numClasses=3,
                    trainROIsPerImage=ROIsPerImage, stepsPerEpoch=steps_per_epoch,
                    miniMaskShape=(128, 128), validationSteps=100,
                    imageMaxDim=256*3, imageMinDim=256*3)
    config.display()

    print(classes, type(classes))

    raise SystemExit(0)

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=logs)

    # Load weights
    images = list()
    for root, subdirs, _ in os.walk(dataset):
        if not subdirs:
            images.append(root)

    shuffle(images)

    testImagesThreshold = int(len(images) * .9)
    evalImagesThreshold = int(testImagesThreshold * .75)
    trainImages = images[:evalImagesThreshold]
    evalImages = images[evalImagesThreshold:testImagesThreshold]

    print('List of unused images saved in: /home/ondrej/unused.txt')
    with open('/home/ondrej/unused.txt', 'w') as unused:
        for filename in images[testImagesThreshold:]:
            unused.write('{}\n'.format(filename))

    print("Loading weights ", modelPath)
    # TODO: Make as a parameter
    if "coco.h5" in modelPath:
        model.load_weights(modelPath, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(modelPath, by_name=True)

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = Dataset()
    dataset_train.load(classes, trainImages)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Dataset()
    dataset_val.load(classes, evalImages)
    dataset_val.prepare()

    # Training - Stage 1
    # Adjust epochs and layers as needed
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=int(epochs / 7),
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10, # without dividing in original
                epochs=int(epochs / 7) * 3,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100, # just 10 original
                epochs=epochs,
                layers='all')

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--classes', required=True,
                        help="Names of classes")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--epochs', required=False,
                        default=200, type=int,
                        help='Number of epochs')
    parser.add_argument('--steps_per_epoch', required=False,
                        default=3000, type=int,
                        help='Number of steps per each epoch')
    parser.add_argument('--rois_per_image', required=False,
                        default=64, type=int,
                        help='Number of ROIs trained per each image')

    args = parser.parse_args()

    train(args.dataset, args.model, args.classes.split(','), args.logs, args.epochs,
          args.steps_per_epoch, args.rois_per_image)