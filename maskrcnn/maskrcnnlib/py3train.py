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
import cv2
import numpy as np
from random import shuffle
import skimage
import argparse

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


#  Training

def train(dataset, modelPath, classes, logs, modelName, epochs=200,
          steps_per_epoch=3000, ROIsPerImage=64, flags=''):

    print("Model: ", modelPath)
    print("Dataset: ", dataset)
    print("Logs: ", logs)

    # Configurations
    # TODO: Make as user parameters
    config = ModelConfig(name=modelName,
                         imagesPerGPU=1,
                         GPUcount=1,
                         numClasses=len(classes) + 1,
                         trainROIsPerImage=ROIsPerImage,
                         stepsPerEpoch=steps_per_epoch,
                         miniMaskShape=(128, 128),
                         validationSteps=100,
                         imageMaxDim=256*3,
                         imageMinDim=256*3)
    config.display()

    # raise SystemExit(0)

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=logs)

    # Load weights
    images = list()
    for root, subdirs, _ in os.walk(dataset):
        if not subdirs:
            images.append(root)

    shuffle(images)

    if 's' in flags:
        testImagesThreshold = int(len(images) * .9)
        print('List of unused images saved in the logs file')
        with open(logs, 'w') as unused:
            for filename in images[testImagesThreshold:]:
                unused.write('{}\n'.format(filename))
    else:
        testImagesThreshold = len(images)

    evalImagesThreshold = int(testImagesThreshold * .75)
    trainImages = images[:evalImagesThreshold]
    evalImages = images[evalImagesThreshold:testImagesThreshold]

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
    dataset_train = utils.Dataset()
    dataset_train.import_contains(classes, trainImages, modelName)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = utils.Dataset()
    dataset_val.import_contains(classes, evalImages, modelName)
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
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    parser.add_argument('--name', required=True,
                        help='Name of output models')
    parser.add_argument('--epochs', required=False,
                        default=200, type=int,
                        help='Number of epochs')
    parser.add_argument('--steps_per_epoch', required=False,
                        default=3000, type=int,
                        help='Number of steps per each epoch')
    parser.add_argument('--rois_per_image', required=False,
                        default=64, type=int,
                        help='Number of ROIs trained per each image')
    parser.add_argument('--flags', required=False,
                        default='',
                        help='Flags')

    args = parser.parse_args()

    train(args.dataset, args.model, args.classes.split(','), args.logs,
          args.name, args.epochs, args.steps_per_epoch, args.rois_per_image,
          flags)