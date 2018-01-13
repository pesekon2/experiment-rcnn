
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[ ]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize



# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[ ]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3
    NAME = 'ondra'

config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights

# In[ ]:

def detect(imagesDir, modelPath, classes, name, masksDir):
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=modelPath, config=config)

    # Load weights trained on MS-COCO

    model.load_weights(modelPath, by_name=True)


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    classNames = ['BG']
    for i in classes:
        classNames.append(i)


    # ## Run Object Detection

    # In[ ]:


    # Load a random image from the images folder
    file_names = next(os.walk(imagesDir))[2]
    a = random.choice(file_names)
    image = skimage.io.imread(os.path.join(imagesDir, a))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    print('Stat:', image, r['rois'], r['masks'], r['class_ids'],
                                classNames, r['scores'])
    print(a)
    print('NEXT VISUALIZE')
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                classNames, r['scores'])


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument('--images_dir', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--classes', required=True,
                        help="Names of classes")
    parser.add_argument('--name', required=True,
                        help='Name of output models')
    parser.add_argument('--masks_dir', required=True,
                        help='Name of output models')

    args = parser.parse_args()

    detect(args.images_dir, args.model, args.classes.split(','), args.name,
          args.masks_dir)
