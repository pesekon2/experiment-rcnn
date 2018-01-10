#!/usr/bin/env python
#
############################################################################
#
# MODULE:	    ann.maskrcnn.train
# AUTHOR(S):	Ondrej Pesek <pesej.ondrek@gmail.com>
# PURPOSE:	    Train your Mask R-CNN network
# COPYRIGHT:	(C) 2017 Ondrej Pesek and the GRASS Development Team
#
#		This program is free software under the GNU General
#		Public License (>=v2). Read the file COPYING that
#		comes with GRASS for details.
#
#############################################################################

#%module
#% description: Train your Mask R-CNN network
#% keyword: ann
#% keyword: vector
#% keyword: raster
#%end
#%option G_OPT_M_DIR
#% key: training_dataset
#% label: Path to the dataset with images and masks
#% required: yes
#%end
#%option G_OPT_V_OUTPUT
#% key: output
#%end
#%option
#% key: model
#% label: Which model to use
#% required: yes
#% options: coco, last
#% multiple: no
#%end


import grass.script as gscript
from grass.pygrass.utils import get_lib_path
import sys
import os
from subprocess import call

path = get_lib_path(modname='maskrcnn', libname='py3train')
if path is None:
    grass.script.fatal('Not able to find the maskrcnn library directory.')

###########################################################
# unfortunately, it needs python3, see file py3train.py
###########################################################
# sys.path.append(path)
# from configs import ModelConfig


def main(options, flags):

    dataset = options['training_dataset']
    initialWeights = options['model']
    logs = "/home/ondrej/workspace/experiment-rcnn/logs"

    print("Model: ", initialWeights)
    print("Dataset: ", dataset)
    print("Logs: ", logs)

    ###########################################################
    # unfortunately, redirect everything to python3
    ###########################################################
    print('python3 {}{}py3train.py'.format(path, os.sep))
    call('python3 {}{}py3train.py --dataset={} --model={} --logs={}'.format(
            path, os.sep, dataset, initialWeights, logs),
         shell=True)


if __name__ == "__main__":
    options, flags = gscript.parser()
    main(options, flags)
