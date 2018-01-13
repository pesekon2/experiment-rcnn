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
#% key: images_directory
#% label: Path to a directory with images to detect
#% required: yes
#%end
#%option
#% key: model
#% type: string
#% label: Path to the .h5 file to use as initial values
#% required: yes
#% multiple: no
#%end
#%option
#% key: classes
#% type: string
#% label: Names of classes separated with ","
#% required: yes
#% multiple: yes
#%end
#%option
#% key: name
#% type: string
#% label: Name for output models
#% required: yes
#%end
#%option G_OPT_M_DIR
#% key: maks_output
#% type: string
#% label: Directory where masks will be saved
#% required: yes
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

    imagesDir = options['images_directory']
    modelPath = options['model']
    classes = options['classes']
    name = options['name']
    # TODO: Use GRASS temp files
    masksDir = options['mask_output']

    ###########################################################
    # unfortunately, redirect everything to python3
    ###########################################################
    call('python3 {}{}py3train.py --images_dir={} --model={} --classes={} '
         '--name={} --masks_dir={}'.format(
            path, os.sep,
            imagesDir,
            modelPath,
            classes,
            name,
            masksDir),
         shell=True)


if __name__ == "__main__":
    options, flags = gscript.parser()
    main(options, flags)
