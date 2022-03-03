"""

Mask R-CNN
Configurations and data loading code for prostate
Written by mo

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights.
    python3 prostate.py train --dataset=/path/to/mri/data --model=imagenet 
    e.g. python prostate.py train --dataset=/Users/mo/medphys/prostate/mri_data/mrcnn_data/images/ --weights=imagenet --logs=logs
    windows python prostate.py train --dataset=C:\CLUSTERDATA\BIOMARKERS\RP\mrcnn_data\images --weights=imagenet --logs=logs
    # Continue training a model that you had trained earlier
    python3 prostate.py train --dataset=/path/to/mri/data --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 prostate.py train --dataset=/path/to/mri/data --model=last

    # Run COCO evaluation on the last model you trained
    python3 prostate.py evaluate --dataset=/path/to/mri/data --model=last
"""

import os
import sys
import json
import datetime
import numpy as np
import imgaug
from glob import glob
import skimage.io
import logging

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), "logs")

############################################################
#  Configurations
############################################################


class ProstateConfig(Config):
    """Configuration for training on the Prostate MRI dataset.
    Derives from the base Config class and overrides some value
    """
    # Give the configuration a recognizable name
    NAME = "prostate"

    # Uncomment to train on 3 GPUs on cluster
    GPU_COUNT = 3

    # We use 3 GPUs with 48GB memory each
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 125

    # Backbone network architecture
    BACKBONE = "resnet101"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + prostate + pz + roi

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64 # default 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 200
    POST_NMS_ROIS_INFERENCE = 100

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "none"

    # Our images have 4 channels, [T2, DCE, ADC, BVAL]
    IMAGE_CHANNEL_COUNT = 4

    # The mean pixel of our images
    MEAN_PIXEL = np.array([97.75, 84.46, 88.44, 89.76])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.15

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    # MASK_SHAPE = [56, 56]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 8

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 8

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 3.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

class ProstateInferenceConfig(ProstateConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Dataset
############################################################

class ProstateDataset(utils.Dataset):

    def load_prostate(self, dataset_dir, subset):
        """Load a subset of the prostate mri dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("mri", 1, "prostate")
        self.add_class("mri", 2, "pz")
        self.add_class("mri", 3, "roi")
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_paths = glob(os.path.join(dataset_dir,"*","*"))

        for image in image_paths:
            self.add_image(
                "mri",
                image_id='.'.join(os.path.basename(image).split('.')[1:-1]),  # use case.slice as unique id
                path=image)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        image_uid = info['id']
        image_path = info['path']
        mask_dir = os.path.join(os.path.dirname(image_path),'..','..','..','masks',image_uid)
        # [height, width, instance_count]
        masks = glob(os.path.join(mask_dir,'*.png'))

        # if image has no masks, use parent method without warning..
        if not masks:
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
            return mask, class_ids

        mask_list = []
        labels = []

        pr = list(filter(lambda x: 'pr.' in x, masks))
        if pr:
            m = skimage.io.imread(pr[0]).astype(bool)
            mask_list.append(m)
            labels.append(1)

        pz = list(filter(lambda x: 'pz.' in x, masks))
        if pz:
            m = skimage.io.imread(pz[0]).astype(bool)
            mask_list.append(m)
            labels.append(2)

        rois = list(filter(lambda x: 'roi.' in x, masks))
        # this loop should not do anything if rois is empty
        for roi in rois:
            m = skimage.io.imread(roi).astype(bool)
            mask_list.append(m)
            labels.append(3)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return  np.stack(mask_list, axis=-1), np.array(labels, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array. 
           Overload since our pictures have 4 channels
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        return image


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ProstateDataset()
    dataset_train.load_prostate(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ProstateDataset()
    dataset_val.load_prostate(args.dataset, "val")
    dataset_val.prepare()


    # Image Augmentation
    # Right/Left flip 50% of the time
    # TODO: ADD MORE
    augmentation = imgaug.augmenters.SomeOf((0, 2), [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.Flipud(0.5),
        imgaug.augmenters.OneOf([imgaug.augmenters.Affine(rotate=90),
                      imgaug.augmenters.Affine(rotate=180),
                      imgaug.augmenters.Affine(rotate=270)]),
        imgaug.augmenters.Multiply((0.8, 1.5))
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Training - Stage 1
    print("Training backbone resenet only")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='backbone',
                augmentation=augmentation)

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 3
    # Finetune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=60,
                layers='all',
                augmentation=augmentation)
############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'eval'")
    parser.add_argument('--dataset', required=False, default='/Users/mo/medphys/prostate/mri_data/mrcnn_data/images/',
                        metavar="/path/to/mri/dataset/",
                        help='Directory of the MRI dataset')
    parser.add_argument('--weights', required=False, default='imagenet',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ProstateConfig()
    else:
        class InferenceConfig(ProstateConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "imagenet":
        # exclude layer that assume 3 channel input
        model.load_weights(weights_path, by_name=True, exclude=["conv1"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "eval":
        print("eval not implement yet")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'eval'".format(args.command))