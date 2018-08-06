import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize

from pyjet.data import ImageDataset, NpDataset

from kaggleutils import dump_args

# All images should be 101 x 101
ORIG_IMG_SIZE = (101, 101)
IMG_SIZE = (128, 128)
EMPTY_THRESHOLD = 5
USE_DEPTHS = False


class SaltData(object):
    @dump_args
    def __init__(self, path_to_data="../input", mode="rgb",
                 img_size=ORIG_IMG_SIZE):
        self.path_to_data = path_to_data
        self.mode = mode
        self.img_size = img_size

        self.path_to_train_images = os.path.join(path_to_data, "train",
                                                 "images")
        self.path_to_test_images = os.path.join(path_to_data, "test", "images")
        self.path_to_train_masks = os.path.join(path_to_data, "train", "masks")
        self.path_to_depths = os.path.join(path_to_data, "depths.csv")
        self.path_to_train_rle = os.path.join(path_to_data, "train.csv")

        self.glob_train_images = os.path.join(self.path_to_train_images,
                                              "*.png")
        self.glob_train_masks = os.path.join(self.path_to_train_masks, "*.png")
        self.glob_test_images = os.path.join(self.path_to_test_images, "*.png")

        self.__depths = None

    @property
    def depths(self):
        if self.__depths is not None:
            return self.__depths
        logging.info("Loading depths from {self.path_to_depths} "
                     "".format(**locals()))
        unnorm_depths = pd.read_csv(self.path_to_depths, index_col=0)
        self.__depths = (unnorm_depths - unnorm_depths.mean()) / \
            unnorm_depths.std()
        return self.__depths

    def load_train(self):
        # Just load the data into a numpy dataset, it ain't that big
        logging.info("Loading train images from {self.path_to_train_images} "
                     "and masks from {self.path_to_train_masks}"
                     "".format(**locals()))
        img_paths = sorted(glob(self.glob_train_images))
        mask_paths = set(glob(self.glob_train_masks))  # Use set to look up
        # Initialize the numpy data containers
        x = np.zeros((len(img_paths), ) + self.img_size + (4, ))
        y = np.zeros((len(mask_paths), ) + self.img_size + (1,))
        ids = []
        for i, img_path in enumerate(tqdm(img_paths)):
            img_basename = os.path.basename(img_path)
            ids.append(os.path.splitext(img_basename)[0])

            x[i, ..., :3] = ImageDataset.load_img(
                img_path, img_size=None, mode=self.mode)[0]
            x[i, ..., 3] = self.depths.loc[ids[-1]]
            # Load the mask
            mask_path = os.path.join(self.path_to_train_masks, img_basename)
            # Use the 0 mask if its not there
            if mask_path not in mask_paths:
                logging.info("Could not find {img_basename} in masks"
                             "".format(**locals()))
                continue
            y[i] = ImageDataset.load_img(
                mask_path, img_size=None, mode="gray")[0]
        print("X shape:", x.shape)
        print("Y Shape:", y.shape)
        return NpDataset(x.astype('float32'), y.astype('float32'), ids=np.array(ids))

    def load_test(self):
        # Just load the data into a numpy dataset, it ain't that big
        logging.info("Loading test images from {self.path_to_test_images}"
                     " and glob {self.glob_test_images}".format(**locals()))
        img_paths = sorted(glob(self.glob_test_images))
        # Initialize the numpy data containers
        x = np.zeros((len(img_paths), ) + self.img_size + (4, ))
        ids = []
        for i, img_path in enumerate(tqdm(img_paths)):
            x[i, ..., :3] = ImageDataset.load_img(
                img_path, img_size=None, mode=self.mode)[0]
            x[i, ..., :4] = self.depths.loc[ids[-1]] / MAX_DEPTH
            # Load the mask
            img_basename = os.path.basename(img_path)
            ids.append(os.path.splitext(img_basename)[0])
        print("Xte Shape:", x.shape)
        return NpDataset(x.astype('float32'), ids=np.array(ids))

    @staticmethod
    def get_stratification_categories(train_dataset, num_categories=10):
        bounds = np.linspace(0.0, 1.0, num=num_categories)
        coverage = np.sum(train_dataset.y, axis=(1, 2, 3))
        categories = np.zeros(len(train_dataset))
        for i in range(1, len(bounds)):
            categories[(coverage > bounds[i-1]) & (coverage <= bounds[i])] = i
        return categories

    @staticmethod
    def save_submission(save_name, preds, test_ids, cutoff=0.5):
        new_test_ids = []
        rles = []
        # Figure out if we have to resize
        resize_imgs = preds.shape[1:3] != ORIG_IMG_SIZE
        logging.info("Resize the images: {}".format(resize_imgs))
        for n, id_ in enumerate(tqdm(test_ids)):
            pred = preds[n]
            if resize_imgs:
                pred = resize(preds[n], ORIG_IMG_SIZE,
                              mode='constant', preserve_range=True)
            pred = pred >= cutoff
            if np.count_nonzero(pred) < EMPTY_THRESHOLD:
                rle = []
            else:
                rle = rle_encoding(pred)
            rles.append(rle)
            new_test_ids.append(id_)

        # Create submission DataFrame
        sub = pd.DataFrame()
        sub['id'] = new_test_ids
        sub['rle_mask'] = pd.Series(rles).apply(
            lambda x: ' '.join(str(y) for y in x))
        sub.to_csv(save_name, index=False)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        # Figure out if we need to start a new run length
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
