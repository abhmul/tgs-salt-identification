import logging
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import label

from pyjet.data import ImageDataset, NpDataset

from kaggleutils import dump_args

# All images should be 101 x 101
ORIG_IMG_SIZE = (101, 101)
IMG_SIZE = (128, 128)


class SaltData(object):
    @dump_args
    def __init__(self, path_to_data="../input", mode="rgb", img_size=IMG_SIZE):
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

    def load_train(self):
        # Just load the data into a numpy dataset, it ain't that big
        logging.info(f"Loading train images from {self.path_to_train_images} "
                     f"and masks from {self.path_to_train_masks}")
        img_paths = sorted(glob(self.glob_train_images))
        mask_paths = set(glob(self.glob_train_masks))  # Use set to look up
        # Initialize the numpy data containers
        x = np.zeros((len(img_paths), ) + self.img_size + (3, ))
        y = np.zeros((len(mask_paths), ) + self.img_size + (1,))
        ids = []
        for i, img_path in enumerate(tqdm(img_paths)):
            x[i] = ImageDataset.load_img(
                img_path, img_size=self.img_size, mode=self.mode)[0]
            # Load the mask
            img_basename = os.path.basename(img_path)
            ids.append(os.path.splitext(img_basename)[0])
            mask_path = os.path.join(self.path_to_train_masks, img_basename)
            # Use the 0 mask if its not there
            if mask_path not in mask_paths:
                logging.info(f"Could not find {img_basename} in masks")
                continue
            y[i] = ImageDataset.load_img(
                mask_path, img_size=self.img_size, mode="gray")[0]
        return NpDataset(x, y, ids=np.array(ids))

    def load_test(self):
        # Just load the data into a numpy dataset, it ain't that big
        logging.info(f"Loading test images from {self.path_to_test_images}")
        img_paths = sorted(glob(self.glob_train_images))
        # Initialize the numpy data containers
        x = np.zeros((len(img_paths), ) + self.img_size + (3, ))
        ids = []
        for i, img_path in enumerate(tqdm(img_paths)):
            x[i] = ImageDataset.load_img(
                img_path, img_size=self.img_size, mode=self.mode)[0]
            # Load the mask
            img_basename = os.path.basename(img_path)
            ids.append(os.path.splitext(img_basename)[0])
        return NpDataset(x, ids=np.array(ids))

    @staticmethod
    def save_submission(save_name, preds, test_ids, cutoff=0.5):
        new_test_ids = []
        rles = []
        for n, id_ in enumerate(tqdm(test_ids)):
            rle = list(prob_to_rles(preds[n], cutoff=cutoff))
            rles += rle
            new_test_ids += ([id_] * len(rle))

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
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)