# General imports
import os
import datetime
import numpy as np
import random
import json

import yaml
import cv2
from PIL import Image
import argparse
import pathlib as pt
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Importing support packages
import pandas as pd
from skimage import io, transform

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Importing torch vision model
import torchvision
import torchvision.models as models
from torchvision import transforms, utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

pod_data_path = pt.Path(r'D:\Machine Learning\Challenge\SoyBeanPodCount\Data\Dev_Phase\training\pod_annotations')

pod_annotations = pod_data_path / r'pod_detection_annotations.csv'
pod_dataset = pod_data_path / 'dataset'

# Hardcoded variables
display_limit = 5


class SoyPodDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform_d=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.soypod_frame = pd.read_csv(csv_file)
        self.root_dir = pt.Path(root_dir)
        self.transform = transform_d

    def __len__(self):
        return len(self.soypod_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir / self.soypod_frame['filename'][idx]
        image = io.imread(img_name)
        bounds_str = self.soypod_frame['region_shape_attributes'][idx].replace('""', '"')
        label_data = re.sub(r'\W', '', self.soypod_frame['region_attributes'][idx])
        bounds = json.loads(bounds_str)
        sample = {'image': image, 'label': label_data}
        sample.update(bounds)

        if self.transform:
            sample = self.transform(sample)
        return sample


soy_dataset = SoyPodDataset(csv_file=pod_annotations, root_dir=pod_dataset, transform_d=None)

fig = plt.figure()

for i in range(len(soy_dataset)):
    sample = soy_dataset[i]

    rect = patches.Rectangle((sample['x'], sample['y']), sample['width'], sample['height'], linewidth=1, edgecolor='r',
                             facecolor='none')
    ax = plt.subplot(1, display_limit, i + 1)
    plt.tight_layout()
    ax.set_title(f"Sample #{i + 1}\n Label: {sample['label']}")
    ax.axis('off')
    ax.imshow(sample['image'])
    ax.add_patch(rect)

    if i == display_limit-1:
        plt.show()
        break

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

dataloader = DataLoader(soy_dataset, batch_size=4,
                        shuffle=True, num_workers=0)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, soy_batch = \
            sample_batched['image'], sample_batched['label']
    x, y, w, h = sample_batched['x'],sample_batched['y'], sample_batched['width'], sample_batched['height']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid)

    # for i in range(batch_size):
    #     rect = patches.Rectangle((sample['x'], sample['y']), sample['width'], sample['height'], linewidth=1,
    #                              edgecolor='r',
    #                              facecolor='none')

    plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch, sample_batched['image'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
