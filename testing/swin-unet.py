from IPython.display import clear_output

import numpy as np
import glob
import os
import tqdm
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import random
import math
from skimage import metrics

random.seed(42)

from __future__ import print_function
import argparse
import os
import time
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from vunet import unet, unet_512
#from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
#from dilated_residual_unet import dilated_residual_UNet
#from LoadData import Dataset, loader, Dataset_val
from SwinUnet import SwinTransformerSys
from LoadImages import load_images, load_image_single, extract_batch, extract_batch_single, ScanFile
from losses import bce_loss, dice_loss, weighted_bce_loss, focal_loss, dice_loss_1, tversky_loss, focal_tversky_loss, combined_bce_dice_loss


import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torchvision
import torchvision.transforms as transforms
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter 
from torch.optim.lr_scheduler import ReduceLROnPlateau

import monai
from monai.networks.layers import Norm
from monai.data import create_test_image_2d, list_data_collate, decollate_batch
from monai.visualize import plot_2d_or_3d_image

from monai.utils import first, set_determinism
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    EnsureChannelFirstd,
    EnsureType,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RemoveSmallObjects,
    ScaleIntensityd,
    EnsureTyped,
)


# from monai.transforms.utility.array import AddChannel

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
set_determinism(seed=42)

# Set directories.

train_path_raw = r'\Users\Y2LabServer4\Downloads\SegSwinUNet\transformed_data\normalized_256\training\*_c1.nii.gz'
train_path_seg = r'\Users\Y2LabServer4\Downloads\SegSwinUNet\transformed_data\normalized_256\training\*_sg.nii.gz'
val_path_raw = r'\Users\Y2LabServer4\Downloads\SegSwinUNet\transformed_data\normalized_256\validation\*_c1.nii.gz'
val_path_seg = r'\Users\Y2LabServer4\Downloads\SegSwinUNet\transformed_data\normalized_256\validation\*_sg.nii.gz'
test_path_raw = r'\Users\Y2LabServer4\Downloads\SegSwinUNet\transformed_data\normalized_256\testing\*_c1.nii.gz'
test_path_seg = r'\Users\Y2LabServer4\Downloads\SegSwinUNet\transformed_data\normalized_256\testing\*_sg.nii.gz'

train_images = sorted(glob.glob(train_path_raw))
train_labels = sorted(glob.glob(train_path_seg))
train_files = [
    {'image': image_name, 'label': label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

val_images = glob.glob(val_path_raw)
val_labels = glob.glob(val_path_seg)
val_files = [
    {'image': image_name, 'label': label_name}
    for image_name, label_name in zip(val_images, val_labels)
]

test_images = glob.glob(test_path_raw)
test_labels = glob.glob(test_path_seg)
test_files = [
    {'image': image_name, 'label': label_name}
    for image_name, label_name in zip(test_images, test_labels)
]

print('{} train_files, {} val_files, {} test_files'.format(len(train_files), len(val_files), len(test_files)))
# assert((len(train_files) + len(test_files) + len(val_files)) == len(images))

daten = input('Input date, which will be used to save relevant plots/images for this run: (yyyymmdd):')
best_model_name = daten +'_best_metric_model.pth'
print(best_model_name)

# Set transforms for training, validation, and saving validation outputs (train_transform2).
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]), # LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            # spatial_size= (96,96,1), #(160, 160),
            spatial_size= (256, 256, 1),
            pos=3,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
          RandAffined(
             keys=['image', 'label'],
             mode=('bilinear', 'nearest'),
             prob=1.0,# spatial_size=(96,96,1),
             spatial_size= (256, 256, 1),
             rotate_range=(0, 0, np.pi/15),
             scale_range=(0.1, 0.1)),
        EnsureTyped(keys=["image", "label"], data_type='tensor'),
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]), # LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[-1,-1,1]),
        EnsureTyped(keys=["image", "label"], data_type='tensor'),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]), # LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
        CenterSpatialCropd(keys=["image", "label"], roi_size=[-1,-1,1]),
        EnsureTyped(keys=["image", "label"], data_type='tensor'),
    ]
)

# See example of image, label.
check_ds = Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"], check_data["label"])
label = label.int()
print(f"image shape: {image.shape}, label shape: {label.shape}")
print(f'label values: {np.unique(label)}')
print(f'image max: {image.max()}, image min: {image.min()}')
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[0,0,...], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0,0,...])
plt.show()

print(f"Label values: {label}")

plt.savefig(f"{daten}data{counter}.png")

print(image.shape)
# Create Datasets and DataLoaders.

# Dataset and DataLoader for training.
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
    pin_memory=torch.cuda.is_available(),
)

# print(torch.cuda.is_available())

# Dataset and DataLoader for validation.
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

# Dataset and DataLoader for testing.
test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)


# Metrics.
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Create UNet, DiceLoss and Adam optimizer.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

num_channels = 1
num_classes = 1

model = SwinTransformerSys(img_size=256,
                           patch_size=4,
                           in_chans=num_channels,
                           num_classes=num_classes,
                           embed_dim=96,
                           depths=[2, 2, 2, 2],
                           num_heads=[3, 6, 12, 24],
                           window_size=8,
                           mlp_ratio=4,
                           qkv_bias=True,
                           qk_scale=None,
                           drop_rate=0.1,
                           attn_drop_rate=0.1,
                           drop_path_rate=0.1,
                           norm_layer=nn.LayerNorm,
                           ape=False,
                           patch_norm=True,
                           use_checkpoint=False).to(device)

loss_function = monai.losses.DiceLoss(sigmoid=True)

optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# Typical PyTorch training loop.

epochs = 200
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])  

for epoch in range(epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        inputs = inputs[...,0]
        labels = labels[...,0]
        optimizer.zero_grad()
        
        # inputs = inputs.resize((256, 256))
        # inputs = inputs.to(device)
        
        outputs = model(inputs)
        
        # print(f"Outputs: {outputs}")
        
        loss = loss_function(outputs, labels)
        loss.backward()
        # loss = loss_function(outputs, labels, opt.tversky_alpha, opt.tversky_beta, opt.focal_tversky_gamma)
        # loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_images = val_images[...,0]
                val_labels = val_labels[...,0]
                # roi_size = (96, 96)
                roi_size = (256, 256)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # Compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            # Aggregate the final mean dice resul
            metric = dice_metric.aggregate().item()
            print(metric)
            # Reset the status for next validation round
            dice_metric.reset()
            
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_name)
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# Plot loss and mean Dice.
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
# plt.savefig(daten + best_model_name[:-4] + '_diceloss.jpg')
plt.show()

# Write epoch loss values and dice values to file.
export_prefix = daten + '_swin-unet'
export_diceloss = export_prefix + '_diceloss.txt'
export_dicemetric = export_prefix + '_dicemetric.txt'

with open(export_diceloss, 'a') as myfile1:
    myfile1.write('\n'.join(str(item) for item in epoch_loss_values)+'\n')
with open(export_dicemetric, 'a') as myfile2:
    myfile2.write('\n'.join(str(item) for item in metric_values)+'\n')

# testing
#from IPython.display import clear_output
# root_dir = r'\Users\Y2LabServer4\downloads\ipynb_checkpoints\ipynb_checkpoints\data'

best_model_name = r'C:\Users\Y2LabServer4\Downloads\SegSwinUNet\20231115_best_metric_model.pth'
checkpoint_tmp = torch.load(best_model_name)
post_trans = Compose([EnsureType(data_type='tensor'), AsDiscrete(threshold=1.4)]) 
# model.load_state_dict(torch.load(best_model_name))
model.load_state_dict(checkpoint_tmp)
test_dice=0
test_hausdorff=0

model.eval()
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
        test_images = test_images[...,0]
        test_labels = test_labels[...,0]
        
        # define sliding window size and batch size for windows inference
        # roi_size = (96, 96)
        roi_size = (256, 256)
        sw_batch_size = 4

        test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
        test_outputs = post_trans(test_outputs)
        test_labels = decollate_batch(test_labels)
        
        # compute metric for current iteration
        dice_metric(y_pred=test_outputs[0], y=test_labels[0])       
        
        # visualize outputs
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image")
        plt.imshow(test_data["image"][0, 0, :, :], cmap="gray")
        #print('val_data-image:' ,np.unique(val_data["image"][0, 0, :, :]))
        plt.subplot(1, 3, 2)
        plt.title(f"label")
        plt.imshow(test_data["label"][0, 0, :, :], cmap='gray')
        print('test_data-label:' ,np.unique(test_data["label"][0, 0, :, :]))
        plt.subplot(1, 3, 3)
        plt.title(f"output")
        plt.imshow(test_outputs[0].detach().cpu()[0, :, :], cmap='gray')
        print('test_outputs shape:', test_outputs[0].shape, test_outputs[0].dtype)
        #print('test_outputs:', np.unique(test_outputs[0].detach().cpu()))
        plt.show()
        #clear_output(wait=True)
        
        if False:
            for test_output in test_outputs:
                saver(test_output)
                
                
    # aggregate the final mean dice result
    test_dice += dice_metric.aggregate().item()
    
        
    
    # reset the status
    dice_metric.reset()
    
    print("evaluation metric:", test_dice)
