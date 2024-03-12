import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import misc
import albumentations as A

def initialize_data(config):
    # Create datasets
    train_params = config["train_params"]
    validation_params = config["validation_params"]
    model_params = config["model_params"]

    train_data = SenNetDataset(train_params, model_params)
    validity_data = SenNetDataset(validation_params, model_params)
    
    train_dl = DataLoader(train_data, config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=train_collate)
    validity_dl = DataLoader(validity_data, config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=train_collate)
    return train_dl, validity_dl

def train_collate(batch):
    # Resized 512x512 images
    low_data = torch.stack([item[0] for item in batch])


    last_two_dims = batch[0][1].size()[-2:]
    if tuple(last_two_dims) == torch.Size((512, 512)):
        mask = torch.stack([item[1] for item in batch])
    else:
        mask = [item[1] for item in batch]  # each element is of size (1, h*, w*). where (h*, w*) changes from mask to another.
    
    imgs = [item[2] for item in batch]

    if len(batch[0]) == 5:
        medium_data = torch.stack([item[3] for item in batch])
        high_data = torch.stack([item[4] for item in batch])

        return low_data, mask, imgs, medium_data, high_data

    return low_data, mask, imgs

def test_collate(batch):
    id = [item[0] for item in batch]  # each element is of size (1, h*, w*). where (h*, w*) changes from mask to another.
    small = torch.stack([item[1] for item in batch])
    medium = torch.stack([item[2] for item in batch])
    large = torch.stack([item[3] for item in batch])
    original = [item[4] for item in batch]
    #shapes = [item[2] for item in batch]

    return id, small, medium, large, original#, shapes 

class ShufflePatches(object):
    def __init__(self, patch_size):
        self.ps = patch_size

    def __call__(self, image, mask):
        # Convert the input tensors to float32
        image = image.float()
        mask = mask.float()

        image = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(mask, 0)

        # divide the batch of images into non-overlapping patches
        u_image = torch.nn.functional.unfold(image, kernel_size=self.ps, stride=self.ps, padding=0)
        u_mask = torch.nn.functional.unfold(mask, kernel_size=self.ps, stride=self.ps, padding=0)
        
        # generate random indices for shuffling
        indices = torch.randperm(u_image.shape[-1])

        # permute the patches of each image in the batch
        pu_image = u_image[:, :, indices]
        pu_mask = u_mask[:, :, indices]
        
        # fold the permuted patches back together
        f_image = torch.nn.functional.fold(pu_image, image.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
        f_mask = torch.nn.functional.fold(pu_mask, mask.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
        
        return f_image[0], f_mask[0]


class SenNetDataset(Dataset):
    def __init__(self, dataset_params, model_params):
        '''
        annotations_file: path to the train_rles.csv
        data_dir: path to the train directory
        name: a list containing which kidney subsets NOT to use
        '''
        self.img_labels = pd.read_csv(dataset_params["annotation_path"])
        self.data_dir = dataset_params["data_path"]
        self.name = dataset_params["kidney_names"]
        self.transform = dataset_params["transforms"]
        self.target_transform = dataset_params["target_transforms"]
        self.valid = dataset_params["validation"]
        self.patchsh = dataset_params["patch_shuffling"]

        self.parallel_settings = model_params["parallel_settings"]
        self.interpolation_settings = model_params["interpolation_settings"]

    def __len__(self):
        if len(self.name) > 0:
            for name in self.name:
                self.img_labels = self.img_labels[self.img_labels['id'].str[:-5] != name]
        return len(self.img_labels)
    
    def augment(self, tensor_image, tensor_mask):
        '''
        tensor_image: Kidney image tensor
        tensor_mask: Mask image tensor

        Applies:
        - Color Jitter
        - Random horizontal flip
        - Random vertical flip
        - Random rotate
        - Shot noise
        - Random invert
        - Salt and Pepper Noise
        - Elastic Transform
        '''
        # Color jitter diversifies lighting for a more robust model.
        color_jitter_transform = transforms.ColorJitter((.3, 1.4),(.3, 1.4))
        tensor_image = color_jitter_transform(tensor_image)

        if random.random() > 0.5:
            tensor_image = TF.hflip(tensor_image)
            tensor_mask = TF.hflip(tensor_mask)

        if random.random() > 0.5:
            tensor_image = TF.vflip(tensor_image)
            tensor_mask = TF.vflip(tensor_mask)

        degrees = random.uniform(0, 360)
        tensor_image = TF.rotate(tensor_image, degrees)
        tensor_mask = TF.rotate(tensor_mask, degrees)

        if random.random() > 0.5:
            tensor_image = misc.shot_noise(tensor_image)
        if random.random() > 0.5:
            tensor_image = TF.invert(tensor_image)

        random_float = random.uniform(0, 0.002)
        tensor_image = misc.add_salt_and_pepper_noise(tensor_image, random_float, random_float)

        # tensor_image = misc.histogram_equalization(tensor_image)
        return tensor_image, tensor_mask

    def __getitem__(self, idx):
        '''
        returns augmented images, augmented masks, and unedited tensors to derive shapes
        '''
        if len(self.name) > 0:
            for name in self.name:
                self.img_labels = self.img_labels[self.img_labels['id'].str[:-5] != name]
        img_id = self.img_labels.iloc[idx]
        dir_name = img_id['id'][:-5]
        img_num = img_id['id'][-4:]

        if dir_name == 'kidney_3_dense':
            img_path = os.path.join(self.data_dir, 'kidney_3_sparse/images', img_num+'.tif')
        else:
            img_path = os.path.join(self.data_dir, dir_name, 'images', img_num+'.tif')

        # Gets the path of the saved mask
        mask_path = os.path.join(self.data_dir, dir_name, 'labels', img_num+'.tif')

        mask = cv2.imread(mask_path, -1)# np.array(Image.open(mask_path)) # Must transform, because dataloader doesn't accept PIL data   

        image = cv2.imread(img_path, -1)
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        image = (image - image.min()) / (image.max() - image.min()) # One last normalization check (remove small numbers)
        #image = Image.open(img_path)

        # Data augmentations
        if self.valid == False:
            base = random.randint(100, 360)
            aug = A.ElasticTransform(p=0.5, alpha=base, sigma=base * 0.05, alpha_affine=base * 0.03)
            augmented = aug(image=np.array(image), mask=np.array(mask))
            image = augmented["image"]
            mask = augmented["mask"]

            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

            image, mask = self.augment(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            
        if self.parallel_settings["flag"]:
            medium_resize = transforms.Compose([
                    transforms.Resize((1024, 1024), antialias=True),
                    #transforms.RandomAdjustSharpness(5.0, p=1.0),    
                ])
            medium_image = medium_resize(image)

            large_resize = transforms.Compose([
                    transforms.Resize((2048, 2048), antialias=True),
                    #transforms.RandomAdjustSharpness(5.0, p=1.0),    
                ])
            large_image = large_resize(image)

        original = image.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform and not self.interpolation_settings["flag"]:
            mask = self.target_transform(mask)
        
        if self.patchsh:
            if random.random() > 0.5:
                image, mask = ShufflePatches(64)(image, mask)
        if self.parallel_settings["flag"]:
            return image, mask.half(), original, medium_image, large_image
            
        return image, mask, original
    
    def getrle(self, idx):
        rle = self.img_labels.iloc[idx, 1]
        return rle
    
    def getCSV(self):
        return self.img_labels
    

class TestDataset(Dataset):
    def __init__(self, test_data_dir, transform=None):
        test_kidneys = []
        submission = pd.DataFrame(columns=['id', 'rle'])
        
        # Get the kidney names from test directory
        for dirnames, subdirnames, _ in os.walk(test_data_dir):
            if dirnames == test_data_dir:
                for sub in subdirnames:
                    test_kidneys.append(sub)

        for kidney in test_kidneys:
            for dirnames, _, imgnames in os.walk(os.path.join(test_data_dir, kidney, 'images').replace("\\","/")):
                for img in sorted(imgnames):
                    img_id = kidney + "_" + img[:-4]
                    entry = {'id': img_id,'rle': None}
                    submission = pd.concat([submission, pd.DataFrame([entry])], ignore_index=True)

        self.annotations = submission
        self.data_dir = test_data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_id = self.annotations.iloc[idx]
        dir_name = img_id['id'][:-5]
        img_num = img_id['id'][-4:]

        img_path = os.path.join(self.data_dir, dir_name, 'images', img_num+'.tif')
        #image = Image.open(img_path)

        #shape = image.size

        image = cv2.imread(img_path, -1)
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        image = (image - image.min()) / (image.max() - image.min())

        image = transforms.ToTensor()(image)

        medium_resize = transforms.Compose([
                    transforms.Resize((1024, 1024), antialias=True),
                    #transforms.RandomAdjustSharpness(5.0, p=1.0),    
                ])
        medium = medium_resize(image)

        large_resize = transforms.Compose([
                    transforms.Resize((2048, 2048), antialias=True),
                    #transforms.RandomAdjustSharpness(5.0, p=1.0),    
                ])
        large = large_resize(image)
            

        if self.transform:
            small = self.transform(image)

        return self.annotations.iloc[idx, 0], small, medium, large, image.unsqueeze(0)
    
    def getSubmission(self):
        return self.annotations