import random
import numpy as np
from csv import writer
import pandas as pd
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
import cv2
from skimage import exposure
import yaml

def initialize_config(root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize train and test paths; searches for the directory with the data and assigns the paths to our TRAIN_PATH and TEST_PATH variables
    for dirnames, subdirnames, filenames in os.walk(root):
        if 'test' and 'train' in subdirnames:
            TRAIN_PATH = os.path.join(dirnames, 'train')
            TEST_PATH = os.path.join(dirnames, 'test')
            ANNO_PATH= os.path.join(dirnames, 'train_rles.csv')

    # Sets up the config dictionary
    config_data = {
        "device": device,
        "train_params": {
            "annotation_path": ANNO_PATH,
            "data_path": TRAIN_PATH,
            "kidney_names": ['kidney_3_dense', 'kidney_1_voi', 'kidney_3_sparse', 'kidney_2'],
            "transforms": transforms.Compose([
                transforms.RandomAdjustSharpness(2.0, p=0.5),
                transforms.Resize((512, 512), antialias=True),
                #transforms.RandomAdjustSharpness(5.0, p=1.0),
                #AdaptiveHistogramEqualization(clip_limit=2.0, tile_grid_size=(8, 8)),
            ]),
            "target_transforms": transforms.Compose([
                transforms.Resize((512, 512), antialias=True)
            ]),
            "patch_shuffling": False,
            "validation": False,
        },
        "validation_params":{
            "annotation_path": ANNO_PATH,
            "data_path": TRAIN_PATH,
            "kidney_names": ['kidney_1_dense', 'kidney_1_voi', 'kidney_2', 'kidney_3_sparse'],
            "transforms": transforms.Compose([
                transforms.Resize((512, 512), antialias=True),
                #transforms.RandomAdjustSharpness(5.0, p=1.0),
            ]),
            "target_transforms": transforms.Compose([
                transforms.Resize((512, 512), antialias=True)
            ]),
            "patch_shuffling": False,
            "validation": True,

        },
        "batch_size": 4,
        "num_workers": 0,
        "test_params":{
            "test_data_path": TEST_PATH,
            "test_transforms": transforms.Compose([
                transforms.Resize((512, 512), antialias=True),
                #transforms.RandomAdjustSharpness(5.0, p=1.0),  
            ]),
        },
        "experiment_name": "unspecified",
        "writer": True,
        "threshold": .01,
        "modelname": 'ResTransUNet',
        "weights": None,
        "learning_rate": .01,
        "momentum": 0.9,
        "decay": 1e-05,
        "epochs": 30,
        "model_params": {
            "parallel_settings":{
                "flag": True,
                "branch_out_channels": 32,
                "concatenate": False,
                "trunk_blocks": 0,
                "branch_blocks": 1,
            },
            "resnet_settings":{
                "width": 64,
                "blocks": [3, 4, 6],
                "normalization": "group",
            },
            "transformer_params":{
                "num_layers": 12,
                "num_heads": 12,
                "hidden_dim": 768,
                "mlp_dim": 2048,
            },
            "interpolation_settings":{
                "flag": True,
                "topper": True,
                "mode": "concatenation",
                "leakyReLU": True,
                "root_out": 32,
                "blocks": 3,
                "groups": 8,
                "normalization":'group',
            },
            "trunk/res_channels": 32,
            "leakyReLU": True,
            "instanceNorm": False,
            "up_mode": 0,
        },
    }

    return config_data
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Adds salt-and-pepper noise to a float32 image.

    Parameters:
    - image (torch.Tensor): Input float32 image tensor.
    - salt_prob (float): Probability of adding salt noise.
    - pepper_prob (float): Probability of adding pepper noise.

    Returns:
    - torch.Tensor: Image tensor with salt-and-pepper noise.
    """

    # Ensure the input image is a float32 tensor
    assert image.dtype == torch.float32, "Input image must be of dtype float32"

    # Generate random noise mask
    noise_mask = torch.rand_like(image)

    # Add salt noise
    image = torch.where(noise_mask < salt_prob, 1.0, image)

    # Add pepper noise
    image = torch.where(noise_mask > (1.0 - pepper_prob), 0.0, image)

    return image

class AdaptiveHistogramEqualization:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert the PyTorch tensor to a NumPy array
        img_np = img.numpy() #.astype(np.int16).squeeze()

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np_clahe = clahe.apply(img_np)

        # Convert back to a PyTorch tensor
        img_clahe = torch.from_numpy(img_np_clahe).to(img.dtype) #.unsqueeze(0)

        return img_clahe

'''def histogram_equalization(image_tensor):
    # Ensure the image tensor is of float type
    image_tensor_float = image_tensor.float()

    # Normalize to the range [0, 1]
    normalized_image = (image_tensor_float - image_tensor_float.min()) / (image_tensor_float.max() - image_tensor_float.min())

    # Flatten the image tensor to a 1D tensor
    flat_image = normalized_image.view(-1)

    # Compute the cumulative distribution function (CDF)
    cdf = torch.cumsum(torch.histc(flat_image, bins=65536), dim=0)

    # Apply histogram equalization
    equalized_flat_image = cdf[torch.floor(flat_image * 65535).long()] / cdf[-1]

    # Reshape the flat image back to the original shape
    equalized_image = equalized_flat_image.view(normalized_image.shape)

    # Convert the equalized tensor back to original data type
    equalized_image_int16 = (equalized_image * 65535).to(image_tensor.dtype)

    return equalized_image_int16'''

def histogram_equalization(image_tensor):
    """
    Apply histogram equalization to a float32 image tensor.

    Args:
        image_tensor (torch.Tensor): Input image tensor in the range [0, 1].

    Returns:
        torch.Tensor: Equalized image tensor.
    """
    # Convert PyTorch tensor to NumPy array
    image_np = image_tensor.numpy()

    # Apply histogram equalization
    equalized_image_np = exposure.equalize_hist(image_np[0])

    # Convert the result back to a PyTorch tensor
    equalized_image_tensor = torch.from_numpy(equalized_image_np).float()

    return equalized_image_tensor.unsqueeze(0)

def shot_noise(image, intensity=(60, 120)):
    intensity = random.randint(intensity[0], intensity[1])
    # Generate shot noise using Poisson distribution
    noisy_image_tensor = torch.poisson(image * intensity) / intensity

    # Clip values to ensure they are within the valid range [0, 1]
    noisy_image_tensor = torch.clamp(noisy_image_tensor, 0, 1)
    return noisy_image_tensor


def initialize_samples(train_set, validation_set, num_samples=5):
    training_samples = []
    validation_samples = []
    train_len = train_set.__len__()
    valid_len = validation_set.__len__()

    train_indices = np.random.choice(range(train_len), size=num_samples, replace=False) 
    valid_indices = np.random.choice(range(valid_len), size=num_samples, replace=False) 


    for i in train_indices:
        training_samples.append(train_set.__getitem__(i))

    for i in valid_indices:
        validation_samples.append(validation_set.__getitem__(i))

    return training_samples, validation_samples

class DiceLoss(torch.nn.Module):
    def __init__(self, device, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.device = device
        
    def forward(self, predicted, target):
        '''
        predicted: List of Tensors generated by the model
        target: List of Tensor masks OR Tensor Masks (B, 1, 512, 512)
        Needs to receive tensors in list format due to varying image sizes.
        '''

        loss = 0
        
        # Get the last two dimensions of the tensor
        last_two_dims = target.size()[-2:]

        # We pass to dice loss
        if tuple(last_two_dims) == torch.Size((512, 512)):
            # Calculate loss in resized masks
            intersection = torch.sum(predicted * target)
            union = torch.sum(predicted) + torch.sum(target)

            dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss = 1.0 - dice_coefficient
        else:
            # We need to reshape the images to their original shapes and calculate loss
            for i in range(len(predicted)):
                shape = list(target[i].shape)
                input = predicted[i]
                input = torch.unsqueeze(input, 0)
                input = F.interpolate(input, size=shape[1:], mode='bilinear', align_corners=False)
                input = torch.where(input > .5, input-input+1, input - input)
                    
                intersection = torch.sum(input * target[i].to(self.device))
                union = torch.sum(input) + torch.sum(target[i].to(self.device))

                dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)
                loss = (1.0 - dice_coefficient)/len(predicted) + loss

        return loss

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle=='':
        rle = '1 0'
    return rle

def rle_decode(rle, shape):
    length = shape[0] * shape[1]
    msk = np.zeros((length,))
    rle = rle.split()
    for i in range(0, len(rle), 2):
        count = int(rle[i + 1])
        for j in range(count):
            msk[int(rle[i]) - 1 + j] = 255

    msk = msk.reshape(shape)

    return np.uint8(msk)

# Initializes the .csv file
def initializeCSV(model_name, col_names=['Epoch', 'Iteration', 'Avg. Train Loss', 'Validation Loss', 'Iteration Train Loss']):
    csv = pd.DataFrame(columns=col_names)
    csv.to_csv('./' + model_name + '.csv', index=False)

def writeToCSV(model_name, data):
    with open('./' + model_name + '.csv', 'a') as file:
        writer_object = writer(file, lineterminator = '\n')
        writer_object.writerow(data)
        file.close()

def loadYAML(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content

# https://www.kaggle.com/code/hengck23/lb0-808-resnet50-2d-unet-xy-zy-zx-cc3d
def remove_small_objects(mask, min_size, threshold):
    # mask = ((mask > threshold))
    # find all connected components (labels)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # create a mask where small objects are removed
    processed = np.zeros_like(mask)
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= min_size:
            processed[label == l] = 1
    return processed