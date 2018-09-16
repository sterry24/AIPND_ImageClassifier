import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        PARAMETERS:
                     image: PIL image object

        RETURNS:
                     np_arr: a numpy array representation of the image
    '''

    # Assume image is alread a PIL image and not a filepath
    w,h = image.size
    ## Set the image dimensions
    new_w = 256 * (w/h) if w > h else 256
    new_h = 256 * (w/h) if h > w else 256
    image.thumbnail((new_w,new_h))
    ## crop the image
    l = (new_w - 224)/2
    r = (new_w + 224)/2
    t = (new_h - 224)/2
    b = (new_h + 224)/2
    crop_img = image.crop((l,t,r,b))
    ## set as np array
    np_img = np.array(crop_img)
    ## normalize
    np_img = np_img / 255.0
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    np_img = (np_img - mean) / std
    ## change order for PyTorch (expects color channel first)
    np_img = np.transpose(np_img,(2,0,1))

    return np_img

def choose_random_test_image(image_dir):

    """Returns a random image from the provided path. Directory is
       expected to be in the same structure required by PyTorch
       ImageFolder, and for this case to be 3 levels deep.
       ex: ./flowers/test/76/image_02472.jpg could be found by
           calling with image_dir = ./flowers/test

       PARAMETERS:
                   image_dir: string; the top level path containing the image

       RETURNS:
                    filepath: string; the path to the random image

    """

    tval = np.random.choice(os.listdir(image_dir))
    img_file = np.random.choice(os.listdir(os.path.join(image_dir,tval)))
    filepath = os.path.join('./',image_dir,tval,img_file)
    print(filepath)
    return filepath

def imshow(image, ax=None, title=None):

    """This function converts a PyTorch tensor and displays it.
       Used as a check for process_image, to compare with original
       image.

       PARAMETERS:
                   image: a pytorch tensor containing the image data
                   ax:    the figure axis to apply the image to
                   title: title to give the plot.

       RETURNS:
                    ax: the figure axis containing the image
    """

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def create_transforms(resize,crop,mean_vals,std_vals,train=False):

    """create the transforms to used on the input dataset

       PARAMETERS:
                     resize:    int; value to resize image to
                     crop:      int; value to crop image to
                     mean_vals: list; list of mean values for normalization
                     std_vals:  list; list of std values for normalization
                     train:     bool; default False, whether to apply randomness
                                      to transform

       RETURNS:
                     transform: torchvision transform to be applied to dataset
    """

    if train:
        transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(crop),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_vals, std_vals),
                                       ])
    else:
        transform = transforms.Compose([transforms.Resize(resize),
                                        transforms.CenterCrop(crop),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_vals, std_vals)
                                       ])
    return transform

def get_loaders(dirpath,transforms,batch_size,shuffle=False):

    """load datasets with ImageFolder.  Check torchvision documentation
       for ImageFolder requirements for inputs.

       PARAMETERS:
                    dirpath:    string; the path to the data directory
                    transforms: torchvision transform to be applied to data

       RETURNS:
                     dset:    ImageFolder of dataset
                     loader:  DataLoader with specified dataset and transforms

    """

    dset = datasets.ImageFolder(dirpath,transform=transforms)
    loader = torch.utils.data.DataLoader(dset,batch_size=batch_size,shuffle=shuffle)

    return dset,loader
