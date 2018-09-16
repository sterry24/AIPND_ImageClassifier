import argparse
from collections import OrderedDict
import json
import numpy as np
import os
from PIL import Image
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import data_utils as dutils
import model_utils as mutils

def main(in_args):

    ## Set some initial values
    resize = 256
    crop = 224
    mean_vals = [0.485,0.456,0.406]
    std_vals = [0.229,0.224,0.225]
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    output_size = len(cat_to_name)
    ## Load the data
    data_dir = in_args.data_dir
    if not os.path.exists(data_dir):
        sys.stdout.write("ERROR: Data Directory %s does not exist!  Exiting" %data_dir)
        sys.exit()
    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')
    test_dir = os.path.join(data_dir,'test')

    ## Define transforms
    train_transforms = dutils.create_transforms(resize,crop,mean_vals,std_vals,train=True)
    test_transforms = dutils.create_transforms(resize,crop,mean_vals,std_vals)
    ## can use test_transforms for valid data

    ## Load datasets
    train_datasets,train_loaders = dutils.get_loaders(train_dir,train_transforms,64,shuffle=True)
    valid_datasets,valid_loaders = dutils.get_loaders(valid_dir,test_transforms,32)
    test_datasets,test_loaders = dutils.get_loaders(test_dir,test_transforms,32)

    ## Load a pretrained model
    arch = in_args.arch
    model = mutils.get_model(arch,pretrain=True)

    ## Build the classifier
    model.classifier = mutils.create_classifier(model,output_size,
                                    drop=0.2, hidden_units=in_args.hidden_units)
    ## criterion and optimizer
    crit = nn.NLLLoss()  ## Reqs LogSoftmax
    optimizer = optim.Adam(model.classifier.parameters(),lr=in_args.learning_rate)

    ## train the model
    model = train(in_args,model,crit,optimizer,train_loaders,valid_loaders)

    ## check for save
    if in_args.save_dir is not None:
        model.class_to_idx = train_datasets.class_to_idx
        doCheckpoint(in_args,model,output_size)

    sys.stdout.write("Testing Model\n")
    valid_test(model,test_loaders, in_args)

def doCheckpoint(in_args, model, output_size):

    """create a checkpoint and save_dir

       PARAMETERS:
                    in_args: argparse.Parser object
                    model:   PyTorch model
                    output_size: int

       RETURNS:
                    None
    """

    checkpoint = {'epochs':in_args.epochs,
                  'input_size': model.classifier[0].in_features,
                  'output_size': output_size,
                  'hidden_size': in_args.hidden_units,
                  'arch':in_args.arch,
                  'class_to_idx': model.class_to_idx,
                 }

    savefile = os.path.join(in_args.save_dir,'checkpoint.pth')
    mutils.save_checkpoint(checkpoint,model,filename=savefile)

    sys.stdout.write("Checkpoint Saved to %s\n" %savefile)

def valid_test(model,dataset,in_args):

    """ check the model on the validation dataset

       PARAMETERS:
                   model: PyTorch model to be tested
                   dataset: the validation dataset (PyTorch DataLoader)
                   in_args: argparse.Parser object

       RETURNS:
                  NONE
    """

    if in_args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            images,labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    sys.stdout.write("Accuracy: %.2f" %(100 * correct / total))

def train(inp_args,model,criterion,optimizer,train_loaders,valid_loaders):

    """train the model on the dataset """

    if inp_args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model.to(device)

    epochs = inp_args.epochs

    print_check = 40
    steps = 0

    model.train()
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loaders):
            steps += 1
            inputs, labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_check == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = mutils.validation(model, valid_loaders, criterion, device)

                sys.stdout.write("Epoch: %d of %d\n" %(e+1,epochs))
                sys.stdout.write("\tTraining Loss:       %.3f\n" %(running_loss/print_check))
                sys.stdout.write("\tValidation Loss:     %.3f\n" %(test_loss/len(valid_loaders)))
                sys.stdout.write("\tValidation Accuracy: %.3f\n" %(accuracy/len(valid_loaders)))

                running_loss = 0
                model.train()

    return model


def regurgitate(in_args):

    """ print the arguments, specified by user or program defaults"""

    print("\nBeginning training of network with following inputs:\n")
    for arg in vars(in_args):
        if getattr(in_args,arg) is not None:
            print("%-14s:\t" %arg,getattr(in_args,arg))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a new network')
    parser.add_argument('data_dir',type=str,help="The path to the data folder",default="./flowers")
    parser.add_argument('--save_dir',type=str,help="The path to save the model",default="/home")
    parser.add_argument('--arch',type=str,help="The newtwork architecture to download",default='vgg16')
    parser.add_argument('--learning_rate',type=float,help="The learning rate to use",default=0.001)
    parser.add_argument('--hidden_units',nargs='+',type=int,help="The hidden units to use")
    parser.add_argument('-e','--epochs',type=int,help="Number of epochs",default=3)
    parser.add_argument('--gpu',action='store_true',help="Use GPU")
    input_args = parser.parse_args()

    regurgitate(input_args)
    main(input_args)
