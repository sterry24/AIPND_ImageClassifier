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

    ## Load the model with Checkpoint
    model = mutils.load_checkpoint(in_args.checkpoint)

    if in_args.gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    model.to(device)

    prob,classes = predict(in_args,model,device)

    with open(in_args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    ## Need to determine index from image filename
    idx = None
    for d in in_args.image.strip().split('/'):
        if d.isdigit():
            idx = d

    if idx is not None:
        correct_val = cat_to_name[idx]
    else:
        sys.stdout.write("ERROR: Could not determine idx from image pathname. Exiting...")
        sys.exit()

    correct_val = cat_to_name[idx]
    sys.stdout.write("\nTruth: %s\n" %correct_val)

    predicted_names = []
    for cidx in classes:
        predicted_names.append(cat_to_name[cidx])
    sys.stdout.write("Returning %d top predictions:\n" %in_args.top_k)
    for i in range(len(prob)):
        sys.stdout.write('\t%d) Predicted %s with probability %.3f\n' %(i+1,predicted_names[i],prob[i]))


def predict(in_args,model,device):

    model.eval()
    model.to(device)

    img = Image.open(in_args.image)
    img_arr = dutils.process_image(img)
    inputs = torch.from_numpy(img_arr).type(torch.cuda.FloatTensor)
    inputs.unsqueeze_(0)
    inputs.to(device)
    with torch.no_grad():
        output = model.forward(inputs)

        prob,labels = torch.topk(output,in_args.top_k)
        top_prob = prob.exp()

    class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}

    classes = list()
    cpu_labels = labels.cpu()
    for label in cpu_labels.detach().numpy()[0]:
        classes.append(class_idx_dict[label])

    return top_prob.cpu().numpy()[0],classes


def regurgitate(in_args):

    """ print the arguments, specified by user or program defaults"""

    print("\nBeginning prediction with following inputs:\n")
    for arg in vars(in_args):
        if getattr(in_args,arg) is not None:
            print("%-14s:\t" %arg,getattr(in_args,arg))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a new network')
    parser.add_argument('image',type=str,help="The path to the image to test")
    parser.add_argument('checkpoint',type=str,help="The path to the checkpoint to load",default="./checkpoint.pth")
    parser.add_argument('--top_k',type=int,help="The number of most likely classes to return",default=5)
    parser.add_argument('--category_names',type=str,help="The path to the JSON mapping categories to names",default='cat_to_name.json')
    parser.add_argument('--gpu',action='store_true',help="Use GPU")
    input_args = parser.parse_args()

    regurgitate(input_args)
    main(input_args)
