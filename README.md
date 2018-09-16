# AIPND_ImageClassifier
Udacity AI Programming with Python Nanodegree Image Classifier Project

## Requirements
- python3
- matplotlib
- numpy
- pillow
- PyTorch

## Training a classifier
### Required Input:
- data_dir: Data Directory is expected to be in the structure defined by PyTorch ImageFolder
### Optional Input:
- --save_dir:      The path to write the model checkpoint file to
- --arch:          The architecture to use, check PyTorch torchvision.models for available options
- --learning_rate: The learning rate to be used in the code
- --hidden_units:  The input values to be applied to the hidden layers
- --epochs:        The number of times to train the classifier
- --gpu:           Use this flag to run with gpu
### Example command
```
python train.py /path/to/data/dir --save_dir /path/to/output/dir --arch vgg16 --learning_rate 0.001 --hidden_units 512 256 --epochs 3 --gpu 
```

## Making a prediction
### Required Input:
- image: The path to the image to predict an output for
- checkpoint: The path to the checkpoint file to read in classifier creation
### Optional Input:
- --top_k:          The number of most likely classes to return
- --category_names: The path to the JSON file mapping categories to names
- --gpu:            Use this flag to run with gpu
### Example command
```
python predict.py /path/to/input/image /path/to/checkpoint/file.pth --top_k 5 --category_names /path/to/file.json --gpu 
```
