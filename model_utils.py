import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def get_model(arch,pretrain=False):
    """Retrieves the model specified by name from torchvision.models
       PARAMETERS:
                  arch:     String; the model architecture to load
                  pretrain: Boolean; get the pretrained model, default False

        RETURNS:
                  model:    The loaded model
    """

    try:
        model =  getattr(models,arch)(pretrained=pretrain)
        for param in model.parameters():
            param.requires_grad = False
        return model
    except:
        print("ERROR: Architecture %s not recognized, please try again" %arch)
        return None


def create_classifier(model,output_size,drop=0.2,hidden_units=None):

    """create a classifier to be added to the model.  Uses nn.Sequential

       PARAMETERS:
                   model:            PyTorch model; the model the classifier
                                     will be added to
                   output_size: int; the size of the output layer
                   drop:             the dropout percent
                   hidden_units: list; a list of the hidden_layer sizes,
                                 default is None, which will force one
                                 hidden_layer
       RETURNS:

                   classifier: the classifier object to be used in the model
    """

    input_size = model.classifier[0].in_features
    if hidden_units is None:
        ## StackExchange:choose num of hidden layers and nodes
        hidden_units = [int(np.mean([input_size,output_size]))]

    ## Set the input layer
    input_layer = [nn.Linear(input_size,hidden_units[0])]

    ## Set any hidden layers
    hidden = []
    layer_sizes = zip(hidden_units[:-1],hidden_units[1:])
    hidden.extend([nn.Linear(iu,ou) for iu,ou in layer_sizes])

    ## combine the input layer with hidden layers
    layers = []
    layers.append(input_layer[0])
    ## add activation and dropout
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(p=drop))
    ## Loop through hidden layers, adding layer,
    ## activation, and dropout
    for i in range(len(hidden)):
        layers.append(hidden[i])
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=drop))

    ## add the output layer
    layers.append(nn.Linear(hidden_units[-1],output_size))
    layers.append(nn.LogSoftmax(dim=1))

    layers = nn.ModuleList(layers)

    classifier = nn.Sequential(*layers)

    return classifier

def validation(model,dataset,criterion,device):

    """perform validation when training a model

       PARAMETERS:
                   model:     the PyTorch model being trained
                   dataset:   the dataset to perform validation on
                   criterion: the loss function

       RETURNS:
                   test_loss: float; measure of validation loss
                   accuracy:  float; measure of prediction strength
    """

    test_loss = 0
    accuracy = 0
    for images, labels in dataset:

        images,labels = images.to(device),labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output,labels).item()

        ps = torch.exp(output).data
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss,accuracy

def save_checkpoint(chkpt,model,filename="checkpoint.pth"):

    """Save the model info to be reused/retrained later.

       PARAMETERS:
                   chkpt: dict; contains the model state
                          and other info to be used for
                          inference or to continue training.
                   filename: string; the filepath/filename
                          to save the checkpoint to.

       RETURNS:
                    None
    """

    chkpt['classifier'] = model.classifier
    chkpt['state_dict'] = model.state_dict()
    torch.save(chkpt,filename)

def load_checkpoint(filepath):

    """Load an existing checkpoint to create a model

       PARAMETERS:
                    filepath: string; the filepath to the
                           checkpoint file

       RETURNS:
                    model: PyTorch model with checkpoint loaded

    """

    checkpoint = torch.load(filepath)

    model = get_model(checkpoint['arch'],pretrain=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model
