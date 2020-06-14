#Imports
import argparse
import torch 
from torchvision import datasets, transforms, models
from torch import nn, optim
import time
from PIL import Image
import numpy as np

#Define arg_parser to parse arguments
def arg_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'choose model architecture')
    parser.add_argument('--save_dir', type = str, default = './checkpoint.pth', help = 'path where model is to be saved')
    parser.add_argument('--lr', type = float, default = 0.002, help = 'set the learning rate for model')
    parser.add_argument('--hidden_units', type = int, default = 4096, help = 'set the number of hidden units')
    parser.add_argument('--epochs', type = int, default = 15, help = 'set the number of epochs for training the model')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'set the dropout value')
    parser.add_argument('--gpu', action = 'store', default = 'cuda')
    
    args = parser.parse_args()
    return args

#Define train_transforms for transformation of training data
def train_transforms(train_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    
    return train_data

#Define valid_transforms for transformation of validation data
def valid_transforms(valid_dir):
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    return valid_data

#Define test_transforms for transformation of test data
def test_transforms(test_dir):
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    return test_data

#Define train_loader to load training data
def train_loader(data, batch_size=64):
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return trainloader

#Define valid_loader to load validation data
def valid_loader(data, batch_size=32):
    validloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return validloader

#Define test_loader to load testing data
def test_loader(data, batch_size=32):
    testloader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return testloader

#Define chech_gpu to check for gpu
def check_gpu(gpu_arg):
    if gpu_arg:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device

# Define load_model to load the pretrained model
def load_model(architecture):

    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    return model

#Define build_classifier to create a classifier
def build_classifier(model, hidden_units, dropout):
    
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    return classifier


#Define train_network to train the network
def train_network(model, criterion, optimizer, epochs, device, trainloader, validloader):
    
    epochs = epochs
    steps = 0
    running_loss = 0
    
    print('Training started..\n')
    
    for epoch in range(epochs):
        start = time.time()
        for images, labels in trainloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:

                    images, labels = images.to(device), labels.to(device)


                    log_ps = model(images)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Accuracy: {:.3f}".format(accuracy/len(validloader)))

            running_loss = 0
            model.train()
            
            end = time.time()
            print("Epoch time: {}".format(end-start))
            
    return model

#Define test_model to test the model with testing data
def test_model(model, criterion, testloader, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    start = time.time()
    with torch.no_grad():
        for images, labels in testloader:
                
            images, labels = images.to(device), labels.to(device)
                
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Test loss: {:.3f}.. ".format(test_loss/len(testloader)),
        "Accuracy: {:.3f}".format(accuracy/len(testloader)))

    running_loss = 0
    model.train()
            
    end = time.time()
    print("Test time: {}".format(end-start))
    
#Define save_checkpoint to save the checkpoint
def save_checkpoint(model, arch, save_dir, train_data):
    
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {
    'architecture': arch,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
    }
    
    torch.save(checkpoint, save_dir)
