#Imports
import argparse
from PIL import Image
import torch
import numpy as np
from math import ceil
from torchvision import models
from funcs import check_gpu

#Define arg_parser to parse arguments
def arg_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type = str, default = './flowers/test/99/image_07838.jpg', help='image path')
    parser.add_argument('--checkpoint', type = str, default = './checkpoint.pth', help = 'path to saved checkpoint')
    parser.add_argument('--topk', type = int, default = 5, help = 'set number of top K matches')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'path to categories file')
    parser.add_argument('--gpu', action = 'store', default = 'gpu')
    
    args = parser.parse_args()
    
    return args

#Define load_checkpoint to load the saved checkpoint for prediction
def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    architecture = checkpoint['architecture']
    
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

#Define process_image to process image for prediction
def process_image(image):
    image = Image.open(image)

    width, height = image.size

    if width < height: 
        resize=[256, 256**600]
    else: 
        resize=[256**600, 256]
        
    image.thumbnail(size=resize)

    center = width/4, height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)/255 
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image-mean)/std
        
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

#Define predict to make the prediction
def predict(image_tensor, model, device, cat_to_name, top_k):
    model.to("cpu")
    model.eval()
    
    image_t = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(torch.FloatTensor)

    
    log_ps = model(image_t)
    ps = torch.exp(log_ps)
    top_ps, top_classes = ps.topk(top_k)
    
    top_ps = np.array(top_ps.detach())[0]
    top_classes = np.array(top_classes.detach())[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[key] for key in top_classes]
    top_flowers = [cat_to_name[key] for key in top_classes]

    return top_ps, top_classes, top_flowers

#Define print_probs to print the top K probabilities of the prediction
def print_probs(flowers, probs):
    for i, j in enumerate(zip(probs, flowers)):
        print('{}. {}: {}%'.format(i+1, j[1], ceil(j[0]*100)))
