import datetime
import multiprocessing as mp
#mp.set_start_method('spawn')
import os
import re
import sys

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

lib_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(lib_path)

from fast_neural_style.neural_style import utils
from fast_neural_style.neural_style.transformer_net import TransformerNet
from fast_neural_style.neural_style.vgg import Vgg16

torch.backends.cudnn.enabled = False
SEED = 42
STYLE_IMAGES_PATH = '../autotrain/style_images'
STYLE_WEIGHTS = [5e9, 1e10]
STYLE_SIZES = [256, 512, None]
GPU_COUNT = torch.cuda.device_count()
IMAGE_SIZE = 256
BATCH_SIZE = 4
DATASET_PATH = '/data/t-elch/datasets/coco2014_15k_selfies_15k'
TEST_IMAGES_PATH = '../autotrain/test_images'
RESULTS_PATH = '../autotrain/results'
EPOCHS = 2
LEARNING_RATE = 1e-3
CONTENT_WEIGHT = 1e5

def check_path(dir_path):
    try:
        if dir_path is not None and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    except OSError as e:
        print(e)
        sys.exit(1)

def save_test_images(transformer, save_dir, use_cuda):
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    test_dataset = datasets.ImageFolder(TEST_IMAGES_PATH, content_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Print results 
    for i, (data, _) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
            
        output = transformer(data)
        if use_cuda:
            output = output.cpu()
        output_data = output.data[0]
        image_name = 'content_image{}.jpg'.format(i + 1)
        path = os.path.join(save_dir, image_name)
        utils.save_image(path, output_data)




def train_style_over_hyperparameters(img, style_name, cuda_device):
    with torch.cuda.device(cuda_device):
        print("Beginning to train style: {}. Using GPU: {}".format(style_name, cuda_device))
        for style_weight in STYLE_WEIGHTS:
            for style_size in STYLE_SIZES:
                train_style(img, style_name, style_weight, style_size)

def train_styles(style_images):
    #processes = []
    # for index, (image, style) in enumerate(style_images):
    #     with torch.cuda.device(index):
            #print "Beginning to train style: {}. Using GPU: {}".format(style, index)
    #mp.set_start_method('spawn')
    #mp.get_context('spawn')
    ctx = mp.get_context('spawn')
    processes = [ctx.Process(target=train_style_over_hyperparameters, args=(image, style, index)) for index, (image, style) in enumerate(style_images)]
            #processes.append(p)
    for p in processes:
        p.start()

            #train_style_over_hyperparameters(image, style, cuda_device)

def automate_train():
    style_files = os.listdir(STYLE_IMAGES_PATH)
    # we train on GPU, so can only support GPU_COUNT styles
    if len(style_files) > GPU_COUNT:
        style_files = style_files[:GPU_COUNT]

    # style_images is a list of PIL Images
    style_images = []
    for f in style_files:
        style_name = re.match('([\s|\w]+)\.jpg', f).group(1)
        style_path = os.path.join(STYLE_IMAGES_PATH, f)
        img = Image.open(style_path)
        style_images.append((img, style_name))
        
    train_styles(style_images)


"""
Parameters:
- Styles: A list of style images
"""
def train_style(style_image, style_name, style_weight, style_size, use_cuda=True):
    now = datetime.datetime.now()
    day = '{}_{}_{}'.format(now.year, now.month, now.day)
    day_dir = os.path.join(RESULTS_PATH, day)
    style_parameters = "styleweight_{:.2e}".format(style_weight)
    if style_size is not None:
        style_parameters += "_stylesize_{}".format(style_size)
    save_results_dir = os.path.join(day_dir, style_name, style_parameters)
    check_path(save_results_dir)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if use_cuda:
        torch.cuda.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(DATASET_PATH, transform)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)


    transformer = TransformerNet()

    #if args.init_model is not None:
    #    transformer.load_state_dict(torch.load(args.init_model))
 
    optimizer = Adam(transformer.parameters(), LEARNING_RATE)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = style_image
    if style_size is not None:
        style = style.resize((style_size, style_size), Image.ANTIALIAS)
    style = style_transform(style)
    style = style.repeat(BATCH_SIZE, 1, 1, 1)

    if use_cuda:
        transformer.cuda()
        vgg.cuda()
        style = style.cuda()

    style_v = Variable(style)
    style_v = utils.normalize_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(EPOCHS):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if use_cuda:
                x = x.cuda()

            y = transformer(x)

            y_norm = utils.normalize_batch(y)
            x_norm = utils.normalize_batch(x)

            features_y = vgg(y_norm)
            features_x = vgg(x_norm)

            content_loss = CONTENT_WEIGHT * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss

            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]


    save_test_images(transformer, save_results_dir, use_cuda)

    # save model
    transformer.eval()
    if use_cuda:
        transformer.cpu()


    model_file_name = style_name + ".pth"
    save_model_path = os.path.join(save_results_dir, model_file_name)
    torch.save(transformer.state_dict(), save_model_path)
    save_image_path = os.path.join(save_results_dir, style_name + ".jpg")
    style_image.save(save_image_path)

    print("\nDone, trained model saved at", save_model_path)
    print("For style weight: {} and style sizei: {}, content loss: {}, style loss: {}".format(style_weight, style_size, agg_content_loss, agg_style_loss))

if __name__ == '__main__':
    #mp.set_start_method('spawn')
    automate_train()
