import argparse
import os
import sys
import time
import datetime

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from PIL import Image

import utils
from transformer_net import TransformerNet
from experimental_transformer_net import *
from transformer_net_padding import TransformerNetZeroPad
from vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        #if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
        #    os.makedirs(args.checkpoint_model_dir)
        if args.intermediate_dir is not None and not (os.path.exists(os.path.join(args.intermediate_dir, args.model_file_name))):
            now = datetime.datetime.now()
            day = '{}_{}_{}'.format(now.year, now.month, now.day)
            os.makedirs(os.path.join(args.intermediate_dir, day, args.model_file_name))

    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    print "gpu: ", args.gpu_id
    print "file name: " + args.model_file_name

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_data)

    if args.test_dataset and args.intermediate_dir:
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        test_dataset = datasets.ImageFolder(args.test_dataset, content_transform)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        now = datetime.datetime.now()
        day = '{}_{}_{}'.format(now.year, now.month, now.day)
        intermediate_dir = os.path.join(args.intermediate_dir, day, args.model_file_name)

    #transformer = TransformerNet()
    transformer = TransformerNetZeroPad()
    if args.init_model is not None:
        transformer.load_state_dict(torch.load(args.init_model))
 
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1)

    if args.cuda:
        transformer.cuda()
        vgg.cuda()
        style = style.cuda()

    style_v = Variable(style)
    style_v = utils.normalize_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_tv_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if args.cuda:
                x = x.cuda()

            y = transformer(x)

            y_norm = utils.normalize_batch(y)
            x_norm = utils.normalize_batch(x)

            features_y = vgg(y_norm)
            features_x = vgg(x_norm)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            if args.tv_weight:
                tv_loss = utils.tv_loss(y, args.tv_weight)
                total_loss += tv_loss

            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            if args.tv_weight:
                agg_tv_loss += tv_loss.data[0]

            # Print training progress
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttv: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  agg_tv_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss + agg_tv_loss) / (batch_id + 1)
                )
                print(mesg)


            # Save intermediate results, starting at first batch
            # Only save first 10 batch intervals of each epoch, to not overload directory with images
            if args.test_dataset is not None and args.intermediate_dir and (batch_id + 1) % args.intermediate_interval == 0: # and (batch_id + 1) <= 10 * args.intermediate_interval:
                if args.cuda:
                    content_output = x.cpu()
                    style_output = y.cpu()

                style_output_data = style_output.data[0]
                content_output_data = content_output.data[0]
                style_name = 'Epoch{}_Batch{}_coco_style.jpg'.format(e + 1, batch_id + 1)
                content_name = 'Epoch{}_Batch{}_coco_content.jpg'.format(e + 1, batch_id + 1)
                style_path = os.path.join(intermediate_dir, style_name)
                content_path = os.path.join(intermediate_dir, content_name)
                utils.save_image(style_path, style_output_data)
                utils.save_image(content_path, content_output_data)

                for i, (data, _) in enumerate(test_loader):
                    if args.cuda:
                        data = data.cuda()
                    data = Variable(data, volatile=True)
                    
                    output_train = transformer(data)
                    if args.cuda:
                        output_train = output_train.cpu()
                    output_train_data = output_train.data[0]
                    train_name = 'Epoch{}_Batch{}_{}_train.jpg'.format(e + 1, batch_id + 1, i + 1)
                    train_path = os.path.join(intermediate_dir, train_name)
                    utils.save_image(train_path, output_train_data)


        # Checkpoint model
        if args.checkpoint_model and (e + 1) < args.epochs:
            transformer.eval()
            if args.cuda:
                transformer.cpu()
            ckpt_model_filename = args.model_file_name + "ckpt_epoch_" + str(e + 1) + ".pth"
            ckpt_model_path = os.path.join(args.save_model_dir, ckpt_model_filename)
            torch.save(transformer.state_dict(), ckpt_model_path)
            if args.cuda:
                transformer.cuda()
            transformer.train()

    # save model
    transformer.eval()
    if args.cuda:
        transformer.cpu()

    save_model_path = os.path.join(args.save_model_dir, args.model_file_name + ".pth")
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--shuffle-data", type=bool, default=False,
                                  help="shuffle training data")
    train_arg_parser.add_argument("--test-dataset", type=str, required=False,
                                  help="path to testing dataset, the path should point to a folder "
                                       "containing another folder with all the testing images")
    train_arg_parser.add_argument("--init-model", type=str, required=False,
                                  help="path to pretrained model to initialize for fine tuning")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--model-file-name", type=str, required=True,
                                  help="name of file to which trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model", type=bool, default=False,
                                  help="checkpoint model after ever epoch")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--tv-weight", type=float, required=False,
                                  help="weight for TV-loss, default is None")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--intermediate-dir", type=str, required=False,
                                  help="path to folder where intermediate results will be saved.")
    train_arg_parser.add_argument("--intermediate-interval", type=int, default=200,
                                  help="number of images after which the model is tested, default is 200")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--gpu-id", type=int, default=0,
                                  help="which GPU to train on, default is first GPU")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        if args.cuda:
            # which gpu to train on
            with torch.cuda.device(args.gpu_id):
                train(args)
        else:
            train(args)

    else:
        content_image = utils.load_image(args.content_image, scale=args.content_scale)
        output_image = utils.stylize(content_image, args.model)
        utils.save_image(args.output_image, output_image)


if __name__ == "__main__":
    main()
