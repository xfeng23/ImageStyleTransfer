import torch
import time
from transformer_net import *

'''
TransformerNet_Flexible takes in number of channels and first 
convolution kernel size. The hope is that it will replace the following
models and will serve as all purpose, flexible model. 
'''
class TransformerNet_Flexible(torch.nn.Module):
    def __init__(self, ch=32, k=9):
        super(TransformerNet_Flexible, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, ch, kernel_size=k, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = ConvLayer(ch, 2 * ch, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(2 * ch, affine=True)
        self.conv3 = ConvLayer(2 * ch, 4 * ch, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(4 * ch, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(4 * ch)
        self.res2 = ResidualBlock(4 * ch)
        self.res3 = ResidualBlock(4 * ch)
        self.res4 = ResidualBlock(4 * ch)
        self.res5 = ResidualBlock(4 * ch)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(4 * ch, 2 * ch, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(2 * ch, affine=True)
        self.deconv2 = UpsampleConvLayer(2 * ch, ch, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(ch, affine=True)
        self.deconv3 = ConvLayer(ch, 3, kernel_size=k, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y

'''
WARNING: Not working. Need to get ConvTranspose to double dimensions.
'''
class TransposeConvTransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransposeConvTransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = ConvTransposeLayer(128, 64, kernel_size=3, stride=2, padding=1)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = ConvTransposeLayer(64, 32, kernel_size=3, stride=2, padding=1)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.conv1(X)
        y = self.in1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.in2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.in3(y)
        y = self.relu(y)
        
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.deconv1(y)
        y = self.in4(y)
        y = self.relu(y)
        y = self.deconv2(y)
        y = self.in5(y)
        y = self.relu(y)
        
        y = self.deconv3(y)
        return y
