from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from fast_neural_style.neural_style.transformer_net import TransformerNet
from io import BytesIO
from pynvml import *
import time, torch, requests, os
import multiprocessing

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def gram_matrix(y):
    """ Calculate gram matrix used in stylization algorithm """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    """ normalize using imagenet mean and std """
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    batch = batch / Variable(std)
    return batch

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def tv_loss(img, tv_weight):
    """
    Computes total variation loss. TV loss encourages smoothness among adjacent pixels.

    Inputs:
    - img: PyTorch Variable of shape (B, 3, H, W) holding an input image. B is batch size. 
    - tv_weight: Scalar giving the weight to use for the TV loss

    Returns: 
    - loss: PyTorch Variable holding a scalar giving the TV loss
      for img weighted by tv_weight
    """
    B, C, H, W = img.size()
    h_var = (img[:, :, :H-1, :W-1] - img[:, :, 1:, :W-1])**2
    w_var = (img[:, :, :H-1, :W-1] - img[:, :, :H-1, 1:])**2
    return tv_weight * torch.sum(h_var + w_var) / B 


def stylize(img, tc, model_path, style_model, gpuHandle, hostip, url, USE_GPU=1):
    """
    Establish and inference the CNN network for stylization.

    Inputs:
    img(PIL object): PIL image that send as input to network.
    tc(telemetry object): App insights telemetry object that used to log trace.
    model_path(str): Path to the pytorch model used for stylization.
    style_model(transformerNet object): CNN network object.
    gpuHandle(nvidia gpu handler): used to get gpu usage percentage info.
    USE_GPU(bool): Whether to use GPU or not.

    Outputs:
    [0](PIL object): Stylized image.
 
    """
    try: 
        stylize_start = time.time()     
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        img = content_transform(img)
        img = img.unsqueeze(0)
        img = Variable(img, volatile=True)
        tc.track_trace('{}|{} image stylization preprocessing done.'.format(url, hostip)) #log
        GPU_start = time.time()
        # track GPU memory cached
        gpuUsage = []
        for gpu in gpuHandle:
            gpuInfo = nvmlDeviceGetMemoryInfo(gpu)
            gpuUsage.append(gpuInfo.used / gpuInfo.total * 100)
        tc.track_trace('{}|{} gpu usages: {}'.format(url, hostip, gpuUsage))
        style_model.load_state_dict(torch.load(model_path))
        tc.track_trace('{}|{} stylization loading model done.'.format(url, hostip)) #log
        style_model.cuda()
        tc.track_trace('{}|{} stylization model to cuda done.'.format(url, hostip)) #log
        img = img.cuda()
        tc.track_trace('{}|{} stylization image to cuda done.'.format(url, hostip)) #log
        output = style_model(img)
        output = output.cpu()
        tc.track_trace('{}|{} stylization done.'.format(url, hostip))
        GPU_time = (time.time() - GPU_start) * 1000
        tc.track_trace('{}|{} stylization gpu runtime: {}'.format(url, hostip, GPU_time)) #log
        output_data = output.data[0]
        img = output_data.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        tc.track_trace('{}|{} image stylization postprocessing done.'.format(url, hostip)) #log
        return img
    except Exception as e:
        tc.track_exception()
        return None
        
