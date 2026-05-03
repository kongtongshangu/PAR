from PIL import Image
from utills import *
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import os.path
import argparse
import torch.optim as optim
import torchvision
from torchvision import transforms
from model import *
from dataprocess import CIFAR10C
import warnings
import sys
import time
import logging
from copy import deepcopy
from collections import OrderedDict
from pprint import pformat

import PAR

warnings.filterwarnings('ignore')

cudnn.benchmark = True
cudnn.deterministic = True


# --------------------------------------------------
# Parse input arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(description='SNN with BNTT TTA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_dir', type=str, help='Model Path')
parser.add_argument('--data_root', type=str, help='Data Path')
parser.add_argument('--seed', default=0, type=int, help='Random seed')
parser.add_argument('--num_steps', default=25, type=int, help='Number of time-step')
parser.add_argument('--batch_size', default=64, type=int,   help='Batch size')
parser.add_argument('--leak_mem',  default=0.95, type=float, help='Leak_mem')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--domain', default='cifar10', type=str, help='domain')
parser.add_argument('--gpu', default='0', type=str, help='gpu')
parser.add_argument('--method', default='None', type=str, help='method [None, PAR]')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--l', default=1, type=int, help='augment strength')
parser.add_argument('--output_dir', default='./result', type=str, help='output_dir')

parser.add_argument('--tau', default=0.35, type=float, help='temporal stability threshold')
parser.add_argument('--eta', default=0.02, type=float, help='Hebbian learning rate')

global args
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logger(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_root = os.path.join(args.output_dir, args.method)
    os.makedirs(save_root, exist_ok=True)

    log_file = os.path.join(save_root, f"{args.method}_{timestamp}.log")

    logger = logging.getLogger("BNTT_CIFAR10C")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized.")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    logger.info("Arguments:\n" + pformat(vars(args)))

    return logger, save_root


# --------------------------------------------------
# Initialize seed
# --------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------------------------------------------------
# SNN configuration parameters
# --------------------------------------------------
# Leaky-Integrate-and-Fire (LIF) neuron parameters
user_foldername = 'cifar10vgg9_timestep25_lr0.3_epoch100_leak0.95'
leak_mem = args.leak_mem

# SNN learning and evaluation parameters
num_steps = args.num_steps

# --------------------------------------------------
# Load  dataset
# --------------------------------------------------
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression'
]

num_cls = 10
img_size = 32

def build_data(domain):
    if domain == 'cifar10':
        adapt_set = None
        test_set = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=False, transform=transform_test)
    else:
        adapt_set = CIFAR10C(root=args.data_root, name=domain, transform=None, level=5)
        test_set = CIFAR10C(root=args.data_root, name=domain, transform=transform_test, level=5)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return adapt_set, testloader

def CTTA_Method(method=None, model=None):
    if method == 'None':
        return model

    elif method == 'PAR':
        model = PAR.configure_model(model)
        model = PAR.PAR(model, optimizer=None, steps=1, episodic=False, input_shape=(1, 3, img_size, img_size), target_layers=("pool3",),
            stability_thresh=0.65, burst_tolerance=1.80, adapt_gate_min=args.tau, gate_scale=8.0, lambda_struct=1.0, lambda_orth=0.05,
            lambda_temporal=1.0, fast_lr=args.eta, fast_decay=0.01, theta_momentum=0.95, rate_momentum=0.95, homeo_weight=0.30, fast_clamp=0.10)
        params, names = PAR.collect_params(model)
        optimizer_adapt = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
        model.set_optimizer(optimizer_adapt)
        model = model.cuda()
        return model


if __name__ == '__main__':
    logger, save_root = setup_logger(args)
    all_acc = []
    model = SNN_VGG9_BNTT(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls).cuda()
    modelsave = torch.load(args.log_dir + '/' + user_foldername + '_bestmodel.pth.tar')
    model.load_state_dict(modelsave['state_dict'])
    model = CTTA_Method(args.method, model)

    for corruption in CORRUPTIONS:
        adapt_set, testloader = build_data(corruption)
        logger.info('=' * 60)
        logger.info(f'Running corruption: {corruption}')
        acc_top1, acc_top5 = [], []
        for j, data in enumerate(testloader):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            out = model(images)
            prec1, prec5 = accuracy(out, labels, topk=(1, 5))
            acc_top1.append(float(prec1))
            acc_top5.append(float(prec5))

        test_accuracy = np.mean(acc_top1)
        final_msg = "Test accuracy on {} domain: {:.2f}%".format(corruption, test_accuracy)
        logger.info(final_msg)
        all_acc.append(test_accuracy)

    mean_acc = np.mean(all_acc)
    logger.info('=' * 60)
    logger.info('All corruptions finished.')
    logger.info('Average accuracy over 15 corruptions: {:.2f}%'.format(mean_acc))