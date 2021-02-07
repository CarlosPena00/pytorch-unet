# STD MODULE
import os
import numpy as np
import cv2
import random

# TORCH MODULE
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torchvision
import torch.backends.cudnn as cudnn

# PYTVISION MODULE
from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

# LOCAL MODULE

from torchlib.datasets import dsxbdata
from torchlib.segneuralnet import SegmentationNeuralNet

from aug import get_transforms_aug, get_transforms_det, get_simple_transforms, get_transforms_geom_color

from argparse import ArgumentParser
import datetime
from matplotlib import pyplot as plt


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean = (0.485, 0.456, 0.406), std  = (0.229, 0.224, 0.225)):
        mean     = torch.as_tensor(mean)
        std      = torch.as_tensor(std)
        std_inv  = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

n = NormalizeInverse()

def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('data', metavar='DIR', 
                        help='path to dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='divice number (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--batch-size-train', default=48, type=int, metavar='N', 
                        help='mini-batch size of train set (default: 48)')    
    parser.add_argument('--batch-size-test', default=48, type=int, metavar='N', 
                        help='mini-batch size of test set (default: 48)') 
    
    parser.add_argument('--count-train', default=48, type=int, metavar='N', 
                        help='count of train set (default: 100000)')    
    parser.add_argument('--count-test', default=48, type=int, metavar='N', 
                        help='count of test set (default: 5000)')     

    parser.add_argument('--num-channels', default=3, type=int, metavar='N', 
                        help='num channels (default: 3)')      
    parser.add_argument('--num-classes', default=3, type=int, metavar='N', 
                        help='num of classes (default: 3)') 
    
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--snapshot', '-sh', default=10, type=int, metavar='N',
                        help='snapshot (default: 10)')
    parser.add_argument('--project', default='./runs', type=str, metavar='PATH',
                        help='path to project (default: ./runs)')
    parser.add_argument('--name', default='exp', type=str,
                        help='name of experiment')
    parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='NAME',
                    help='name to latest checkpoint (default: none)')
    parser.add_argument('--arch', default='simplenet', type=str,
                        help='architecture')
    parser.add_argument('--finetuning', action='store_true', default=False,
                    help='Finetuning')
    parser.add_argument('--loss', default='cross', type=str,
                        help='loss function')
    parser.add_argument('--opt', default='adam', type=str,
                        help='optimize function')
    parser.add_argument('--scheduler', default='fixed', type=str,
                        help='scheduler function for learning rate')

    parser.add_argument('--image-crop', default=512, type=int, metavar='N',
                        help='image crop')
    parser.add_argument('--image-size', default=256, type=int, metavar='N',
                        help='image size')
    
    parser.add_argument('--parallel', action='store_true', default=False,
                    help='Parallel')    
    parser.add_argument('--post-method', default='map', type=str,
                        help='Post processing method, | map | th | wts')
    parser.add_argument('--weight', default='', type=str,
                        help='weight, | SAW | DWM | ')
    parser.add_argument('--pad', default=0, type=int,
                        help='pad px')
    parser.add_argument('--load-segments', default=False,
                        help='load segments')
    parser.add_argument('--count-segs', type=int, default=5,
                        help='count of segs')
    return parser



def main():
    
    # parameters
    parser       = arg_parser()
    args         = parser.parse_args()
    parallel     = args.parallel
    imcrop       = args.image_crop
    imsize       = args.image_size
    num_classes  = args.num_classes
    num_channels = args.num_channels    
    count_train  = args.count_train #10000
    count_test   = args.count_test #5000
    post_method  = args.post_method
    weight       = args.weight
    pad          = int(args.pad)
    count_segs   = int(args.count_segs)
    load_segs    = args.load_segments
    
    use_weights  = weight != ''
    
    folders_contours ='touchs'
        
    print('Baseline clasification {}!!!'.format(datetime.datetime.now()))
    
    
    network = SegmentationNeuralNet(
        patchproject=args.project,
        nameproject=args.name,
        no_cuda=args.no_cuda,
        parallel=parallel,
        seed=args.seed,
        print_freq=args.print_freq,
        gpu=args.gpu
        )

    network.create( 
        arch=args.arch, 
        num_output_channels=num_classes, 
        num_input_channels=num_channels+count_segs if load_segs else num_channels,
        loss=args.loss, 
        lr=args.lr, 
        momentum=args.momentum,
        optimizer=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning,
        size_input=imsize
        )
    
    epoch, value = (network.resume( os.path.join(network.pathmodels, args.resume ) ))
    assert epoch != 0, "Model W not found"
    
    cudnn.benchmark = True

    test_data = dsxbdata.ISBIDataset(
        args.data, 
        "test", 
        folders_labels=f'labels{num_classes}c',
        count=None,
        num_classes=num_classes,
        num_channels=num_channels,
        transform=get_simple_transforms(pad=pad),
        use_weight=use_weights,
        weight_name=weight,
        load_segments=load_segs,
        shuffle_segments=False,
        count_segments=count_segs
    )
        
    test_loader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, 
        num_workers=args.workers, pin_memory=network.cuda, drop_last=False)
    
    print(f"Model Val set: {value:0.3f}")
    network.evaluate( test_loader, epoch)
                   
    print("Eval Finished!")
    print("DONE!!!")



if __name__ == '__main__':
    main()
