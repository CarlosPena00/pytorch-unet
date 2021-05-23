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
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--load-extra', type=bool, default=False,
                        help='load extra')
    parser.add_argument('--cascade', type=str, default='none',
                        help='load extra')
    parser.add_argument('--load-segs', type=int, default=1,
                        help='load segments')
    parser.add_argument('--segs-per-forward', type=int, default=-1,
                        help='count of segs')
    parser.add_argument('--use-ori', type=int, default=1,
                        help='Use Original image')
    parser.add_argument('--just-eval', type=int, default=0,
                        help='If just eval')
    parser.add_argument('--use-bagging', type=int, default=0,
                        help='If just eval')
    parser.add_argument('--bagging-seed', type=int, default=2021)
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
    use_ori      = int(args.use_ori)
    use_weights  = weight!=''
    segs_per_forward = int(args.segs_per_forward)
    load_segs    = bool(args.load_segs)
    just_eval    = int(args.just_eval)
    use_bagging  = int(args.use_bagging)
    bagging_seed = int(args.bagging_seed)
    
    
    folders_contours ='touchs'
        
    print('Baseline clasification {}!!!'.format(datetime.datetime.now()))
    print('\nArgs:')
    [ print('\t* {}: {}'.format(k,v) ) for k,v in vars(args).items() ]
    print('')
    
    num_input_channels = (num_channels * use_ori) + (load_segs * segs_per_forward)
    writer = SummaryWriter('logs/'+args.name)
    
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
        num_input_channels=num_input_channels,
        loss=args.loss, 
        lr=args.lr, 
        momentum=args.momentum,
        optimizer=args.opt,
        lrsch=args.scheduler,
        pretrained=args.finetuning,
        size_input=imsize,
        cascade_type=args.cascade,
        writer=writer,
        segs_per_forward=segs_per_forward,
        use_ori=use_ori,
        data_name=args.data
        )
    
    cudnn.benchmark = False #due to augmentation

    # resume model
    if args.resume:
        network.resume( os.path.join(network.pathmodels, args.resume ) )
        
    if not just_eval:

        # datasets
        # training dataset
        print("Warring! training with shuffle false")
        train_data = dsxbdata.ISBIDataset(
            args.data, 
            'train', 
            folders_labels=f'labels{num_classes}c',
            count=count_train,
            num_classes=num_classes,
            num_channels=num_channels,
            transform=get_transforms_geom_color(pad=pad),
            use_weight=use_weights,
            weight_name=weight,
            load_segments=load_segs,
            shuffle_segments=True,
            use_ori=use_ori,
            use_bagging=use_bagging,
            bagging_seed=bagging_seed
        )

        train_loader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True, 
            num_workers=args.workers, pin_memory=False, drop_last=True )

        val_data = dsxbdata.ISBIDataset(
            args.data, 
            "val", 
            folders_labels=f'labels{num_classes}c',
            count=count_test,
            num_classes=num_classes,
            num_channels=num_channels,
            transform=get_simple_transforms(pad=pad),
            use_weight=use_weights,
            weight_name=weight,
            load_segments=load_segs,
            shuffle_segments=True,
            use_ori=use_ori
        )

        val_loader = DataLoader(val_data, batch_size=args.batch_size_test, shuffle=False, 
            num_workers=args.workers, pin_memory=False, drop_last=False)
        print("*"*60, args.batch_size_train, args.batch_size_test, '*'*61)
        print("*"*60, len(train_loader), len(val_loader), '*'*61)



        # print neural net class
        print('SEG-Torch: {}'.format(datetime.datetime.now()) )
        #print(network)

        # training neural net
        def count_parameters(model):

            return sum(p.numel() for p in model.net.parameters() if p.requires_grad)

        print('N Param: ', count_parameters(network))

        network.fit( train_loader, val_loader, args.epochs, args.snapshot )

        print("Optimization Finished!")
        print("DONE!!!")

        del val_data
        del train_data

    np.random.seed(0)

    test_data = dsxbdata.ISBIDataset(
        args.data, 
        "test", 
        folders_labels=f'labels{num_classes}c',
        count=254,
        num_classes=num_classes,
        num_channels=num_channels,
        transform=get_simple_transforms(pad=pad),
        use_weight=use_weights,
        weight_name=weight,
        load_segments=load_segs,
        shuffle_segments=True,
        use_ori=use_ori
    )
        
    test_loader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, 
        num_workers=args.workers, pin_memory=network.cuda, drop_last=False)
    
    network.evaluate( test_loader, -2, tag='Test')
                   



if __name__ == '__main__':
    main()
