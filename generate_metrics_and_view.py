#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')

import os
import sys
import numpy as np
import pandas as pd
import csv
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from skimage import io, transform
from skimage import color
import scipy.misc
import scipy.ndimage as ndi
from glob import glob
from pathlib import Path
from pytvision import visualization as view
from pytvision.transforms import transforms as mtrans
from tqdm import tqdm
sys.path.append('../')
from torchlib.datasets import dsxbdata
from torchlib.datasets.dsxbdata import DSXBExDataset, DSXBDataset
from torchlib.datasets import imageutl as imutl
from torchlib import utils
from torchlib.models import unetpad
from torchlib.metrics import get_metrics
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.style.use('fivethirtyeight')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
from pytvision.transforms import transforms as mtrans
from torchlib import metrics

from torchlib.segneuralnet import SegmentationNeuralNet
from torchlib import post_processing_func


DATASET_LABEL = 'DATASET-LABEL'
summary_log   = "extra/summary.csv"
map_post      = post_processing_func.MAP_post()
th_post       = post_processing_func.TH_post()
wts_post      = post_processing_func.WTS_post()

normalize = mtrans.ToMeanNormalization(
    mean = (0.485, 0.456, 0.406),  
    std  = (0.229, 0.224, 0.225), 
    )


softmax          = torch.nn.Softmax(dim=0)

pathdataset      = os.path.expanduser( '/home/chcp/Datasets' )

#namedataset     = 'Seg33_1.0.4'
#namedataset     = 'Seg1009_0.3.2'
namedataset      = 'FluoC2DLMSC_0.1.1'
#namedataset     = 'Bfhsc_1.0.0'

folders_images   = 'images'
folders_contours = 'touchs'
folders_weights  = 'weights'
folders_segment  = 'outputs'
num_classes      = 4
num_channels     = 3
pathname         = pathdataset + '//' + namedataset

use_cuda, gpu_id = 1, 0

models_root      = r'/home/chcp/Documents/Mestrado/MedicalImageSegmentation/'
models_root     += r'Projects/pytorch-unet/out/Fluo/baseline_*/models/*'

model_list =  [Path(url) for url in glob(models_root) 
               if "chk000000.pth.tar" not in url]

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

def get_simple_transforms(pad=0):
    return transforms.Compose([
        #mtrans.CenterCrop( (1008, 1008) ),
        mtrans.ToPad( pad, pad, padding_mode=cv2.BORDER_CONSTANT ),
        mtrans.ToTensor(),
        normalize,      
    ])


def get_flip_transforms(pad=0):
    return transforms.Compose([
        #mtrans.CenterCrop( (1008, 1008) ),
        mtrans.ToRandomTransform( mtrans.VFlip(), prob=0.5 ),
        mtrans.ToRandomTransform( mtrans.HFlip(), prob=0.5 ),
        
        mtrans.ToPad( pad, pad, padding_mode=cv2.BORDER_CONSTANT ),
        mtrans.ToTensor(),
        normalize,      
    ])

def tensor2image(tensor, norm_inverse=True):
    tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if norm_inverse:
            tensor = n(tensor)
    img = tensor.numpy().transpose(1,2,0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

def show(src, titles=[], suptitle="", 
         bwidth=4, bheight=4, save_file=False, show_fig=False,
         show_axis=True, show_cbar=False, last_max=0):

    num_cols = len(src)
    
    fig = plt.figure(figsize=(bwidth * num_cols, bheight))
    plt.suptitle(suptitle)

    for idx in range(num_cols):
        plt.subplot(1, num_cols, idx+1)
        if not show_axis: plt.axis("off")
        if idx < len(titles): plt.title(titles[idx])
        
        if idx == num_cols-1 and last_max:
            plt.imshow(src[idx]*1, vmax=last_max, vmin=0)
        else:
            plt.imshow(src[idx]*1)
        if type(show_cbar) is bool:
            if show_cbar: plt.colorbar()
        elif idx < len(show_cbar) and show_cbar[idx]:
            plt.colorbar()
        
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
        
    if not show_fig:
        plt.close(fig)
        
def show2(src, titles=[], suptitle="", 
         bwidth=4, bheight=4, save_file=False, show_fig=False,
         show_axis=True, show_cbar=False, last_max=0):

    num_cols = len(src)//2
    fig = plt.figure(figsize=(bwidth * num_cols, bheight*2))
    plt.suptitle(suptitle)

    for idx in range(num_cols*2):
        plt.subplot(2, num_cols, idx+1)
        if not show_axis: plt.axis("off")
        if idx < len(titles): plt.title(titles[idx])
        
        if idx == num_cols-1 and last_max:
            plt.imshow(src[idx]*1, vmax=last_max, vmin=0)
        else:
            plt.imshow(src[idx]*1)
        if type(show_cbar) is bool:
            if show_cbar: plt.colorbar()
        elif idx < len(show_cbar) and show_cbar[idx]:
            plt.colorbar()
        
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
        
    if not show_fig:
        plt.close(fig)




def load_model(full_url, use_cuda=False, gpu_id=0):
    full_path     = str(full_url)
    splits        = full_path.split(r"/")
    patchproject  = r'/'.join(splits[:10])
    ckpt_path     = '/'.join(splits[11:])
    exp_type, nameproject, _, file_name = splits[-4:]
    
    net = SegmentationNeuralNet(
        patchproject=patchproject, 
        nameproject=nameproject, 
        no_cuda=not use_cuda, parallel=False, seed=2021, 
        print_freq=False, gpu=gpu_id
    )

    try:

        if net.load( full_path ) is not True:
            print("*"*30)
            print("Not Found Warring: ",full_path)
            print("*"*30)
            
            return False, None, None
        

    except Exception as e:
        print("LOAD Error: ", e)
        print("*"*30)
        return False, None, None
    
    save_path = fr'extra/outputs/{DATASET_LABEL}/' + '_'.join(np.array(splits)[[10, 12]]) + r'/'
    
    if use_cuda:
        net.net.cuda(gpu_id)
    net.net.eval()
    
    return True, net, save_path

def load_data(pathname, subset, use_cuda=False):
    data = dsxbdata.ISBIDataset(
        pathname, subset, 
        folders_labels=f'labels{num_classes}c',
        count=None, num_classes=num_classes,
        num_channels=num_channels,
        transform=get_simple_transforms(pad=0),
        use_weight=False, weight_name='',
        load_segments=False, shuffle_segments=True,
        use_ori=1
    )

    data_loader = DataLoader(data, batch_size=1, shuffle=False, 
        num_workers=0, pin_memory=use_cuda, drop_last=False)
    
    return data_loader

def forward(net, sample, use_cuda=False, gpu_id=0, post_label='map'):
    inputs, labels = sample['image'], sample['label']
    
    if use_cuda:
        inputs = inputs.cuda(gpu_id)
        
    outputs     = net(inputs).cpu()
    amax        = outputs[0].argmax(0)
    view_inputs = tensor2image(inputs[0, :3])
    view_labels = labels[0].argmax(0)
    prob        = outputs[0] / outputs[0].sum(0)
    
    return  labels, outputs, amax, view_inputs, view_labels, prob

def update_metrics(results, wpq, wsq, wrq, pcells, total_cells):

    wpq += results['pq'] * n_cells
    wsq += results['sq'] * n_cells
    wrq += results['rq'] * n_cells
    pcells += results['n_cells']
    total_cells += n_cells

    return wpq, wsq, wrq, pcells, total_cells

def show_cells(results, n_cells, post_label,
               v_inputs, v_labels, amax, predictionlb, prob,
               save_path, namedataset, subset, idx, save_out=True, show_fig=False):
    
    res_str = f"Post {post_label} | Nreal {n_cells} | Npred {results['n_cells']} | PQ {results['pq']:0.2f} " +             f"| SQ {results['sq']:0.2f} | RQ {results['rq']:0.2f}"

    show2([v_inputs, v_labels, amax, predictionlb, prob[0], prob[1], prob[2], prob[3]], 
         show_axis=False, suptitle=res_str, show_fig=show_fig,
         show_cbar=[False, False, False, False, True, True, True, True], 
         save_file=f"{save_path}/{namedataset}_{subset}_{post_label}_{idx:03d}.jpg",
         titles=['Original', 'Label', 'MAP', 'Cells', 'Prob 0', 'Prob 1', 'Prob 2', 'Prob 3'], bheight=4.5)
    
    cv2.imwrite(f"{save_path}/{namedataset}_{subset}_{post_label}_{idx:03d}_post.png", predictionlb)


def write_logger(namedataset, subset, post_label, num_images, model_url_base, 
                 wpq, wsq, wrq, pcells, total_cells, summary_log="extra/summary.csv"):
    
    row = [namedataset, subset, model_url_base, post_label, wpq/total_cells, 
           wsq/total_cells, wrq/total_cells, pcells, total_cells, num_images]
    row = list(map(str, row))
    header = ["dataset", 'subset', 'model', 'post', 'WPQ', 'WSQ', "WRQ", "PCells", "Cells", 'Images']
        
    write_header = not Path(summary_log).exists()
    with open(summary_log, 'a') as f:
        if write_header:
            f.writelines(','.join(header)+'\n')
        f.writelines(','.join(row)+'\n')
        


# # Erro analysis


def print_logger(namedataset, subset, post_label, num_images, model_url_base, 
                 wpq, wsq, wrq, pcells, total_cells, extra_str):
    
    row = [extra_str, wpq/total_cells, wsq/total_cells, wrq/total_cells, pcells, total_cells, num_images]
    
    print(row)


df = pd.read_csv(summary_log)
df = df[['dataset', 'subset', 'model', 'post']]

th_thresh=0.5
thresh_background=0.45
thresh_foreground=0.20

for model_url_base in tqdm(model_list):
    err, net, save_path = False, None, None
    exit_model = False

    for subset in ['test', 'val', 'train']:
        
        data_loader      = load_data(pathname, subset)
        num_images       = len(data_loader)
        
        if exit_model: break
        
        for post_label in ['th', 'wts', 'map']:
            
            has_info = ((df.dataset==namedataset) & (df.subset==subset) & 
                        (df.model==str(model_url_base)) & (df.post==post_label)).any()
            if has_info: continue
                
            if net is None:
                err, net, save_path = load_model(model_url_base, use_cuda, gpu_id)
                
                if not err: 
                    exit_model = True
                    break

                save_path = save_path.replace(DATASET_LABEL, namedataset)
                Path(save_path).mkdir(exist_ok=True, parents=True)

            wpq, wsq, wrq, pcells, total_cells = 0, 0, 0, 0, 0

            for idx, sample in enumerate(data_loader):

                labels, outputs, amax, v_inputs, v_labels, prob = forward(net, sample, use_cuda, gpu_id, post_label)
                results, n_cells, preds = get_metrics(labels, outputs, post_label=post_label, morph=2, morph_label=2,
                                                      th_thresh=0.5, thresh_background=0.45, thresh_foreground=0.20)
                predictionlb, prediction, region, _ = preds
                wpq, wsq, wrq, pcells, total_cells = update_metrics(results, wpq, wsq, wrq, pcells, total_cells)

                show_cells(results, n_cells, post_label,
                   v_inputs, v_labels, amax, predictionlb, prob,
                   save_path, namedataset, subset, idx)

            write_logger(namedataset, subset, post_label, num_images, model_url_base, 
                         wpq, wsq, wrq, pcells, total_cells, summary_log=summary_log)


