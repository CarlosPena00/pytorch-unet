

import os
import math
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import time
from tqdm import tqdm

from . import models as nnmodels
from . import losses as nloss
from . import metrics
from . import utils

from pytvision.neuralnet import NeuralNetAbstract
#from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
#from pytvision import graphic as gph
from pytvision import netlearningrate
from pytvision import utils as pytutils

#----------------------------------------------------------------------------------------------
# Neural Net for Segmentation


class SegmentationNeuralNet(NeuralNetAbstract):
    """
    Segmentation Neural Net 
    """

    def __init__(self,
        patchproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        view_freq=1,
        half_precision=False
        ):
        """
        Initialization
            -patchproject (str): path project
            -nameproject (str):  name project
            -no_cuda (bool): system cuda (default is True)
            -parallel (bool)
            -seed (int)
            -print_freq (int)
            -gpu (int)
            -view_freq (in epochs)
        """

        super(SegmentationNeuralNet, self).__init__( patchproject, nameproject, no_cuda, parallel, seed, print_freq, gpu  )
        self.view_freq = view_freq
        self.half_precision = half_precision

 
    def create(self, 
        arch, 
        num_output_channels, 
        num_input_channels,  
        loss, 
        lr, 
        optimizer, 
        lrsch,    
        momentum=0.9,
        weight_decay=5e-4,      
        pretrained=False,
        size_input=388,
        cascade_type='none',
        writer=None,
        ):
        """
        Create
        Args:
            -arch (string): architecture
            -num_output_channels, 
            -num_input_channels, 
            -loss (string):
            -lr (float): learning rate
            -optimizer (string) : 
            -lrsch (string): scheduler learning rate
            -pretrained (bool)
            
        """
        
        cfg_opt={ 'momentum':momentum, 'weight_decay':weight_decay } 
        cfg_scheduler={ 'step_size':100, 'gamma':0.1  }
        
        super(SegmentationNeuralNet, self).create( 
            arch, 
            num_output_channels, 
            num_input_channels, 
            loss, 
            lr, 
            optimizer, 
            lrsch, 
            pretrained,
            cfg_opt=cfg_opt, 
            cfg_scheduler=cfg_scheduler
        )
        self.size_input = size_input
        self.num_output_channels = num_output_channels
        self.cascade_type = cascade_type
        
        if self.cascade_type == 'none':
            self.step = self.default_step
        elif self.cascade_type == 'ransac':
            self.step = self.ransac_step
        elif self.cascade_type == 'simple':
            self.step = self.cascate_step
        else:
            raise "Cascada not found"
        
        if num_output_channels == 2:
            dice_dim = (1,)
        if num_output_channels == 4:
            dice_dim = (1,2,3)
        
        self.writer = writer
        self.norm_inv = utils.NormalizeInverse()
        
        #self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(256,256) )
        #self.visimshow = gph.ImageVisdom(env_name=self.nameproject, imsize=(256,256) )
        
        if self.half_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            
    def write_data(self, tag, metrics_mean, epoch):
        
        self.writer.add_scalar(f'Loss/{tag}',  metrics_mean['loss'], epoch)
        #self.writer.add_scalar(f'Accs/{tag}',  metrics_mean['Accs'], epoch)
        if tag == 'Val' or tag == 'Test':
            self.writer.add_scalar(f'PQ/{tag}', metrics_mean['pq'], epoch)
            self.writer.add_scalar(f'SQ/{tag}', metrics_mean['sq'], epoch)
            self.writer.add_scalar(f'RQ/{tag}', metrics_mean['rq'], epoch)
            
    def get_mean_metrics(self, metrics_sum):
        metrics_mean = {}
        metrics_mean['loss'] = np.mean(metrics_sum['loss'])
        metrics_mean['pq'] = metrics_sum['pq'] / metrics_sum['n_cells']
        metrics_mean['sq'] = metrics_sum['sq'] / metrics_sum['n_cells']
        metrics_mean['rq'] = metrics_sum['rq'] / metrics_sum['n_cells']
        
        return metrics_mean

    def show_data(self, tag, inputs, targets, outputs, epoch):
        grid   = torchvision.utils.make_grid(inputs)
        grid   = self.norm_inv(grid.cpu().detach())
        self.writer.add_image(f'{tag}/Inputs', grid, epoch)

        grid   = torchvision.utils.make_grid(targets.argmax(1, keepdim=True))
        self.writer.add_image(f'{tag}/Targets', grid, epoch)

        grid   = torchvision.utils.make_grid(outputs.argmax(1, keepdim=True))
        self.writer.add_image(f'{tag}/Map', grid, epoch)
        prob = F.softmax(outputs.detach().cpu().float(), dim=1)
        
        for ch in range(prob.shape[1]):
            fig = utils.get_fig_wcolor(prob[0,ch])
            self.writer.add_figure(f'{tag}/Prob{ch}', fig, epoch)
            
    def ransac_step(self, inputs, targets, max_deep=3, segs_per_forward=17, src_c=3, verbose=False):
        srcs = inputs[:, :src_c]
        segs = inputs[:, src_c:]
        lv_segs = segs#.clone()
        

        first = True
        final_loss = 0.0
        for lv in range(max_deep):
            n_segs = segs.shape[1]
            new_segs = []
            actual_c = segs_per_forward ** (max_deep - lv)
            if verbose: print(segs.shape, actual_c)
            actual_seg_ids = np.random.choice(range(n_segs), size=actual_c)
            step_segs = segs[:, actual_seg_ids]
            
            for idx in range(0, actual_c, segs_per_forward):
                mini_inp = torch.cat((srcs, step_segs[:, idx:idx+segs_per_forward]), dim=1)
                mini_out = self.net(mini_inp)
                ## calculate loss
                final_loss += self.criterion(mini_out, targets) * 1
                new_segs.append(mini_out.argmax(1, keepdim=True))
                

                if verbose: print(mini_inp.shape, idx, idx+segs_per_forward, actual_loss.item())
            
            segs = torch.cat(new_segs, dim=1)
            
        return final_loss, mini_out


    def cascate_step(self, inputs, targets, segs_per_forward=17, src_c=3, verbose=False):

        srcs = inputs[:, :src_c]
        segs = inputs[:, src_c:]
        lv_segs = segs.clone()

        final_loss = 0.0
        n_segs = lv_segs.shape[1]
        actual_seg_ids = np.random.choice(range(n_segs), size=n_segs, replace=False)
        lv_segs = lv_segs[:, actual_seg_ids]

        while n_segs > 1:

            if verbose: print(n_segs)

            inputs_seg = lv_segs[:, :segs_per_forward]
            inputs_seg_ids = np.random.choice(range(inputs_seg.shape[1]), 
                                              size=segs_per_forward, replace=inputs_seg.shape[1]<segs_per_forward)
            inputs_seg = inputs_seg[:, inputs_seg_ids]

            mini_inp = torch.cat((srcs, inputs_seg), dim=1)
            mini_out = self.net(mini_inp)
            ## calculate loss
            final_loss += self.criterion(mini_out, targets)

            if verbose: print(mini_inp.shape, segs_per_forward, actual_loss.item())
            lv_segs = torch.cat((lv_segs[:, segs_per_forward:], mini_out.argmax(1, keepdim=True)), dim=1)
            n_segs = lv_segs.shape[1]
        return final_loss, mini_out
    
    def default_step(self, inputs, targets):
        outputs = self.net(inputs)            
        loss    = self.criterion(outputs, targets)
        return loss, outputs

      
    def training(self, data_loader, epoch=0, tag='Train'):        
        #reset logger
        metrics_sum = {'loss':[]}
        
        # switch to evaluate mode
        self.net.train()

        start = time.time()
        for i, sample in enumerate(data_loader):
            
            # get data (image, label, weight)
            inputs, targets = sample['image'], sample['label']
            weights = None
            if 'weight' in sample.keys():
                weights = sample['weight']
                
            batch_size = inputs.shape[0]

            if self.cuda:
                inputs  = inputs.cuda() 
                targets = targets.cuda() 
                if type(weights) is not type(None):
                    weights = weights.cuda()
                
            # fit (forward)
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    
                    loss, outputs = self.step(inputs, targets)
                    
                    self.optimizer.zero_grad()            
                    self.scaler.scale(loss*batch_size).backward()  
                    self.scaler.step(self.optimizer) 
                    self.scaler.update()

            else:
                loss, outputs = self.step(inputs, targets)
                    
                self.optimizer.zero_grad()
                (batch_size*loss).backward() #batch_size
                self.optimizer.step()
            
            #pq    = metrics.pq_metric(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
            #pq, n_cells  = metrics.pq_metric(targets, outputs)
            metrics_sum['loss'].append(loss.item())

            # measure elapsed time
        mean_metrics = {'loss':np.mean(metrics_sum['loss'])}
        self.write_data(tag, mean_metrics, epoch)
        self.show_data(tag, inputs, targets, outputs, epoch)
        end = time.time()
        print(f"{tag:6} | {epoch:03d} | {mean_metrics['loss']:0.4f} | ------ | ", end='')
        print(f"------ | ------ | ------ | { (end - start ):0.4f} |",flush=True)
        
    def evaluate(self, data_loader, epoch=0, tag='Val'):
        
        metrics_sum = {'pq':0, 'sq':0, 'rq':0, 'f1':0, 'loss':[] ,'n_cells':0}
        total_cells = 0
        

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            start = time.time()
            
            for i, sample in enumerate(data_loader):
                # get data (image, label)
                inputs, targets = sample['image'], sample['label']
                weights = None
                if 'weight' in sample.keys():
                    weights = sample['weight']
                #inputs, targets = sample['image'], sample['label']
                               
                batch_size = inputs.shape[0]

                #print(inputs.shape)

                if self.cuda:
                    inputs  = inputs.cuda()
                    targets = targets.cuda()
                    if type(weights) is not type(None):
                        weights = weights.cuda()

                # fit (forward)
                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        loss, outputs = self.step(inputs, targets)   
                else:
                    loss, outputs = self.step(inputs, targets)
                
                # measure accuracy and record loss
                metrics_sum['loss'].append(loss.item())
                
                if epoch == 0:
                    metrics_sum['pq'] = 0
                    metrics_sum['sq'] = 0
                    metrics_sum['rq'] = 0
                    metrics_sum['n_cells'] = 1
                else:
                    #pq, n_cells    = metrics.pq_metric(targets, outputs)
                    if False:#self.skip_background:
                        out_shape = outputs.shape
                        zeros   = torch.zeros((out_shape[0], 1, out_shape[2], out_shape[3])).cuda()
                        outputs = torch.cat([zeros, outputs], 1)

                    
                    all_metrics, n_cells, _ = metrics.get_metrics(targets,outputs)
                    
                    metrics_sum['pq'] += all_metrics['pq'] * n_cells
                    metrics_sum['sq'] += all_metrics['sq'] * n_cells
                    metrics_sum['rq'] += all_metrics['rq'] * n_cells
                    metrics_sum['n_cells'] += n_cells
                  
                # measure elapsed time
                
   
        #save validation loss
        mean_metrics = self.get_mean_metrics(metrics_sum)
        self.write_data(tag,mean_metrics, epoch)
        self.show_data(tag, inputs, targets, outputs, epoch)
        
        end = time.time()
        if epoch % 10 == 0:
            pass
        print(f"Tag    | Epo |  Loss  |   PQ   |   SQ   |   RQ   |  Time  |")
        print(f"{tag:6} | {epoch:03d} | {mean_metrics['loss']:0.4f} | ", end='')
        print(f"{mean_metrics['pq']:0.4f} | {mean_metrics['sq']:0.4f} | {mean_metrics['rq']:0.4f} | { (end - start ):0.4f} |")
        

        return mean_metrics['pq']


    def test(self, data_loader ):
        
        masks = []
        ids   = []
        k=0

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, sample in enumerate( tqdm(data_loader) ):
                
                # get data (image, label)
                inputs, meta  = sample['image'], sample['metadata']    
                idd = meta[:,0]         
                x = inputs.cuda() if self.cuda else inputs    
                
                # fit (forward)
                yhat = self.net(x)
                yhat = F.softmax(yhat, dim=1)    
                yhat = pytutils.to_np(yhat)

                masks.append(yhat)
                ids.append(idd)       
                
        return ids, masks

    
    def __call__(self, image ):        
        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            x = image.cuda() if self.cuda else image    
            yhat = self.net(x)
            yhat = F.softmax( yhat, dim=1 )
            #yhat = pytutils.to_np(yhat).transpose(2,3,1,0)[...,0]
        return yhat


    def _create_model(self, arch, num_output_channels, num_input_channels, pretrained ):
        """
        Create model
            -arch (string): select architecture
            -num_classes (int)
            -num_channels (int)
            -pretrained (bool)
        """    

        self.net = None    

        #-------------------------------------------------------------------------------------------- 
        # select architecture
        #--------------------------------------------------------------------------------------------
        #kw = {'num_classes': num_output_channels, 'num_channels': num_input_channels, 'pretrained': pretrained}

        kw = {'num_classes': num_output_channels, 'in_channels': num_input_channels, 'pretrained': pretrained}
        self.net = nnmodels.__dict__[arch](**kw)
        
        self.s_arch = arch
        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels

        if self.cuda == True:
            self.net.cuda()
        if self.parallel == True and self.cuda == True:
            self.net = nn.DataParallel(self.net, device_ids= range( torch.cuda.device_count() ))

    def _create_loss(self, loss):

        # create loss
        if loss == 'wmce': # Not tested
            self.criterion = nloss.WeightedMCEloss()
        elif loss == 'bdice': # Fail
            self.criterion = nloss.BDiceLoss()
        elif loss == 'wbdice': # Fail
            self.criterion = nloss.WeightedBDiceLoss()
        elif loss == 'wmcedice': # Fail
            self.criterion = nloss.WeightedMCEDiceLoss()
        elif loss == 'mcedice': # Fail
            self.criterion = nloss.MCEDiceLoss()  
        elif loss == 'bce': # Pass
            self.criterion = nloss.BCELoss()
        elif loss == 'bce2c': # Pass
            self.criterion = nloss.BCELoss2c()
        elif loss == 'mce': # Pass
            self.criterion = nloss.MCELoss()
        elif loss == 'wbce':
            self.criterion = nloss.WeightedBCELoss()
        elif loss == 'wce': # Pass
            self.criterion = nloss.WeightedCrossEntropyLoss()
        elif loss == 'wfocalce': # Pass
            self.criterion = nloss.WeightedCEFocalLoss()
        elif loss == 'focaldice': # Pass
            self.criterion = nloss.FocalDiceLoss()
        elif loss == 'wfocaldice': # Pass
            self.criterion = nloss.WeightedFocalDiceLoss()
        elif loss == 'dice':# FAIL
            self.criterion = nloss.DiceLoss()
        elif loss == 'msedice':# FAIL
            self.criterion = nloss.MSEDICELoss()
        elif loss == 'mcc': # FAIL
            self.criterion = nloss.MCCLoss()
        elif loss == 'mdice': # FAIL
            self.criterion = nloss.MDiceLoss()
        elif loss == 'wcefd':
            self.criterion = nloss.WeightedCEFocalDice()
        elif loss == 'jreg':
            if self.num_output_channels == 2:
                lambda_dict={'0':{'0':  '1', '1':'0.5'},
                             '1':{'0':'0.5', '1':  '1'}}
            if self.num_output_channels == 4:
                lambda_dict={'0':{'0':  '1', '1':'0.5', '2':'0.5', '3':'0.5'},
                             '1':{'0':'0.5', '1':  '1', '2':'0.5', '3':'0.5'},
                             '2':{'0':'0.5', '1':'0.5', '2':'1'  , '3':'0.5'},
                             '3':{'0':'0.5', '1':'0.5', '2':'0.5', '3':  '1'}}
            
            self.criterion = nloss.WCE_J_SIMPL(lambda_dict=lambda_dict)
        else:
            assert(False)

        self.s_loss = loss





