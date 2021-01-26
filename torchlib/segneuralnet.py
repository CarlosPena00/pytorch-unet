

import os
import math
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
from tqdm import tqdm

from . import models as nnmodels
from . import losses as nloss
from . import metrics

from pytvision.neuralnet import NeuralNetAbstract
from pytvision.logger import Logger, AverageFilterMeter, AverageMeter
from pytvision import graphic as gph
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
        half_precision=True
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
        
        self.accuracy = nloss.Accuracy()
        if num_output_channels == 2:
            dice_dim = (1,)
        if num_output_channels == 4:
            dice_dim = (1,2,3)
        
        self.dice = nloss.Dice(dice_dim)
       
        # Set the graphic visualization
        self.logger_train = Logger( 'Train', ['loss'], ['accs', 'dices'], self.plotter  )
        self.logger_val   = Logger( 'Val  ', ['loss'], ['accs', 'dices', 'PQ'], self.plotter )

        self.visheatmap = gph.HeatMapVisdom(env_name=self.nameproject, heatsize=(100,100) )
        self.visimshow = gph.ImageVisdom(env_name=self.nameproject, imsize=(100,100) )
        if self.half_precision:
            self.scaler = torch.cuda.amp.GradScaler()

      
    def training(self, data_loader, epoch=0):        

        #reset logger
        self.logger_train.reset()
        data_time = AverageMeter()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            # measure data loading time
            data_time.update(time.time() - end)
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
                    outputs = self.net(inputs)            
                    loss    = self.criterion(outputs, targets, weights)
                    self.optimizer.zero_grad()            
                    self.scaler.scale(loss*batch_size).backward()  
                    self.scaler.step(self.optimizer) 
                    self.scaler.update()

            else:
                outputs = self.net(inputs)            
                loss    = self.criterion(outputs, targets, weights)            
                self.optimizer.zero_grad()
                (batch_size*loss).backward() #batch_size
                self.optimizer.step()
            
            accs  = self.accuracy(outputs, targets)
            dices = self.dice(outputs, targets)
            #pq    = metrics.pq_metric(outputs.cpu().detach().numpy(), targets.cpu().detach().numpy())
            #pq, n_cells  = metrics.pq_metric(targets, outputs)

            # update
            self.logger_train.update(
                {'loss': loss.item() },
                {'accs': accs.item(), 
                 #'PQ': pq,
                 'dices': dices.item() },      
                batch_size,          
                )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:  
                self.logger_train.logger( epoch, epoch + float(i+1)/len(data_loader), i, len(data_loader), batch_time,   )


    def evaluate(self, data_loader, epoch=0):
        
        # reset loader
        pq_sum      = 0
        total_cells = 0
        self.logger_val.reset()
        batch_time = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
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
                #print(inputs.shape)
                                 
                # fit (forward)
                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.net(inputs)
                        
                else:
                    outputs = self.net(inputs)

                # measure accuracy and record loss
                
                loss  = self.criterion(outputs, targets, weights)   
                accs  = self.accuracy(outputs, targets )
                dices = self.dice( outputs, targets )   
                
                #targets_np = targets[0][1].cpu().numpy().astype(int)
                if epoch == 0:
                    pq      = 0
                    n_cells = 1
                else:
                    #pq, n_cells    = metrics.pq_metric(targets, outputs)
                    if False:#self.skip_background:
                        out_shape = outputs.shape
                        zeros   = torch.zeros((out_shape[0], 1, out_shape[2], out_shape[3])).cuda()
                        outputs = torch.cat([zeros, outputs], 1)

                    
                    all_metrics, n_cells, _ = metrics.get_metrics(targets,outputs)
                    pq = all_metrics['pq']
                    
                pq_sum += pq * n_cells
                total_cells += n_cells
                  
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                #print(loss.item(), accs, dices, batch_size)
                self.logger_val.update( 
                    {'loss': loss.item() },
                    {'accs': accs.item(), 
                     'PQ': (pq_sum/total_cells) if total_cells > 0 else 0,
                     'dices': dices.item() },      
                    batch_size,          
                    )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch, epoch, i,len(data_loader), 
                        batch_time, 
                        bplotter=False,
                        bavg=True, 
                        bsummary=False,
                        )
                

                

        #save validation loss
        if total_cells == 0:
            pq_weight = 0
        else:
            pq_weight = pq_sum / total_cells

        print(f"PQ: {pq_weight:0.4f}, {pq_sum:0.4f}, {total_cells}")

        self.vallosses = self.logger_val.info['loss']['loss'].avg
        acc = self.logger_val.info['metrics']['accs'].avg
        #pq = pq_weight
        

        self.logger_val.logger(
            epoch, epoch, i, len(data_loader), 
            batch_time,
            bplotter=True,
            bavg=True, 
            bsummary=True,
            )

        #vizual_freq
        if epoch % self.view_freq == 0:
            
            prob = F.softmax(outputs.cpu().float(), dim=1)
            prob = prob.data[0]
            maxprob = torch.argmax(prob, 0)
            
            self.visheatmap.show('Label', targets.data.cpu()[0].numpy()[1,:,:] )
            #self.visheatmap.show('Weight map', weights.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Image', inputs.data.cpu()[0].numpy()[0,:,:])
            self.visheatmap.show('Max prob',maxprob.cpu().numpy().astype(np.float32) )
            for k in range(prob.shape[0]):                
                self.visheatmap.show('Heat map {}'.format(k), prob.cpu()[k].numpy() )
        

        print(f"End Val: wPQ{pq_weight}")
        return pq_weight


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





