
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import filterfalse as ifilterfalse
import kornia

class WeightedMCEloss(nn.Module):

    def __init__(self ):
        super(WeightedMCEloss, self).__init__()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h )
        weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)
        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        return loss

    

def getweightmap(label):
    lshape = label.shape
    mask = torch.zeros((lshape[0],lshape[2],lshape[3]), dtype=torch.uint8)#.to(label.device)
    mask[label[:, 0]==1] = 0
    mask[label[:, 1]==1] = 1
    mask[label[:, 2]==1] = 2
    w_c = torch.empty(mask.shape)#.to(label.device)
    classes = lshape[1]
    frecs = []
    for i in range(classes):frecs.append( ( torch.sum(mask == i).float() / (lshape[-2]*lshape[-1])))
                                 
    # Calculate
    for i in range( classes ): w_c[mask == i] = 1 / (classes*frecs[i])
    
    return w_c

class SimpleCrossEntropyLossnn(nn.Module):
    def __init__(self):
        super(SimpleCrossEntropyLossnn, self).__init__()
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, y_pred, y_true, weight=True):
        #breakpoint()
        if weight and False:
            pos_weight = getweightmap(y_true)
            loss = self.loss(y_pred, y_true).cpu()
            loss = (loss * pos_weight).mean()
            return loss.cuda()
        
        if weight or True:
            loss = self.loss(y_pred, y_true)
            loss[:, 0] *= 0.445
            loss[:, 1] *= 1.379
            loss[:, 2] *= 1438.257
            return loss.mean()
            
        return self.loss(y_pred, y_true)



class WeightedMCEFocalloss(nn.Module):
    # Fail
    
    def __init__(self, gamma=2.0 ):
        super(WeightedMCEFocalloss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight=None):
        if type(weight) == type(None):
            weight = torch.ones(y_true.shape)
        if type(weight) == type(None) and False:
            bg = y_true[:, 0].sum()
            fg = y_true[:, 1].sum()
            th = y_true[:, 2].sum()
            total = bg + fg + th
            bg_w = torch.log(total/bg)
            fg_w = torch.log(total/fg)
            if th == 0:
                th = total
                th_w = 1
            else:
                th_w = torch.log(total/th)
            weight = y_pred.argmax(dim=1)
            weight[weight==0] = bg_w
            weight[weight==1] = fg_w
            weight[weight==2] = th_w
            

        #n, ch, h, w = y_pred.size()
        #y_true = centercrop(y_true, w, h )
        #weight = centercrop(weight, w, h )
        
        y_pred_log =  F.log_softmax(y_pred, dim=1)

        fweight = (1 - F.softmax(y_pred, dim=1) ) ** self.gamma
        
        if fweight.is_cuda:
            weight = weight.cuda().float()
        weight  = weight*fweight

        logpy = torch.sum( weight * y_pred_log * y_true, dim=1 )
        #loss  = -torch.sum(logpy) / torch.sum(weight)
        loss  = -torch.mean(logpy)
        
        return loss



class WBCELoss(nn.Module):
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, weights ):        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        loss_0 = self.bce(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce(y_pred[:, 1], y_true[:, 1])
        loss_2 = self.bce(y_pred[:, 2], y_true[:, 2])
        loss = loss_0 * 0.2 + loss_1 * 0.5 + loss_2 * 0.3
        return loss

class WeightedBDiceLoss(nn.Module):
    
    def __init__(self ):
        super(WeightedBDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        weight = centercrop(weight, w, h )
        y_pred = self.sigmoid(y_pred)
        smooth = 1.
        w, m1, m2 = weight, y_true, y_pred
        score = (2. * torch.sum(w * m1 * m2) + smooth) / (torch.sum(w * m1) + torch.sum(w * m2) + smooth)
        loss = 1. - torch.sum(score)
        return loss


class BDiceLoss(nn.Module):
    
    def __init__(self):
        super(BDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)

        smooth = 1.
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        score = (2. * torch.sum(y_true_f * y_pred_f) + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return 1. - score

    
class BLogDiceLoss(nn.Module):
    
    def __init__(self, classe = 1 ):
        super(BLogDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classe = classe

    def forward(self, y_pred, y_true, weight=None ):
        
        n, ch, h, w = y_pred.size()
        y_true = centercrop(y_true, w, h)
        y_pred = self.sigmoid(y_pred)
        if y_true.max() <= 0:
            return 0

        eps = 1e-14
        dice_target = (y_true[:,self.classe,...] == 1).float()
        dice_output = y_pred[:,self.classe,...]
        intersection = (dice_output * dice_target).sum()
        union = dice_output.sum() + dice_target.sum() + eps
        if intersection < 0 :
            print("Error: inter < 0: ", intersection)
            breakpoint()
        if intersection > union:
            print("Union < inter")
            breakpoint()
        blogdiceloss = -torch.log(2 * intersection / union) 
        if torch.isnan(blogdiceloss).any():
            breakpoint()
            maybeLoss = torch.log_softmax(2 * intersection / union)
            
        return blogdiceloss

class WeightedMCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0  ):
        super(WeightedMCEDiceLoss, self).__init__()
        self.loss_mce = WeightedMCEFocalloss()
        self.loss_dice = BLogDiceLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight ):        
        
        alpha = self.alpha
        weight = torch.pow(weight,self.gamma)
        loss_dice = self.loss_dice(y_pred, y_true)
        loss_mce = self.loss_mce(y_pred, y_true, weight)
        loss = loss_mce + alpha*loss_dice        
        return loss

class MCEDiceLoss2(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.5  ):
        super(MCEDiceLoss, self).__init__()
        self.loss_mce = BCELoss()
        self.loss_dice_fg = BLogDiceLoss( classe=1  )
        self.loss_dice_th = BLogDiceLoss( classe=2  )
        
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight=None ):        
        
        alpha = self.alpha  
        gamma = self.gamma

        # bce(all_channels) +  dice_loss(mask_channel) + dice_loss(border_channel)  
        loss_all  = self.loss_mce( y_pred[:,:2,...], y_true[:,:2,...]) 
        loss_fg   = self.loss_dice_fg( y_pred, y_true )         
        loss_th   = self.loss_dice_th( y_pred, y_true )  if y_true[:,2,... ].sum() > 0 else 0 

        #print(y_pred[0, 0], y_true[0,0])
        #print(loss_all, loss_fg, loss_th) 
        #print('*'*60)
        #print(y_pred.shape, y_pred.max()) 
        #print(y_true.shape, y_true.max()) 
        #print('*'*60)
        
        
        loss      = loss_all + alpha*loss_fg + gamma*loss_th     
        if torch.isnan(loss).any():
            print(f"Loss_all: {loss_all} :: loss_fg: {loss_fg} :: loss_th: {loss_th}")
            print(y_pred.max(), y_true.max())
            breakpoint()
        return loss

class MCEDiceLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.5  ):
        super(MCEDiceLoss, self).__init__()
        self.loss_mce = BCELoss()
        self.loss_dice_fg = BLogDiceLoss( classe=1  )
        self.loss_dice_th = BLogDiceLoss( classe=2  )
        self.loss_dice_gp = BLogDiceLoss( classe=3  )
        
        
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, weight=None ):  

        
        alpha = self.alpha  
        gamma = self.gamma

        # bce(all_channels) +  dice_loss(mask_channel) + dice_loss(border_channel)  
        loss_all  = self.loss_mce( y_pred[:,:2,...], y_true[:,:2,...]).clamp(0,10) 
        loss_fg   = self.loss_dice_fg( y_pred, y_true ).clamp(0,0.1) 
        loss_th   = self.loss_dice_th( y_pred, y_true ).clamp(0,0.1) 
        loss_gp   = self.loss_dice_gp( y_pred, y_true ).clamp(0,0.1) 
        
        loss      = loss_all + alpha*loss_fg + gamma*loss_th + loss_gp*gamma
        #print(f"Loss: {loss_all} ; loss_fg: {loss_fg} ; loss_th: {loss_th}")
        return loss





def to_one_hot(mask, size):    
    n, c, h, w = size
    ymask = torch.FloatTensor(size).zero_()
    new_mask = torch.LongTensor(n,1,h,w)
    if mask.is_cuda:
        ymask = ymask.cuda(mask.get_device())
        new_mask = new_mask.cuda(target.get_device())
    new_mask[:,0,:,:] = torch.clamp(mask.data, 0, c-1)
    ymask.scatter_(1, new_mask , 1.0)    
    return Variable(ymask)

def centercrop(image, w, h):        
    nt, ct, ht, wt = image.size()
    padw, padh = (wt-w) // 2 ,(ht-h) // 2
    if padw>0 and padh>0: image = image[:,:, padh:-padh, padw:-padw]
    return image

def flatten(x):        
    x_flat = x.clone()
    x_flat = x_flat.view(x.shape[0], -1)
    return x_flat


#####################################################################
############################## Checked ##############################
#####################################################################


class WCE_J_SIMPL(nn.Module):

    def __init__(self,lambda_dict=None, power=1,lambda_vect=None):
        super(WCE_J_SIMPL, self).__init__()
        self.power = power
        self.lambda_mat=dict()
        
        if lambda_dict!=None and lambda_vect!=None:
            print('Please provide only one lambda parameter')
            raise
        
        if lambda_dict==None and lambda_vect==None:
            self.lambda_mat=None
        
        if lambda_dict!=None:
            c=len(list(lambda_dict.keys()))
            self.lambda_mat=self.ones_dict(c,c)
            for key1, row in lambda_dict.items():
                if int(key1) not in list(self.lambda_mat.keys()):
                    self.lambda_mat[int(key1)]=dict()
                for key2, value in row.items():
                    self.lambda_mat[int(key1)][int(key2)]= float(value)
        
        elif lambda_vect!=None:
            llv=len(lambda_vect)
            if np.round(np.sqrt(llv))**2 == llv:
                c=np.round(np.sqrt(llv)).astype(int)
                self.lambda_mat=self.ones_dict(c,c)
                k=0
                for i in range(c):
                    for j in range(c):
                        self.lambda_mat[i][j]=lambda_vect[k]
                        k+=1
            else:
                c=-1
                for i in range(1000):
                    if ((2*llv)/(i+1))==i:
                        c=i
                        break
                if c==-1:
                    print('Please provide a valid lambda vector')
                    raise
                
                self.lambda_mat=self.ones_dict(c,c)
                k=0
                for i in range(c):
                    for j in range(i,c):
                        self.lambda_mat[i][j]=lambda_vect[k]
                        self.lambda_mat[j][i]=lambda_vect[k]
                        k+=1

        if self.lambda_mat!=None:
            print('Using lambda matrix:')
            for key1, row in self.lambda_mat.items():
                for key2, value in row.items():
                    print('{0:.4f}'.format(self.lambda_mat[int(key1)][int(key2)]),end=" ")
                print('')
        
      
    def ones_dict(self,d1,d2):
        lambda_mat=dict()
        for i in range(d1):
            if i not in list(lambda_mat.keys()):
                    lambda_mat[i]=dict()
            for j in range(d2):
                lambda_mat[i][j]=float(1)
        return lambda_mat

    def zf_dict(self,d1,d2):
        lambda_mat=dict()
        for i in range(d1):
            if i not in list(lambda_mat.keys()):
                    lambda_mat[i]=dict()
            for j in range(d2):
                lambda_mat[i][j]=float(0.5)
        return lambda_mat
    
    def crop(self,w,h,target):
        nt,ht,wt= target.size()
        offset_w,offset_h=(wt-w) // 2 ,(ht-h) // 2
        if offset_w>0 and offset_h>0:
            target=target[:,offset_h:-offset_h,offset_w:-offset_w].clone()
        
        return target

    def to_one_hot(self,target,size):
        n, c, h, w = size
        
        ymask = torch.FloatTensor(size).zero_()
        new_target=torch.LongTensor(n,1,h,w)
        if target.is_cuda:
            ymask=ymask.cuda(target.get_device())
            new_target=new_target.cuda(target.get_device())

        new_target[:,0,:,:]=torch.clamp(target.detach(),0,c-1)
        ymask.scatter_(1, new_target , 1.0)
        
        return (ymask.requires_grad_())


    def forward(self, input, target,weight=None):
        target = target.argmax(1)
        eps=0.00000001

        n, c, h, w = input.size()
        log_p = F.log_softmax(input,dim=1)
        
        if self.lambda_mat==None:
            lamb=self.zf_dict(c,c)
        else:
            assert len(list(self.lambda_mat.keys()))==c, 'Lambda is expected to have dimension {}x{} but got {}x{} instead'.format(c,c,len(list(self.lambda_mat.keys())),len(list(self.lambda_mat.keys())))
            lamb=self.lambda_mat

        #target=self.crop(w,h,target)
        ymask=self.to_one_hot(target,log_p.size())
        ymask1=ymask.clone()

        # mcc
        p = F.softmax(input,dim=1)
        lossd=0

        for i in range(c):
            for j in range(c):
                if i==j:
                    continue
                iflat = p[:, i].contiguous().view(-1)
                tflat = ymask[:, i].contiguous().view(-1)
                tflat2 = ymask[:, j].contiguous().view(-1)
                ni=tflat.sum()
                nj=tflat2.sum()
                if ni.item()>0 and nj.item()>0:
                    mcc = (iflat*tflat/ni -iflat*tflat2/nj)
                    lossd+= torch.log( (0.5*(mcc.sum())+0.5)**self.power  + eps ) * (-lamb[i][j])

        # weighted cross entropy
        if weight is not None:
            #weight=self.crop(w,h,weight)
            for classes in range(c):
                ymask1[:,classes,:,:]=  ymask1[:,classes,:,:].clone() * lamb[classes][classes] * (weight)

        logpy = (log_p * ymask1).sum(1)
        loss = -(logpy).mean()


        return loss + lossd
    
class Accuracy(nn.Module):
    # Check
    def __init__(self):
        super(Accuracy, self).__init__()
    
    def forward(self, input, target):
        
        input_a = input.argmax(1)
        target_a = target.argmax(1)
        return (input_a[(input_a>0)|(target_a>0)] == target_a[(input_a>0)|(target_a>0)]).float().mean()
    
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()    
        self.focal_loss = kornia.losses.FocalLoss(0.5, reduction='mean')
    
    def forward(self, y_pred, y_true, weights=None):
        return self.focal_loss(y_pred, y_true.argmax(1))

class WeightedFocalLoss(nn.Module):
    def __init__(self):
        super(WeightedFocalLoss, self).__init__()    
        self.focal_loss = kornia.losses.FocalLoss(0.5, reduction='none')
    
    def forward(self, y_pred, y_true, weights=None):
        return (self.focal_loss(y_pred, y_true.argmax(1)) * weights).mean()

class FocalDiceLoss(nn.Module):
    def __init__(self):
        super(FocalDiceLoss, self).__init__()    
        self.focal_loss = kornia.losses.FocalLoss(0.5, reduction='mean')
        self.dice_loss  = DiceLoss()
    
    def forward(self, y_pred, y_true, weights=None):
        floss = self.focal_loss(y_pred, y_true.argmax(1))
        dloss = self.dice_loss(y_pred, y_true)
        return ((floss + dloss)*100)

class WeightedFocalDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedFocalDiceLoss, self).__init__()    
        self.focal_loss = kornia.losses.FocalLoss(0.5, reduction='none')
        self.dice_loss  = DiceLoss()
    
    def forward(self, y_pred, y_true, weights=None):
        
        floss = self.focal_loss(y_pred, y_true.argmax(1)) * weights
        floss = floss.mean()
        dloss = self.dice_loss(y_pred, y_true)
        return ((floss + dloss)*10).clamp(0,20)
    
class WeightedCEFocalLoss(nn.Module):
    def __init__(self):
        super(WeightedCEFocalLoss, self).__init__()    
        self.focal_loss = kornia.losses.FocalLoss(0.5, reduction='none')
        self.bce = nn.CrossEntropyLoss(reduction='none')
     
    def forward(self, y_pred, y_true, weights=None):
        y_true_a = y_true.argmax(1)
        loss = (self.bce(y_pred, y_true_a) * self.focal_loss(y_pred, y_true_a) * weights)
        return loss.mean().clamp(0, 100)
    
class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.bce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, y_pred, y_true, weight):
        y_true_hot = y_true.argmax(1)
        loss = self.bce(y_pred, y_true_hot.long()) * weight
        return (loss.mean()*10).clamp(0, 20)


class MCELoss(nn.Module):

    def __init__(self):
        super(MCELoss, self).__init__()
        self.bce = nn.CrossEntropyLoss(weight=torch.tensor([1, 14.6]).cuda())
        
    def forward(self, y_pred, y_true, weight):
        y_true_hot = y_true.argmax(1)
        loss = self.bce(y_pred, y_true_hot.long())  
        return loss.mean() * 10

class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true, weights=None):        
        
        loss_0 = self.bce(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce(y_pred[:, 1], y_true[:, 1])
        loss_2 = self.bce(y_pred[:, 2], y_true[:, 2])
        loss_3 = self.bce(y_pred[:, 3], y_true[:, 3])
        
        loss = loss_0 * 0.1 + loss_1 * 0.5 + loss_2 * 0.3 + loss_3 * 0.3
        return loss * 100
    

class BCELoss2c(nn.Module):

    def __init__(self):
        super(BCELoss2c, self).__init__()
        self.bce0 = nn.BCEWithLogitsLoss()
        self.bce1 = nn.BCEWithLogitsLoss()

        print("INIT BCE LOSS2C")

    def forward(self, y_pred, y_true, weights=None):        
        
        loss_0 = self.bce0(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce1(y_pred[:, 1], y_true[:, 1])
        
        
        loss = (loss_0 )+ (loss_1)#* 14.6 * 4)## old 0.2 | 1
        return loss 
    
class WeightedBCELoss(nn.Module):

    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, weights):        
        
        loss_0 = self.bce(y_pred[:, 0], y_true[:, 0])
        loss_1 = self.bce(y_pred[:, 1], y_true[:, 1])
        loss_2 = self.bce(y_pred[:, 2], y_true[:, 2])
        loss_3 = self.bce(y_pred[:, 3], y_true[:, 3])
        
        #loss = (loss_0 * 0.1 + loss_1 * 0.5 + loss_2 * 0.3 + loss_3 * 0.3) * weights
        loss = (loss_0 + loss_1 + loss_2  + loss_3 ) * (weights)
        return loss.mean() * 10
    
class DiceLoss(nn.Module):
    # Adapted from: 
    ## https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html
    ## max value: 1.00 
    def __init__(self, dims=(1, 2, 3)) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.dims = dims

    def forward( self, input: torch.Tensor, target: torch.Tensor, weights=None) -> torch.Tensor:
        
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        smooth = 1
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)
        # compute the actual dice score
        intersection = torch.sum(input_soft * target, self.dims)
        cardinality  = torch.sum((input_soft + target), self.dims)

        dice_score = (2. * intersection + smooth)/ (cardinality + smooth +self.eps)
        return torch.mean(1. - dice_score)
    
class Dice(nn.Module):
    def __init__(self, dims=(1, 2, 3)) -> None:
        super(Dice, self).__init__()
        self.dice_loss = DiceLoss(dims)
    
    def forward( self, input: torch.Tensor, target: torch.Tensor, weights=None) -> torch.Tensor:
        return +1 - self.dice_loss(input, target)
    
class MSELoss(nn.Module):
    def __init__(self) -> None:
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        input_soft = F.softmax(input, dim=1)
        return self.mse_loss(input_soft, target)*10
    
class MSEDICELoss(nn.Module):
    def __init__(self) -> None:
        super(MSEDICELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        input_soft = F.softmax(input, dim=1)
        return self.mse_loss(input_soft, target)*10 + self.dice_loss(input, target)
    
class MCCLoss(nn.Module):
    # Adpte from: https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/7
    def __init__(self, eps=1e-6):
        super(MCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, w=None):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_mean = torch.mean(y_true)
        y_pred_mean = torch.mean(y_pred)
        y_true_var = torch.var(y_true)
        y_pred_var = torch.var(y_pred)
        y_true_std = torch.std(y_true)
        y_pred_std = torch.std(y_pred)
        vx = y_true - torch.mean(y_true)
        vy = y_pred - torch.mean(y_pred)
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + self.eps) * torch.sqrt(torch.sum(vy ** 2) + self.eps))
        ccc = (2 * pcc * y_true_std * y_pred_std) / \
              (y_true_var + y_pred_var + (y_pred_mean - y_true_mean) ** 2)
        ccc = 1 - ccc
        return ccc * 10
    
class MDiceLoss(nn.Module):
    def __init__(self) -> None:
        super(MDiceLoss, self).__init__()
        self.dice_loss_bg = DiceLoss(dims=(0))
        self.dice_loss_fg = DiceLoss(dims=(1))
        self.dice_loss_th = DiceLoss(dims=(2))
        self.dice_loss_gp = DiceLoss(dims=(3))
        
    def forward( self, input: torch.Tensor, target: torch.Tensor, w=None) -> torch.Tensor:
        bg = self.dice_loss_bg(input, target, w)
        fg = self.dice_loss_fg(input, target, w)
        tg = self.dice_loss_th(input, target, w)
        gp = self.dice_loss_gp(input, target, w)
        return (bg*0.2) + (fg*1.1) + (tg*3) + (gp*3)
    
    
class WeightedCEFocalDice(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=1.0):
        super(WeightedCEFocalDice, self).__init__()
        kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'none'}
        self.focal_loss = kornia.losses.FocalLoss(**kwargs)
        self.loss_mce   = nn.CrossEntropyLoss( reduction='none')
        self.loss_dice  = DiceLoss()                    
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, w=None ):
        
        loss_f  = self.focal_loss(y_pred, y_true.argmax(1))
        loss_m  = self.loss_mce( y_pred, y_true.argmax(1)) 
        loss    = (loss_f*w*loss_m).mean()
        
        loss_d  = self.loss_dice( y_pred, y_true )          
        loss = self.alpha*loss + self.gamma*loss_d
        return loss