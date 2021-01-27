
import cv2
import imutils
import os
import numpy as np

from torch.utils.data import Dataset

from pytvision.transforms.aumentation import  (
    ObjectImageMaskAndWeightTransform, ObjectImageAndMaskTransform,
    ObjectImageMaskAndSegmentationsTransform)
from pytvision.datasets import utility

from .imageutl import dsxbExProvide, nucleiProvide2, TCellsProvide, ISBIProvide
from .utility import to_one_hot
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")



train = 'train'
validation = 'val'
test  = 'test'


class DSXBDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    Args
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        folders_weights='weights',
        ext='png',
        num_channels=3,
        transform=None,
        ):
           
        self.data = dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )

        self.transform = transform    
        self.num_channels = num_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        image, label, contours, weight = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )  
        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        weight_t = weight[:,:,np.newaxis]        

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()




class DSXBExDataset(Dataset):
    '''
    Mnagement for Data Science Bowl image dataset
    https://www.kaggle.com/c/data-science-bowl-2018/data
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        folders_weights='weights',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):
        """           
        """            
           
        self.data = dsxbExProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                folders_weights,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label, contours, weight = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = np.zeros( (label.shape[0],label.shape[1],3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)

        weight_t = weight[:,:,np.newaxis]        

        obj = ObjectImageMaskAndWeightTransform( image_t, label_t, weight_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

    
    
class DSDataset(Dataset):
    '''Mnagement for Data Segmentation image dataset
    Args
    '''

    def __init__(self, 
        data,
        ext='png',
        num_channels=3,
        transform=None,
        ):
           
        self.data = data
        self.transform = transform    
        self.num_channels = num_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):   
        image, label = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )  
        label_t = np.zeros( (label.shape[0],label.shape[1],2) )
        label_t[:,:,0] = (label < 1)
        label_t[:,:,1] = (label >= 1)            

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()


class NucleiDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):

        self.data = nucleiProvide2(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_contours,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label, contours = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = np.zeros( (label.shape[0], label.shape[1], 3) )
        label_t[:,:,0] = (label < 128)
        label_t[:,:,1] = (label > 128)
        label_t[:,:,2] = (contours > 128)     

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

    
class TCellsDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):

        self.data = TCellsProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   

        label_t = (label > 127).astype(np.uint8)

        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()

class TCellsDataset2(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        folders_contours='touchs',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        ):

        self.data = TCellsProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                ext
                )


        self.transform = transform  
        self.count = count  
        self.num_channels = num_channels

    def __len__(self):
        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        image, label = self.data[idx] 
        image_t = utility.to_channels(image, ch=self.num_channels )   
        label_t = np.zeros_like( label )
        label_t[:,:,0] = (label[...,0] == 0)
        label_t[:,:,1] = (label[...,0] == 1)
        label_t[:,:,2] = (label[...,0] >= 2)     
        
        obj = ObjectImageAndMaskTransform( image_t, label_t  )
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()


class GenericDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels',
        ext='png',
        transform=None,
        count=1000,
        num_channels=3,
        use_weight=False,
        ):

        self.data = TCellsProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                ext,
                use_weight
                )


        self.transform    = transform  
        self.count        = count  
        self.num_channels = num_channels
        self.use_weight   = use_weight

    def __len__(self):
        if self.count is None:
            return len(self.data)

        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        data = self.data[idx]
        if self.use_weight:
            image, label, weight = data
        else:
            image, label = data
            
        label = (label == 255).astype(float) # 1024, 1024, 3, max= 1

        # image: 1024, 1024, 3, max = 255
        image_t = utility.to_channels(image, ch=self.num_channels )   
        # image_t: 1024, 1024, 3, max = 255
        
        if self.use_weight:
            obj = ObjectImageMaskAndWeightTransform(image_t, label, weight)
        else:
            obj = ObjectImageAndMaskTransform( image_t, label  )
        
        
        if self.transform: 
            obj = self.transform( obj )
        return obj.to_dict()
    
    
    
# TO MODIFY
class CoseStyleDataset(Dataset):

    def __init__(self, 
        base_folder,
        sub_folder,
        ext='png',
        count=1000,
        resize=256,
        transform=None,
        n_channels=1,
        n_inputs=2,
        style=None
        ):
        
        self.base_folder    = base_folder
        self.sub_folder     = sub_folder
        self.resize         = resize
        self.transform      = transform
        self.n_inputs       = n_inputs
        self.n_channels     = n_channels
        self.labels_tag     = 'labels' 
        self.original_tag   = 'images'
        self.style          = style
        self.count          = count
        
        path_data           = base_folder
        outputs_path      = f'{self.base_folder}/{self.sub_folder}/outputs'
        
        if sub_folder == 'train':
            self.sub_folder_path = 'train'
        elif sub_folder == 'val' or sub_folder == 'validation':
            self.sub_folder_path = 'validation'
        elif sub_folder == "test":
            self.sub_folder_path = 'test'
        else:
            print("Error: CoseDataset, set not define ", sub_folder)
        
        self.ids  = [Path(url).stem for url in glob.glob(base_folder + f'/{self.sub_folder_path}/{self.labels_tag}/*')]
        
        self.len  = len(self.ids)
        
    def get_label_url(self, idx):
        
        return f"{self.base_folder}/{self.sub_folder_path}/{self.labels_tag}/{self.ids[idx]}.png"
        
    def get_original_url(self, idx):

        return f"{self.base_folder}/{self.sub_folder_path}/{self.original_tag}/{self.ids[idx]}.png"   

    def get_src_url(self, idx):
        if self.style == "none" or self.style == 'original':
            return self.get_original_url(idx)
        return f"{self.base_folder}/{self.sub_folder_path}/style/{self.style}/{self.ids[idx]}.png"   
        
    def __len__(self):
        if self.count is None:
            return (self.len)

        return self.count  

    def one2two(self, src):
        dst = np.zeros(src.shape + (2,) ).astype(np.uint8)
        dst[..., 1] = src.max() == src
        dst[..., 0] = 1 - dst[..., 1]
        dst *= 255
        return dst
    
    def one2three(self, src):
        dst = np.zeros(src.shape + (3,)).astype(np.uint8)
        dst[..., 2] = src
        dst[..., 1] = src.max() - src
        return dst
    
    def join_src(self, src_list):
        
        dst = np.zeros((  (len(src_list),) + src_list[0].shape))
        for idx in range(len(src_list)):
            dst[idx] = src_list[idx]
        return dst
    
        
    def apply_augmentation(self, src, augs):
        if augs:
            for aug in augs:
                src = aug(src)
        return src   

    def fix_seeds(self, seed=10):
        #random.seed(seed)
        np.random.seed(seed)
        #torch.manual_seed(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        
    def get_item(self, idx):
        
        label_url  = self.get_label_url(idx)
        ori_url    = self.get_original_url(idx)
        src_url    = self.get_src_url(idx)
        
        gt         = cv2.imread(label_url, 0)
        ori        = cv2.imread(ori_url, 0)
        src        = cv2.imread(src_url)
        
        if src.shape[0] != self.resize or src.shape[1] != self.resize:
            src = imutils.resize(src, width=self.resize)
            src = src[:self.resize, :self.resize] # ensure size  
        
        
        if gt.shape[0] != self.resize or gt.shape[1] != self.resize:
            gt = imutils.resize(gt, width=self.resize)
            gt = gt[:self.resize, :self.resize] # ensure size   
            gt = ((gt>0)*255).astype(np.uint8)
        
        if ori.shape[0] != self.resize or ori.shape[1] != self.resize:
        
            ori = imutils.resize(ori, width=self.resize)
            ori = ori[:self.resize, :self.resize] # ensure size   

        gt         = self.one2three(gt)        
        
        obj = ObjectImageAndMaskTransform( src, gt.astype(np.float32)/255)
        
        if self.transform: 
            obj = self.transform( obj )
        obj = obj.to_dict()
        obj['ori'] = ori.astype(np.float32)/255
        
        return obj
        
    def __getitem__(self, idx):   
        if idx == 0 and (self.sub_folder == 'val' or self.sub_folder == 'validation' or self.sub_folder == 'test'):
            self.fix_seeds()
        
        idx  = idx % self.len
        data = self.get_item(idx)
        

        return data
    
    
    
class ISBIDataset(Dataset):

    def __init__(self, 
        base_folder, 
        sub_folder,  
        folders_images='images',
        folders_labels='labels4c',
        folders_weights='weights',
        folders_segments='outputs',
        ext='tif',
        transform=None,
        count=1000,
        num_channels=3,
        num_classes=4,
        use_weight=False,
        weight_name='SAW',
        load_segments=False,
        shuffle_segments=False,
        count_segments=5,
        ):

        self.data = ISBIProvide(
                base_folder, 
                sub_folder, 
                folders_images, 
                folders_labels,
                folders_weights,
                folders_segments,
                ext,
                use_weight,
                weight_name,
                load_segments                
                )


        self.transform        = transform  
        self.count            = count  
        self.num_channels     = num_channels
        self.use_weight       = use_weight
        self.num_classes      = num_classes
        self.load_segments    = load_segments
        self.shuffle_segments = shuffle_segments
        self.count_segments   = count_segments
        assert not (self.use_weight and self.load_segments)

    def __len__(self):
        if self.count is None:
            return len(self.data)

        return self.count  

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        data = self.data[idx]
        if self.use_weight:
            image, label, weight = data
        elif self.load_segments:
            image, label, segs = data
            if self.shuffle_segments:
                segs = segs[..., np.random.permutation(segs.shape[-1])]
            segs = segs[..., :self.count_segments]
        else:
            image, label = data
        
        image_t = utility.to_channels(image, ch=self.num_channels)
        
        label   = to_one_hot(label, self.num_classes)
        
        if self.use_weight:
            obj = ObjectImageMaskAndWeightTransform(image_t, label, weight)
        elif self.load_segments:
            obj = ObjectImageMaskAndSegmentationsTransform( image_t, label, segs )
        else:
            obj = ObjectImageAndMaskTransform( image_t, label  )
        
        if self.transform: 
            obj = self.transform( obj )
            
        obj = obj.to_dict()
        
        if self.load_segments: ## Warring!
            axis = np.argmin(obj['segment'].shape)
            inputs = np.concatenate((obj['image'], obj['segment']), axis=axis)
            obj['image'] = inputs
            obj.pop('segment')        
        return obj