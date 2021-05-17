import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

def tolabel(mask):
    labeled, nr_true = ndi.label(mask)
    return labeled

def torgb(im):
    if len(im.shape)==2:
        im = np.expand_dims(im, axis=2) 
        im = np.concatenate( (im,im,im), axis=2 )
    return im
        
def setcolor(im, mask, color):
    
    tmp=im.copy()
    tmp=np.reshape( tmp, (-1, im.shape[2])  )   
    mask = np.reshape( mask, (-1,1))      
    tmp[ np.where(mask>0)[0] ,:] = color
    im=np.reshape( tmp, (im.shape)  )
    return im

def lincomb(im1,im2,mask, alpha):
    
    #im = np.zeros( (im1.shape[0], im1.shape[1], 3) )
    im = im1.copy()    
    
    row, col = np.where(mask>0)
    for i in range( len(row) ):
        r,c = row[i],col[i]
        #print(r,c)
        im[r,c,0] = im1[r,c,0]*(1-alpha) + im2[r,c,0]*(alpha)
        im[r,c,1] = im1[r,c,1]*(1-alpha) + im2[r,c,1]*(alpha)
        im[r,c,2] = im1[r,c,2]*(1-alpha) + im2[r,c,2]*(alpha)
    return im

def makebackgroundcell(labels):
    ch = labels.shape[2]
    cmap = plt.get_cmap('Set1')
    imlabel = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        imlabel = setcolor(imlabel,mask,color[:3])
    return imlabel

def makeedgecell(labels, thickness=2):
    ch = labels.shape[2]
    cmap = plt.get_cmap('Set1')
    imedge = np.zeros( (labels.shape[0], labels.shape[1], 3) )    
    for i in range(ch):
        mask  = labels[:,:,i]
        color = cmap(float(i)/ch)
        mask = mask.astype(np.uint8)
        if cv2.__version__[0] == '4':
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
        else:
            _,contours,_ = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE )
        for cnt in contours: cv2.drawContours(imedge, cnt, -1, color[:3], thickness)
    return imedge

def makeimagecell(image, labels, alphaback=0.3, alphaedge=0.3, edge_thickness=2):
    
    imagecell = image.copy()
    imagecell = imagecell - np.min(imagecell)
    imagecell = imagecell / np.max(imagecell)    
    imagecell = torgb(imagecell)
     
    mask  = np.sum(labels, axis=2)
    imagecellbackground = makebackgroundcell(labels)
    imagecelledge = makeedgecell(labels, edge_thickness)
    
    maskedge = np.sum(imagecelledge, axis=2)
    
    imagecell = lincomb(imagecell,imagecellbackground, mask, alphaback )
    imagecell = lincomb(imagecell,imagecelledge, maskedge, alphaedge )
            
    return imagecell

def decompose(labeled):
    nr_true = labeled.max()
    masks = []
    for i in range(1, nr_true + 1):
        msk = labeled.copy()
        msk[msk != i] = 0.
        msk[msk == i] = 255.
        masks.append(msk)
    if not masks: return np.array([labeled])
    else: return np.array(masks)
    
def makeimagecellview(original, pred): # pred == predictionlb
    dec = decompose(pred)
    dec = dec[np.random.permutation(dec.shape[0])].transpose(1,2,0)
    dec = makeimagecell(original, dec, alphaback=0.2, alphaedge=0.9, edge_thickness=3)
    return dec