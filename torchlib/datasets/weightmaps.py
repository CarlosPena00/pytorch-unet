# Source: 
## Autor: Fidel Guerrero Pena and Pedro Diamel
## Extra: https://arxiv.org/pdf/1505.04597.pdf

import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image, thin
from scipy.spatial.distance import cdist

def getunetweightmap( merged_mask, masks, w0=10, sigma=5, ):
    
    # WxHxN to NxWxH
    #masks = masks.transpose( (2,0,1) )
    
    weight = np.zeros(merged_mask.shape)
    # calculate weight for important pixels
    distances = np.array([ndimage.distance_transform_edt(m==0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell 
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    w_b = np.exp(-(d1+d2)**2/(2*sigma**2)).astype(np.float32)
    w_c = getweightmap(merged_mask)
    w = w_c + w0*w_b
    
    #weight = 1 + (merged_mask == 0) * w
    weight = 1 + w
    
    return weight


#weight map
def getweightmap(mask):
    
    w_c = np.empty(mask.shape)
    classes = np.unique(mask)
    frecs = [ np.sum(mask == i)/float(mask.size) for i in classes ] 
            
    # Calculate
    n = len(classes)
    for i in range( n ):
        w_c[mask == i] = 1 / (n*frecs[i])
    
    return w_c


# class balance weight map
def balancewm(mask):

    wc = np.empty(mask.shape)
    classes = np.unique(mask)
    freq = [ 1.0 / np.sum(mask==i) for i in classes ]
    freq /= max(freq)


    for i in range(len(classes)):
        wc[mask == classes[i]] = freq[i]

    return wc 

# unet weight map
def unetwm(mask,w0,sigma):
    mask = mask.astype('float')
    wc = balancewm(mask)

    cells,cellscount = ndimage.measurements.label(mask==1)
    maps = np.zeros((mask.shape[0],mask.shape[1],cellscount))
    
    for ci in range(1,cellscount+1):
        maps[:,:,ci-1] = ndimage.distance_transform_edt(cells!=ci)
    maps = np.sort(maps,axis=2)
    d1 = maps[:,:,0]
    if cellscount>1:
        d2 = maps[:,:,1]
    else:
        d2 = d1
    uwm = 1 + wc + (mask==0).astype('float')*w0*np.exp( (-(d1+d2)**2) / (2*sigma)).astype('float')
    
    return uwm

def distranfwm(mask,beta):
    mask = mask.astype('float')
    wc = balancewm(mask)

    dwm = ndimage.distance_transform_edt(mask!=1)
    dwm[dwm>beta] = beta
    dwm= wc + (1.0 - dwm/beta) +1
    
    return dwm

def shapeawewm(mask,sigma):
    diststeps=10000
    mask = mask.astype('float')
    wc = balancewm(mask)
    binimage=(mask==1).astype('float')
    
    cells,cellscount = ndimage.measurements.label(binimage)
    chull = np.zeros_like(mask)
    # convex hull of each object
    for ci in range(1,cellscount+1):
        I = (cells==ci).astype('float')
        R = convex_hull_image(I) - I
        R = ndimage.binary_opening(R,structure=np.ones((3,3))).astype('float')
        R = ndimage.binary_dilation(R,structure=np.ones((3,3))).astype('float')
        chull += R

    # distance transform to object skeleton
    skcells=thin(binimage)
    dtcells=ndimage.distance_transform_edt(skcells!=1)
    border=binimage-ndimage.binary_erosion(input=(binimage),structure=np.ones((3,3)),iterations=1).astype('float')
    tau=np.max(dtcells[border==1])+0.1
    dtcells=np.abs(1-dtcells*border/tau)*border

    # distance transform to convex hull skeleton
    skchull=thin(chull)
    dtchull=ndimage.distance_transform_edt(skchull!=1)
    border=chull-ndimage.binary_erosion(input=(chull),structure=np.ones((3,3)),iterations=1).astype('float')
    dtchull=np.abs(1-dtchull*border/tau)*border

    # maximum border
    saw=np.concatenate((dtcells[:,:,np.newaxis],dtchull[:,:,np.newaxis]),2)
    saw = np.max(saw,2)
    saw /= np.max(saw)

    # propagate contour values inside the objects
    prop=binimage+chull
    prop[prop>1]=1
    prop=ndimage.binary_erosion(input=(prop),structure=np.ones((3,3)),iterations=1).astype('float')
    current_saw=saw

    for i in range(20):
        tprop=ndimage.binary_erosion(input=(prop),structure=np.ones((3,3)),iterations=1).astype('float')
        border=prop-tprop
        prop=tprop

        x1,y1 = np.where(border!=0)
        x2,y2 = np.where(current_saw!=0)

        if x1.size==0 or x2.size==0: break

        tsaw=np.zeros_like(saw)
        for a in range(0,x1.size,diststeps):
            minl=np.min(np.array([diststeps+a-1,x1.size-1])) +1
            dis=cdist(np.vstack((x2,y2)).transpose(),np.vstack((x1[a:minl],y1[a:minl])).transpose() )
            ind=np.argmin(dis,axis=0)
            tsaw[x1[a:minl],y1[a:minl]]=current_saw[x2[ind],y2[ind]]

        saw=np.concatenate((saw[:,:,np.newaxis],tsaw[:,:,np.newaxis]),2)
        saw = np.max(saw,2)
        saw=ndimage.filters.gaussian_filter(saw,sigma)
        saw/=np.max(saw)
        current_saw=saw*(border!=0).astype('float')

    saw = saw + wc +1
    return saw
