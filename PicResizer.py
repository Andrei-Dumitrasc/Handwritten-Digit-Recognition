# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 20:08:14 2017

@author: Andrei
"""

import os
from PIL import Image
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import time
np.set_printoptions(threshold=400, edgeitems=10)

def getImgDims(path, name):
    with Image.open(os.path.join(path,name)) as img:
        return img.size

def allResInFolder(folder_name):
    path=folder_name
    names=os.listdir(path)    
    dims= [getImgDims(path,name) for name in names]
    widths, heights = [ [item[i] for item in dims] for i in [0,1]]
    return (np.array(widths,dtype=np.uint8),np.array(heights,dtype=np.uint8),names)

def findMaxResInFolder(folder_name):
    path=folder_name
    names=os.listdir(path)
    w=0
    h=0
    for name in names:
        with Image.open(os.path.join(path,name)) as img:
            width, height = img.size
        if width > w:
            w=width
        if height > h:
            h=height
    return (w,h)

def countImgInSize(folder_name):
    path=folder_name
    names=os.listdir(path)
    count=0
    for name in names:
        with Image.open(os.path.join(path,name)) as img:
            # cu aceste limite avem (6618 V + 6680 T) din 14000 (94.98%) poze valide
            if img.width>=16 and img.width<=61 and img.height>=25 and img.height<=75:
                count=count+1
    return count
            

def mtxPad(mtx,toHeight,toWidth):
    h,w=mtx.shape
    w0=round((toWidth-w)/2)
    h0=round((toHeight-h)/2)
    m=np.ones((toHeight,toWidth))
    m[h0:h0+h,w0:w0+w]=mtx
    return m

def mtxScale(mtx,toHeight,toWidth):
    h,w = mtx.shape
    factor = min([toHeight/h, toWidth/w])
    img = Image.fromarray(mtx)    
    return np.array(img.resize((int(factor*w),int(factor*h))))

def isValidSize(mtx,toHeight,toWidth):
    h,w = mtx.shape
    if w>=16 and w<=61 and h>=25 and h<=75:
        return True
    return False

def eagleEye(all_img):
#    f=plt.figure(0)
    while True:
        i=all_img[np.random.randint(0,all_img.shape[0])].reshape(75,61)
#        f.clear()
        f=plt.imshow(i,'gray')
#        wait=input()
        time.sleep(1)
        
# max rezo h=103 w=98
startOver=0
if startOver:    
    path='D:/Python_Projects/repository/BazeDate/cvl-digits/valid'
    widthsV,heightsV,namesV=allResInFolder(path)
    
    path='D:/Python_Projects/repository/BazeDate/cvl-digits/train'
    widthsT,heightsT,namesT=allResInFolder(path)
    
#all_img=1-sp.io.loadmat('D:/Python_Projects/repository/BazeDate/cvl-digits/3/cvl-imgs.mat')['all_img'].todense()
#eagleEye(all_img)
    
#widths=np.hstack((widthsT,widthsV))
#heights=np.hstack((heightsT,heightsV))
#origWsize=widths.size
#origHsize=heights.size
#
#heights=heights[ np.logical_and(heights>=25, heights<=75) ]
#print(heights.size /origHsize )
#
#widths=widths[ np.logical_and(widths>=16, widths<=61) ]
#print(widths.size /origWsize )

#widthsDist=np.bincount()
#heightsDist=np.bincount()
#
#plt.bar(np.arange(widthsDist.size),widthsDist)
#plt.figure()
#plt.bar(np.arange(heightsDist.size),heightsDist)


#plt.plot(widths)
#names=os.listdir(path)

#w,h=findMaxResInFolder(path)

#img=mpimg.imread(os.path.join(path,'0-0169-01-02.png'))
#imgplot = plt.imshow(img)
#
#img2 = io.imread(os.path.join(path,'0-0169-01-02.png'), as_grey=True)
#imgplot = plt.imshow(img2,'gray')
#
#img3=mtxPad(img2,96,74)
#imgplot = plt.imshow(img3,'gray')

#w=61
#h=75
#all_img=[]
#all_lbl=[]
#for path in ['D:/Python_Projects/repository/BazeDate/cvl-digits/train',
#             'D:/Python_Projects/repository/BazeDate/cvl-digits/valid']:
#    names=os.listdir(path)
#    for name in names:
#        img = io.imread(os.path.join(path,name), as_grey=True)
#        if isValidSize(img,h,w):
#            img = mtxScale(img,h,w)
#            img = mtxPad(img,h,w)
#            all_img.append(img.ravel())
#            all_lbl.append(int(name[0]))
#all_img=np.array(all_img)
#all_img=sp.sparse.csr_matrix(1-all_img)
#sp.io.savemat('cvl-imgs',{'all_img':all_img})
        

#s=(75,61)
#giant=np.array([mtxPad(io.imread(os.path.join(path,name), as_grey=True),*s).ravel()  for name in names])