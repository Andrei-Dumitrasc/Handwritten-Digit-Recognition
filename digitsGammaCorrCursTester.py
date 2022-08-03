# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 14:43:17 2017

@author: Andrei
"""

import numpy as np
import scipy as sp
import time
import matplotlib.pylab as mpl
import recognition as rcg
from PIL import Image,ImageFilter
import mnistImporter as qm
import random 


np.set_printoptions(threshold=400, edgeitems=10)

#    SET DIGITS
all_img,all_lbl=rcg.getDataSet('CVL');
# # partial dataset
# cap=2*140
# all_img=all_img[:cap]
# all_lbl=all_lbl[:cap]

nr_total_poze, nr_pixeli=(int(z) for z in all_img.shape)
rezo=np.sqrt(nr_pixeli).astype(np.int8)



#print(all_img.max())

 # ~440 poze/clasa
#cap=2
#all_img=np.vstack([all_img[i:i+cap] for i in range(0,nr_total_poze,437)])
#all_lbl=np.hstack([all_lbl[i:i+cap] for i in range(0,nr_total_poze,437)])
#nr_total_poze=int(all_img.shape[0])

#shuffled_idx=np.arange(nr_total_poze)
## nu neaparat 4, ci orice numar; ideea este sa avem aceeasi ordine la
## fiecare run
#random.Random(4).shuffle(shuffled_idx)
#all_lbl=all_lbl[shuffled_idx]
#all_img=all_img[shuffled_idx]


#   FLOAT <--> INT
# all_img*=255

  #     NEGATIVARE
#  print('NEGATIVARE ')
# all_img=1-all_img

#       BLURARE  
  # CERC 
# k=ImageFilter.Kernel((3, 3), (
     # 1, 1, 1,
     # 1, 1, 1,
     # 1, 1, 1
     # ))
  
  # PLUS  
# k=ImageFilter.Kernel((3, 3), (
   # 0, 1, 0,
   # 1, 1, 1,
   # 0, 1, 0
   # ))

#  #   
# k=ImageFilter.Kernel((5, 5), (
   # 1, 1, 1, 1, 1, 
   # 1, 1, 1, 1, 1, 
   # 1, 1, 1, 1, 1, 
   # 1, 1, 1, 1, 1, 
   # 1, 1, 1, 1, 1 
   # ))

#print('BLURARE ')
# for i in range(nr_total_poze):
 # # poza=(255*all_img[i,:].reshape(rezo,rezo)).astype(np.uint8)
 # poza=(255*all_img[i,:].reshape(75,61)).astype(np.uint8)
 # im=Image.fromarray(poza)
 # # im=im.filter(k)
 # for j in range(1):
   # im=im.filter(ImageFilter.SHARPEN)
 # all_img[i,:]=np.array(im).reshape((nr_pixeli,))/255

# # MARGINALIZARE
for i in range(nr_total_poze):
 # poza=all_img[i,:].reshape(rezo,rezo)
 poza=all_img[i,:].reshape(75,61)# pentru CVL   
 norm_lin=sp.linalg.norm(poza,axis=1)
 norm_col=sp.linalg.norm(poza,axis=0)
 # ELIMINAM MARGINEA (LINII SI COLOANE DE NORMA 0)
 poza_marg=poza[norm_lin!=0]
 poza=poza_marg[:,norm_col!=0]
 poza=(255*poza).astype(np.uint8)
 im=Image.fromarray(poza)
 # im=im.resize((rezo,rezo))
 im=im.resize((75,61))# pentru CVL 
 all_img[i,:]=np.array(im).reshape((nr_pixeli,))/255
          
    #   BINARIZARE
#print('BINARIZARE ')
# all_img[all_img>0]=1;
# all_img[all_img<1]=0;# pentru CVL


    
       # RAPORT PIXELI-CIFRA/PIXELI-TOTAL
#print ('\n cifra/total= '+'{:.6f}'.format(np.median(np.linalg.norm(all_img,
#         ord=0,axis=1)/all_img.shape[1])))
#
##        #   PRINTARE
#for i_img in range(0,10,1):
#  mp=all_img[i_img,:].reshape(28,28,order='C');    
#  np.savetxt("digitsGammaMSH10"+str(i_img)+".txt",mp, fmt='%.5f')
#  rp=(255*mp).astype(np.uint8); 
#  Image.fromarray(rp).save("digitsGammaMSH10"+str(i_img)+".png")

folds=10
nr_algs=3
nr_ant=int((folds-1)/folds*nr_total_poze)
nr_test=nr_total_poze-nr_ant
# nr_test+1
lims=np.linspace(0,nr_test, nr_test+1).astype(np.uint16)
rec_rate=np.zeros((nr_algs,folds))
idx_ant=np.concatenate((np.ones((nr_ant),dtype=bool),np.zeros((nr_test),dtype=bool)))
norme=['1','2'] 
print('\nnorm  NN      kNN     EF    Lan    COD')  
for norma in norme[1]:
  clin = time.time()
  # CROSS-VALIDATION
  for i_fold in range(folds)[0:]:
    idx_ant=np.roll(idx_ant,nr_test)
    idx_test=np.logical_not(idx_ant)
    poze_ant=all_img[idx_ant,:]
    lbl_ant=all_lbl[idx_ant]
    poze_test=all_img[idx_test,:]
    lbl_test=all_lbl[idx_test]
    fold_time=time.time()
  #  print(lbl_test)

    found_lbl=[]
    
#    found_lbl.append([rcg.NN(poze_ant,lbl_ant,poza,norma) for poza in poze_test])
    
    # NN
    part_found=[] 
    # for i in range(len(lims)-1):
        # part_found=part_found + rcg.query(np.eye(len(poze_test[0])), lbl_ant,
                                # poze_ant, poze_test[lims[i]:lims[i+1]],norma)
    # found_lbl.append(part_found)
    
#    
    # found_lbl.append([rcg.kNN(poze_ant,lbl_ant,poza,norma,4) for poza in poze_test])

    V,media,proiectie=rcg.EFpreproc(poze_ant,32)
#    print('done prj')
    # found_lbl.append([rcg.EFquery(V,lbl_ant,media,proiectie,poza,norma) for poza in poze_test])

    part_found=[] 
    for i in range(len(lims)-1):
      part_found=part_found + rcg.query(V, lbl_ant, proiectie, poze_test[lims[i]:lims[i+1]]-media,norma) 
    found_lbl.append(part_found)
    
#    prt=time.time()
    V,proiectie=rcg.Lanpreproc(poze_ant,61)
#    print(str(time.time()-prt)+' durata proiectie')
    # found_lbl.append([rcg.LanCODquery(V,lbl_ant,proiectie,poza,norma) for poza in poze_test])
#    found_lbl.append(rcg.query(Q,lbl_ant,proiectie,poze_test,norma))
    
    part_found=[] 
    for i in range(len(lims)-1):
        part_found=part_found + rcg.query(V, lbl_ant, proiectie, poze_test[lims[i]:lims[i+1]],norma) 
    found_lbl.append(part_found)
     
    V,proiectie=rcg.CODpreproc(poze_ant,57)
    # found_lbl.append([rcg.LanCODquery(V,lbl_ant,proiectie,poza,norma) for poza in poze_test])
#    found_lbl.append(rcg.query(V,lbl_ant,proiectie,poze_test,norma))
    
    part_found=[] 
    for i in range(len(lims)-1):
        part_found=part_found + rcg.query(V, lbl_ant, proiectie, poze_test[lims[i]:lims[i+1]],norma) 
    found_lbl.append(part_found)

  #  print(found_lbl)
  #  print(poze_test[0].reshape((20,20),order='F'))

    for j in range(len(found_lbl)):
      recogn=[np.where(lbl_test[i] in found_lbl[j][i],True,False) for i in range(nr_test)]
      rec_rate[j,i_fold]=np.mean(recogn)  
    
#    print(str(time.time() - fold_time)+' secs to complete a fold')
      
  l=list(map(lambda x: '{:3.2f}'.format(x),np.mean(rec_rate,axis=1)*100))
  sep='  '
  print(' '+norma+'    '+sep.join(l))
  print(str(time.time() - clin)+' secs')
#print(all_lbl)
#print(idx_ant)