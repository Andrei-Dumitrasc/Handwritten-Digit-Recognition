# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 18:27:08 2017

@author: Andrei
"""

import numpy as np
import scipy as sp
from scipy import io
import collections as cs
import mnistImporter as qm
import random
import os
from time import time

#   INTOARCE BAZA DE DATE RANDOMIZATA/ARANJATA PE SETURI 0-9, 0-9 etc,
#   PIXELI IN [0,1]
def getDataSet(numeSet):

  if (numeSet=='MNIST'):
    all_lbl1,all_img1=qm.read('/digits/t10k-images-idx3-ubyte',
      '/digits/t10k-labels-idx1-ubyte')
    all_lbl2,all_img2=qm.read('/digits/train-images-idx3-ubyte',
      '/digits/train-labels-idx1-ubyte')
    all_lbl=np.concatenate((all_lbl1,all_lbl2))
    all_img=np.concatenate((all_img1,all_img2))/255
    shuffled_idx=np.arange(len(all_lbl))


  elif (numeSet=='digitsMNIST'):
    all_lbl,all_img=qm.read('/digits/seewald/MNIST-transformed-data/all-digits-MNIST-gammaC.data',
      '/digits/seewald/MNIST-transformed-data/all-digits-MNIST-gammaC.labels')
    nr_total_poze=int(all_img.shape[0])
    # nu neaparat 4, ci orice numar; ideea este sa avem aceeasi ordine la
    # fiecare run
    shuffled_idx=np.arange(nr_total_poze)
    all_img=all_img/255

  elif (numeSet=='stanf'):
    # IMPORT POZE; CREARE LABELS
    all_img=io.loadmat('/digits/cifre_stanf_orig/cifre_stanf.mat')['m'].T
    nr_total_poze=int(all_img.shape[0])
#    all_lbl=np.tile(np.arange(10),(int(nr_total_poze/10)))
    all_lbl=np.tile(np.arange(10),(int(nr_total_poze/10),1)).ravel(order='F')
    all_img=all_img/255
    shuffled_idx=np.arange(nr_total_poze)

  elif (numeSet=='CVL'):
    all_img=np.array(1-io.loadmat('/digits/cvl-imgs.mat')['all_img'].todense())

    nr_total_poze=int(all_img.shape[0])
    all_lbl=np.squeeze(io.loadmat('/digits/cvl-lbl.mat')['all_lbl'])

    # SCHIMBAM DOMENIUL IN [0;1]
    all_img=np.array([i-i.min() for i in all_img])
    all_img=np.array([i/i.max() for i in all_img])
      
    shuffled_idx=np.arange(nr_total_poze)

  elif (numeSet=='digits'):
    all_img=np.vstack((io.loadmat('/digits/seewald/train_img.mat')['train_img'],
      io.loadmat('/digits/seewald/test_img.mat')['test_img']))

    all_lbl=np.vstack((io.loadmat('/digits/seewald/train_lbl.mat')['train_lbl'],
      io.loadmat('/digits/seewald/test_lbl.mat')['test_lbl']))     

    # SCHIMBAM DOMENIUL DIN [-1;1] IN [0;1]
    all_lbl=all_lbl.squeeze()
    all_img=(all_img+1)/2

    shuffled_idx=np.arange(len(all_lbl))
    

#  # AMESTECAM POZELE
  random.Random(4).shuffle(shuffled_idx)
  all_lbl=all_lbl[shuffled_idx]
  all_img=all_img[shuffled_idx]

  return (all_img, all_lbl)

#  pe linii
def CODpreproc(A, k):
  n,m=A.shape
  uc = np.zeros((k + 1, m));
  vc = np.zeros((k + 1, n));
  ut = np.zeros((k, m ));
  vt = np.zeros((k, n ));

  uc[0] = np.dot( np.ones((n)), A )
  uc[0] = uc[0] / sp.linalg.norm(uc[0],ord=2)
  vc[0] = np.dot(np.ones((m)), A.T )
  vc[0] = vc[0] / sp.linalg.norm(vc[0],ord=2)
  A2 = A.copy();

  for i in range(k):
    u=np.dot(vc[i], A2)
    un=sp.linalg.norm(u,ord=2)
    ut[i]=u/un

    v=np.dot(uc[i], A2.T)
    vn=sp.linalg.norm(v,ord=2)
    vt[i]=v/vn

    s=np.dot(un,vn)/np.dot(uc[i].T,u)
    A2=A2-s*np.outer(vt[i].T,ut[i])

    u2 = np.dot(np.ones((n)),  A2  )
    uc[i+1] = u2 / sp.linalg.norm(u2,ord=2)
    v2 = np.dot( np.ones((m)), A2.T )
    vc[i+1] = v2 / sp.linalg.norm(v2,ord=2)

  ut=ut.T
  proiectie=np.dot( A, ut )
  return (ut, proiectie)

##   pe coloane ~ Matlab
#def CODpreproc(A, k):
#  A=A.T
#  m,n=A.shape
#  uc = np.zeros((m, k + 1));
#  vc = np.zeros((n, k + 1));
#  ut = np.zeros((m, k ));
#  vt = np.zeros((n, k ));
#
#  uc[:, 0] = np.dot(A , np.ones((n)))
#  uc[:, 0] = uc[:, 0] / sp.linalg.norm(uc[:, 0],ord=2)
#  vc[:, 0] = np.dot(A.T , np.ones((m)))
#  vc[:, 0] = vc[:, 0] / sp.linalg.norm(vc[:, 0],ord=2)
#  A2 = A;
#
#  for i in range(k):
#    u=np.dot(A2,vc[:,i])
#    un=sp.linalg.norm(u,ord=2)
#    ut[:,i]=u/un
#
#    v=np.dot(A2.T,uc[:,i])
#    vn=sp.linalg.norm(v,ord=2)
#    vt[:,i]=v/vn
#
#    s=np.dot(un,vn)/np.dot(uc[:,i].T,u)
#    A2=A2-s*np.outer(ut[:,i],vt[:,i].T)
#
#    u2 = np.dot(A2 , np.ones((n)))
#    uc[:, i+1] = u2 / sp.linalg.norm(u2,ord=2)
#    v2 = np.dot(A2.T , np.ones((m)))
#    vc[:, i+1] = v2 / sp.linalg.norm(v2,ord=2)
#
#  proiectie=np.dot(A.T , ut)
#  return (ut, proiectie)

def LanCODquery(Q, L, proiectie, poza, numeNorma):
#  pentru o poza
  n=proiectie.shape[0]
  poza=poza.astype(np.float64)
  poza_pro=np.dot(poza.T,Q)
  z=sp.linalg.norm(proiectie-poza_pro,ord=int(numeNorma),axis=1)
  mz=min(z)
  iPoza=[int(i) for i, x in enumerate(z) if x == mz]
  return L[iPoza]

def query(Q, L, proiectie, poze, numeNorma):
#  proiectam pozele pe subspatiul de dim mai mica  
  poze_pro=np.dot(poze,Q)
#  calculam norma diferentei dintre fiecare poza de test si fiecare poza de
#       antrenare;
#  folosim broadcast pentru a potrivi dimensiunile matricelor
  norme=sp.linalg.norm(proiectie[np.newaxis,:,:]-poze_pro[:,np.newaxis,:],
          ord=int(numeNorma),axis=2)  
  
#  proiectie=proiectie[np.newaxis,:,:]
#  poze_pro=poze_pro[:,np.newaxis,:]
#  norme=sp.linalg.norm(proiectie-poze_pro,ord=int(numeNorma),axis=2) 
  
  min_norme=np.min(norme,axis=1)
  
#  determinam pt fiecare linie/poza de testare indicele pozei/pozelor cele mai 
#     apropiate (diferenta de norma minima)
  truth=norme==min_norme[:,np.newaxis]
  found_idx=[r.nonzero() for r in truth]
#  returnam o lista cu etichetele pozelor cele mai app de fiecare poza (jagged
#        array)
  return [L[idx_closest] for idx_closest in found_idx]

#   pe coloane ~ Matlab
#def Lanpreproc(A, k):
#  A=A.T
#  m,n=A.shape
#  b=0
#  Q=np.zeros((m,k+1))
#  Q[:,1]=np.ones((m))
#  Q[:,1]=Q[:,1]/sp.linalg.norm(Q[:,1],ord=2)
#  for i in range(1,k):
#    w=np.dot(A,np.dot(A.T,Q[:,i]))-np.dot(b,Q[:,i-1])
#    al=np.dot(w,Q[:,i-1])
#    w=w-np.dot(al,Q[:,i])
#    b=sp.linalg.norm(w,ord=2)
#    Q[:,i+1]=w/b
#  Q=Q[:,1:k+1]
#  proiectie=np.dot(A.T,Q)
#  return (Q, proiectie)

#  pe linii
def Lanpreproc(A, k):
  n,m=A.shape
  b=0
  Q=np.zeros((k+1,m))
  Q[1]=np.ones((m))
  Q[1]=Q[1]/sp.linalg.norm(Q[1],ord=2)
  for i in range(1,k):
    w=np.dot(A.T,np.dot(A,Q[i]))-b*Q[i-1]
    al=np.dot(w,Q[i-1])
    w=w-al*Q[i]
    b=sp.linalg.norm(w,ord=2)
    Q[i+1]=w/b
  Q=Q[1:k+1]
  Q=Q.T
  proiectie=np.dot(A,Q)
  return (Q, proiectie)

def EFpreproc(O,k):
  A=O.copy()  
  m,n=A.shape
#  A=A.astype(np.float64)
  media=np.mean(A,axis=0)
  A-=media
  if n>m:
    L=np.dot(A,A.T)
  else:
    L=np.dot(A.T,A)
  ev,eV=np.linalg.eig(L)
#  ev=ev.real.astype(np.float64)
#  eV=eV.real.astype(np.float64)
  ev=ev.real
  eV=eV.real
  eV=eV[:,np.argsort(ev)]
  if (n>m):
    eV=np.dot(A.T,eV)
  V=eV[:,::-1][:,0:k]
  proiectie=np.dot(A,V)
  return (V,media,proiectie)

def EFqueryV2(V, L, media, proiectie, poze, numeNorma):
  return query(V, L, proiectie, poze-media, numeNorma)

def EFquery(V, L, media, proiectie, poza, numeNorma):
  n=proiectie.shape[0]
  poza=poza.astype(np.float64)
  poza-=media
  poza_pro=np.dot(poza.T,V)
  z=sp.linalg.norm(proiectie-poza_pro,ord=int(numeNorma),axis=1)
  mz=min(z)
  iPoza=[int(i) for i, x in enumerate(z) if x == mz]
  return L[iPoza]

def w(A):
  print( A.reshape((20,20),order='F') )

def NN(A,L, poza, numeNorma):
  z=sp.linalg.norm(A-poza,ord=int(numeNorma),axis=1)
  mz=min(z)
  iPoza=[int(i) for i, x in enumerate(z) if x == mz]
  return L[iPoza]

def kNN(A,L, poza, numeNorma, k):
  z=sp.linalg.norm(A-poza,ord=int(numeNorma),axis=1)
  M=np.vstack((z,L))
  M2=M[:,np.argsort(M[0,:])]
  topM=np.sort(M2[1,0:k])
  counter=cs.Counter(topM)
  mx=max(counter.values())
  return [k for k,v in counter.items() if v==mx]
