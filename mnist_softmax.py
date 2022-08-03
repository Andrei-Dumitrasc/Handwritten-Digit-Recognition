# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
import numpy as np
import scipy as sp
import recognition as rcg
import time

from PIL import Image,ImageFilter
#from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


import tensorflow as tf

FLAGS = None

def imgdisp(img_mat, idx_list):
    for idx in idx_list:
        plt.imshow(np.reshape(img_mat[idx],(103,98)),'gray')

def main(_):
  # Import data

#  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
#  all_img=np.concatenate((mnist.train.images,mnist.test.images,mnist.validation.images))
#  all_lbl=np.concatenate((mnist.train.labels,mnist.test.labels,mnist.validation.labels))
  all_img,lbl_ind=rcg.getDataSet('CVL')
#  d_1=np.linalg.inv(np.diag(np.diag(all_img)))
#  all_img=np.dot(np.dot(d_1,all_img),d_1)
#  print(type(input_data))


  nr_total_poze, nr_pixeli=(int(z) for z in all_img.shape)
  rezo=np.sqrt(nr_pixeli).astype(np.int8)
  print(nr_total_poze, nr_pixeli)
#  print(all_img.max())
  print(lbl_ind.shape)

  all_lbl=np.zeros((nr_total_poze,10))
  all_lbl[np.arange(nr_total_poze),lbl_ind]=1

#  imgdisp(all_img,[0,699,700,1399,1400,7000,7699])
#  print(lbl_ind)

#  print(all_lbl.shape)

  # Create the model
  x = tf.placeholder(tf.float32, [None, nr_pixeli])
  W = tf.Variable(tf.zeros([nr_pixeli, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()

  tf.global_variables_initializer().run()

#  poza=255*all_img[0,:].reshape(28,28).astype(np.uint8)
#  print(all_img[0,:])
#  print(poza)
#  im=Image.fromarray(poza)
#  im.save('unblurred.png')
#  im=im.filter(ImageFilter.BLUR)
#  im.save('blurred.png')

#  import pdb; pdb.set_trace()

#   FLOAT <--> INT
  # all_img*=255

  #     NEGATIVARE
#  print('NEGATIVARE ')
  # all_img=1-all_img

#       BLURARE
#  print('BLURARE ')
# CERC
  # print('cerc')
  # k=ImageFilter.Kernel((3, 3), (
           # 1, 1, 1,
           # 1, 1, 1,
           # 1, 1, 1
           # ))
           
# PLUS 
  # print('plus')
  # k=ImageFilter.Kernel((3, 3), (
     # 0, 1, 0,
     # 1, 1, 1,
     # 0, 1, 0
     # ))

  # k=ImageFilter.Kernel((5, 5), (
   # 1, 1, 1, 1, 1,
   # 1, 1, 1, 1, 1,
   # 1, 1, 1, 1, 1,
   # 1, 1, 1, 1, 1,
   # 1, 1, 1, 1, 1
   # ))

#  k=ImageFilter.SHARPEN

  # for i in range(nr_total_poze):
   # # poza=(255*all_img[i,:].reshape(rezo,rezo)).astype(np.uint8)   
   # poza=(255*all_img[i,:].reshape(75,61)).astype(np.uint8) # pentru CVL   
   # im=Image.fromarray(poza)
   # im=im.filter(k)
   # # for j in range(10):
     # # im=im.filter(ImageFilter.SHARPEN)
   # all_img[i,:]=np.array(im).reshape((nr_pixeli,))/255

# MARGINALIZARE
  # for i in range(nr_total_poze):
   # # poza=all_img[i,:].reshape(rezo,rezo)
   # poza=all_img[i,:].reshape(75,61)# pentru CVL   
   # norm_lin=sp.linalg.norm(poza,axis=1)
   # norm_col=sp.linalg.norm(poza,axis=0)
   # # ELIMINAM MARGINEA (LINII SI COLOANE DE NORMA 0)
   # poza_marg=poza[norm_lin!=0]
   # poza=poza_marg[:,norm_col!=0]
   # poza=(255*poza).astype(np.uint8)
   # im=Image.fromarray(poza)
   # # im=im.resize((rezo,rezo))
   # im=im.resize((75,61))# pentru CVL   
   # all_img[i,:]=np.array(im).reshape((nr_pixeli,))/255

    #   BINARIZARE
#  print('BINARIZARE ')
#  all_img[all_img>0]=1;
   # pentru CVL
  
  mins=np.amin(all_img,axis=1)
  for i in range(nr_total_poze):
     all_img[all_img>mins[i]]=1;
     all_img[all_img<=mins[i]]=0;
  print('bin')
   
 # GAMMA CORRECTION
#  all_img/=all_img.max(axis=1)[:,None]

    # THRESHOLD
#  print('THRESHOLD ')
#  prag=70/255
#  print('prag')
#  print(prag)
#  all_img[all_img<prag]=0

# #        #   PRINTARE
  # for i_img in range(7,208,10):
    # rp=all_img[i_img,:].reshape(rezo,rezo,order='F');
    # # MATRICEA SALVATA FLOAT
    # np.savetxt("digitsMNIST"+str(i_img)+".txt",rp, fmt='%.5f')
    # rp=(255*rp).astype(np.uint8);
     # # MATRICEA SALVATA INT
  # #  np.savetxt("digitsO"+str(i_img)+".txt",rp, fmt='%3d')
    # Image.fromarray(rp).save("digitsMNIST"+str(i_img)+".png")

#  print(mnist.train.images[0,:].reshape(28,28))
#  print(mnist.train.images[2220])

  # GRUPARE POZE
#  sp_lbl=np.array([ j for i in sp.sparse.lil_matrix(all_lbl).rows for j in i ])
#  arranged_all_img=np.zeros_like(all_img)
#  k_start=0
#  nr_reps=np.zeros(10);
#  for lbl in range(10):
#      idx_lbl=np.where(sp_lbl==lbl)
#      k_end=k_start+idx_lbl[0].size
#      nr_reps[lbl]=k_end-k_start
##      print(k_end)
#      arranged_all_img[k_start:k_end,:]=all_img[idx_lbl,:]
#      k_start=k_end
#  nr_reps=nr_reps.astype(np.uint16)


  #     CORELARE
#  tempSim=np.corrcoef(all_img)
#  xi=0
#  yi=0
#  xf=0
#  yf=0
#  S=np.zeros((10,10))
#  for i1 in range (10):
#    yi=yi+yf
#    yf=yf+nr_reps[i1]
#    for i2 in range (i1,10):
#      xi=xi+xf
#      xf=xf+nr_reps[i2]
#      bloc=np.corrcoef(all_img[yi:yf+1],all_img[xi:xf+1])
#      if (i2==i1):
#        bloc=bloc[bloc!=0]
#      linie=bloc.flatten()
#      linie=linie[np.logical_not(np.isnan(linie))]
#      S[i1,i2]=np.median(linie)
#      print(S)
#  simil=S+np.tril(S.T,-1)
#  print(np.sum(np.absolute(simil-np.diag(simil)*np.ones(10)),1))

  #  CORELARE 2
#  S=np.zeros((10,10))
#  limits=np.cumsum(np.hstack((np.array([0]),nr_reps)))
#
#  for i1 in range(9,10):
#    g1=all_img[limits[i1]:limits[i1+1]]
#    for i2 in range(i1,10):
#      g2=all_img[limits[i2]:limits[i2+1]]
#      S[i1,i2]=np.median(np.array([np.corrcoef(p1,p2)[1,0] for p1 in g1 for p2 in g2]))

  # LISTA RANK, COND PT FIECARE POZA
#  ranks=[np.linalg.matrix_rank(poza.reshape(rezo,rezo))/rezo for poza in all_img]
#  conds=[np.linalg.cond(poza.reshape(rezo,rezo)) for poza in all_img]



         # RAPORT PIXELI-CIFRA/PIXELI-TOTAL
#  print ('\n cifra/total= '+'{:.6f}'.format(np.median(np.linalg.norm(all_img,
#         ord=0,axis=1)/all_img.shape[1])))

  clin = time.time()
  folds=10
  nr_ant=int((folds-1)/folds*nr_total_poze)
  nr_test=nr_total_poze-nr_ant
  steps=1000
  print('Nr pasi de antrenare= '+ str(steps))
  # dim_bloc=int(3/1*nr_ant/steps)
  dim_bloc=50
  print('Dimensiunea blocului de antrenare = '+ str(dim_bloc))

  rec_rate=np.zeros(folds)

  idx_ant=np.concatenate((np.zeros((nr_test),dtype=bool),np.ones((nr_ant),dtype=bool)))
  idx_grad=np.concatenate((np.ones((dim_bloc),dtype=bool),np.zeros((nr_ant-dim_bloc),dtype=bool)))

  # CROSS-VALIDATION
  for i_fold in range(folds):
    idx_ant=np.roll(idx_ant,nr_test)
    idx_test=np.logical_not(idx_ant)
    poze_ant=all_img[idx_ant,:]
    lbl_ant=all_lbl[idx_ant,:]
    poze_test=all_img[idx_test,:]
    lbl_test=all_lbl[idx_test,:]

    # TRAINING SETUP
    for i_step in range(steps):
      idx_grad=np.roll(idx_grad,i_step*dim_bloc)
      poze_grad=poze_ant[idx_grad,:]
      lbl_grad=lbl_ant[idx_grad,:]
      # Train
      sess.run(train_step, feed_dict={x: poze_grad, y_: lbl_grad})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    rec_rate[i_fold]=sess.run(accuracy, feed_dict={x: poze_test,y_: lbl_test})

  print(np.mean(rec_rate)*100)
  print(str(time.time() - clin)+' secs')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

