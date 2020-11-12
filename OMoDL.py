#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:43:01 2020

@author: apramanik
"""

"""
Creating single-channel KDSLR model to be trained.
"""


import tensorflow as tf
from os.path import expanduser
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)

def convLayer(x, szW,i): #create convolution layer
    with tf.name_scope('layers'):
        with tf.variable_scope('lay'+str(i)):
            W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
            y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME',data_format='NHWC')
            if i!=5 and i!=10:
                y=tf.nn.relu(y)
    return y


def fivelkx(inp,features): #CNN for x-gradient of k-space
    inp=tf.squeeze(inp,axis=1)
    with tf.name_scope('Unetkx'):
        x=convLayer(inp,(3,3,2,features),1)
        x=convLayer(x,(3,3,features,features),2)
        x=convLayer(x,(3,3,features,features),3)
        x=convLayer(x,(3,3,features,features),4)
        x=convLayer(x,(3,3,features,2),5)
        x=inp+x
    x=tf.expand_dims(x,axis=1)
    return x

def fivelky(inp,features): #CNN for y-gradient of k-space
    inp=tf.squeeze(inp,axis=1)
    with tf.name_scope('Unetky'):
        x=convLayer(inp,(3,3,2,features),6)
        x=convLayer(x,(3,3,features,features),7)
        x=convLayer(x,(3,3,features,features),8)
        x=convLayer(x,(3,3,features,features),9)
        x=convLayer(x,(3,3,features,2),10)
        x=inp+x
    x=tf.expand_dims(x,axis=1)
    return x


def r2c(x): #convert multi-channel data from real to complex
    re,im=tf.split(x,[1,1],-1)
    x=tf.complex(re,im)
    return x

def c2r(x): #convert multi-channel data from complex to real
    x=tf.concat([tf.real(x),tf.imag(x)],-1)
    return x

def grad(x,gr): #compute gradient along an axis
    x=c2r(r2c(x)*gr)
    return x

def gradh(x,gr):
    x=c2r(r2c(x)*tf.conj(gr)) #compute hermitian of gradient operation
    return x

def dc(rhs, mask, lam1,Glhs): #data consistency block
    lam1=tf.complex(lam1,0.)
    lhs=mask+(lam1*Glhs)
    rhs=r2c(rhs)
    output=tf.div(rhs,lhs)
    output=c2r(output)
    return output





def makeModel(atb,jwx,jwy,mask,K):
    """
    This is the main function that creates the model.
    atb: The undersampled k-space 
    jwx: Gradient weighting along x-axis
    jwy: Gradient weighting along y-axis
    mask: Undersampling mask
    K: Number of iterations of the model
    """
    scale=tf.complex(tf.sqrt(256.0*170.0),0.0)
    Glhs=(jwx*tf.conj(jwx))+(jwy*tf.conj(jwy))
    out={}
    out['dc0']=atb
    features=96
    with tf.name_scope('UNET'):
        with tf.variable_scope('Wts',reuse=tf.AUTO_REUSE):
            for i in range(1,K+1):
                j=str(i)
                out['dwkx'+j]=gradh(fivelkx(grad(out['dc'+str(i-1)],jwx),features),jwx)
                out['dwky'+j]=gradh(fivelky(grad(out['dc'+str(i-1)],jwy),features),jwy)
                lam1=1.0
                rhs=atb + out['dwkx'+j] + out['dwky'+j]
                out['dc'+j]=dc(rhs,mask,lam1,Glhs)
    outf=r2c(out['dc'+str(K)])
    outf=tf.squeeze(outf,axis=-1)
    outf=tf.squeeze(outf,axis=1)
    outf=tf.signal.fftshift(tf.ifft2d(tf.signal.ifftshift(outf,axes=(-2,-1))),axes=(-2,-1))*scale
    outf=tf.abs(outf)
    return outf

     

