from __future__ import print_function
import os
import sys
import time
import random
from glob import glob
import numpy as np
import scipy.io
from scipy import misc
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import tensorflow as tf
import pdb
from parse_opt import get_arguments, get_arguments_auto
from deeplab_resnet import HyperColumn_Deeplabv2#, read_data_list
import runway


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

sess_sss = None
model_sss = None



def calc_pca(feature):
    # Filter out super high numbers due to some instability in the network
    feature[feature>5] = 5
    feature[feature<-5] = -5
    #### Missing an image guided filter with the image as input
    ##
    ##########
    # change to double precision
    feature = np.float64(feature)
    # retrieve size of feature array
    shape = feature.shape
    [h, w, d] = feature.shape
    # resize to a two-dimensional array
    feature = np.reshape(feature, (h*w,d))
    # calculate average of each column
    featmean = np.average(feature,0)
    onearray = np.ones((h*w,1))
    featmeanarray = np.multiply(np.ones((h*w,1)),featmean)
    feature = np.subtract(feature,featmeanarray)
    feature_transpose = np.transpose(feature)
    cover = np.dot(feature_transpose, feature)
    # get largest eigenvectors of the array
    val,vecs = eigs(cover, k=3, which='LI')
    pcafeature = np.dot(feature, vecs)
    pcafeature = np.reshape(pcafeature,(h,w,3))
    pcafeature = np.float64(pcafeature)
    return pcafeature


def normalise_0_1(feature):
    max_value = np.amin(feature)
    min_value = np.amax(feature)
    subtract = max_value - min_value
    for i in range(0,3):
        feature[:,:,i] = feature[:,:,i] - np.amin(feature[:,:,i])
        feature[:,:,i] = feature[:,:,i] / np.amax(feature[:,:,i])
    return feature


def sss_read_img(img_contents, input_size, img_mean): # optional pre-processing arguments
    img = tf.convert_to_tensor(np.array(img_contents), dtype='uint8')
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    img -= img_mean
    if input_size is not None:
        h, w = input_size
        # Randomly scale the images and labels.
        newshape = tf.squeeze(tf.stack([h, w]), squeeze_dims=[1])
        img2 = tf.image.resize_images(img, newshape)
    else:
        img2 = tf.image.resize_images(img, tf.shape(img)[0:2,]*2)
    return img2, img


@runway.setup(options={'model_name':runway.text})
def setup(options):
    global sess_sss, model_sss
    args = get_arguments_auto()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess_sss = tf.Session(config=config)
    sess_sss.as_default()
    model_sss = HyperColumn_Deeplabv2(sess_sss, args)
    model_sss.load('model')
    return model_sss


@runway.command('convert', inputs={'image': runway.image}, outputs={'output': runway.image})
def convert(model_sss, inp):
    global sess_sss
    img = np.array(inp['image'])
    padsize = 50
    _, ori_img = sss_read_img(img, input_size = None, img_mean = IMG_MEAN)
    pad_img = tf.pad(ori_img, [[padsize, padsize], [padsize, padsize], [0,0]], mode='REFLECT')
    cur_embed = model_sss.test(pad_img.eval(session=sess_sss))  #
    cur_embed = np.squeeze(cur_embed)
    pca_feature = calc_pca(cur_embed)
    normalise_feature = normalise_0_1(pca_feature)
    img = (255 * normalise_feature).astype('uint8')
    output = img[padsize:-padsize, padsize:-padsize, :]
    return dict(output=output)


if __name__ == '__main__':
    runway.run()

