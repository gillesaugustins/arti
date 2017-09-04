# RCP209 Project - Implementation of Image Style Transfer
# Gilles Augustins - 03/07/2017

import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import vgg19

def preprocess_image(x):
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = vgg19.preprocess_input(x)
    return x

def deprocess_image(x):
    # Mean values of the ImageNet dataset
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    # BGR -> RGB
    x = x[:,:,::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def gram_matrix (x):
    assert K.ndim(x)==3
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram = K.dot(features, K.transpose(features)) 
    return gram

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)
