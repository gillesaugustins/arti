# RCP209 Project - Implementation of Image Style Transfer
# Gilles Augustins - 03/07/2017

from scipy.misc import imsave, imshow
import numpy as np
from keras.applications import vgg16, vgg19
from keras import backend as K
from keras.preprocessing import image
from utils import *
from scipy.optimize import fmin_l_bfgs_b

# dimensions of the generated pictures for each filter.
img_width, img_height = 128, 128

show_pictures = 1

# Optimization algo
# Turn BFGS to 0 for using gradient descent algo 
BFGS = 1
# Parameters of BFGS 
bfgs_iter = 10
# Parameters of the gradient descent
gdsc_step = 0.00001
gdsc_iter = 50

############### Model build ###############################
# build the VGG19 network with ImageNet weights, no fully connected layers
model = vgg19.VGG19(weights='imagenet', include_top=False)
model.summary()
outputs_dict = dict([layer.name, layer.output] for layer in model.layers)

# Content Image
img = image.load_img('elephant.jpg', target_size=(img_width, img_height))
if (show_pictures == 1):
    imshow(img)
imsave("original_content.png", img)
content_img = preprocess_image(img)


############### Response from Content image  ###############################

content_layers =  ['block1_conv1', 'block1_conv2']
content_layers += ['block2_conv1', 'block2_conv2']
content_layers += ['block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4']
content_layers += ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4']
content_layers += ['block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']

for layer_name in content_layers:
    get_response = K.function([model.input], [outputs_dict[layer_name]])
    content_feature = K.constant(get_response([content_img])[0])

    # Gradients and loss 
    layer_output = outputs_dict[layer_name][0]
    loss = K.sum(K.square(content_feature - layer_output))
    grads = K.gradients(loss, model.input)[0]
    func_loss = K.function([model.input], [loss])
    func_grads = K.function([model.input], [grads])

    # we start from a white noise image with some random noise
    gray_img = np.random.random((1, img_width, img_height, 3)) # Channel Last

    if BFGS == 1:
        # Gradients for white noise image using min_l_bfgs_b 
        def fn_loss (x):
            x = x.reshape((1, img_width, img_height, 3))
            l = func_loss([x])[0]
            return l.flatten().astype('float64')
        
        def fn_grads (x):
            x = x.reshape((1, img_width, img_height, 3))
            g = func_grads([x])[0]
            return g.flatten().astype('float64')
        
        
        for i in range (0, bfgs_iter):
            print ('iteration ',i)
            gray_img, min_val, info = fmin_l_bfgs_b(fn_loss, gray_img, fn_grads, maxfun=20)
            print('Current loss value:', min_val.sum())
            
        rec_img = gray_img.reshape((img_width, img_height, 3))
        bfgs='bfgs_'

    else: 
        # Gradient descent 
        for i in range (gdsc_iter):
            loss_value = func_loss([gray_img])[0]
            grads_value = func_grads([gray_img])[0]
            gray_img -= grads_value * gdsc_step
            print('[', layer_name, ']', 'Iter', i, ': Current loss value =', loss_value.sum())
        rec_img = gray_img[0]
        bfgs=''

    # decode the resulting input image
    reconstructed_img = deprocess_image(rec_img)
    if (show_pictures == 1):
        imshow(reconstructed_img)
    img_name = 'img_reconst_'+bfgs+layer_name+'.png'
    imsave(img_name, reconstructed_img)
