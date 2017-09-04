# RCP209 Project - Implementation of Image Style Transfer
# Gilles Augustins - 03/07/2017

from scipy.misc import imsave, imshow
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from keras.preprocessing import image
from utils import *
from scipy.optimize import fmin_l_bfgs_b

# dimensions of the generated pictures for each filter.
img_width, img_height = 128, 128 

show_pictures = 1

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

# Style Image
img = image.load_img('vangogh.jpg', target_size=(img_width, img_height))
style = 'vg'
if (show_pictures == 1):
    imshow(img)
imsave("original_style.png", img)
style_img = preprocess_image(img)

# Global Parameters
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
style_weights = [0.2,            0.2,            0.2,           0.2,             0.2           ]
#content_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2']
#alpha_list = [1e-4, 1e-5, 1e-6, 1e-7]
content_layers = ['block4_conv2']
alpha_list = [1e-6]
bfgs_iter = 5
beta = 1.0

for content_layer in content_layers:

    for alpha in alpha_list:

        print('Content layer =', content_layer)
        print('Alpha =', str(alpha))

        ############### Response from Content image  ###############################
        get_response = K.function([model.input], [outputs_dict[content_layer]])
        content_feature = K.constant(get_response([content_img])[0])
        
        ############### Response from Style image  ###############################
        style_features = []
        for style_layer in style_layers:
            get_response = K.function([model.input], [outputs_dict[style_layer]])
            style_features.append(K.constant(get_response([style_img])[0]))
        
        ############### Content Loss ###############################
        content_layer_output = outputs_dict[content_layer][0]
        content_loss = K.sum(K.square(content_feature - content_layer_output))
        
        ############### Style Loss ###############################
        style_loss = 0
        i = 0
        for style_layer in style_layers:
            style_layer_output = outputs_dict[style_layer][0]
            style_feature = style_features[i]
            w = style_weights[i]
            l = w * K.sum(K.square(gram_matrix(style_feature[0]) - gram_matrix(style_layer_output)))
            # loss normalization
            M = img_width * img_height
            N = style_layer_output.shape.dims[2].value
            l /= 4*(N**2)*(M**2)
            style_loss += w * l
            i += 1
        
        loss = alpha * content_loss + beta * style_loss
        
        ############### Gradients ###############################
        grads = K.gradients(loss, model.input)[0]
        
        ############### Optimization for white noise image using min_l_bfgs_b 
        func_loss = K.function([model.input], [loss])
        func_grads = K.function([model.input], [grads])
        
        # we start from a white noise image with some random noise
        gray_img = np.random.random((1, img_width, img_height, 3)) # Channel Last
        
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
    
        # decode the resulting input image
        reconstructed_img = deprocess_image(rec_img)
        if (show_pictures == 1):
            imshow(reconstructed_img)
        img_name = 'combination_'+style+str(img_height)+'x'+str(img_width)+'_'+content_layer+'_alpha_'+str(alpha)+'.png'
        imsave(img_name, reconstructed_img)
