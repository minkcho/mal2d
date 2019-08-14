#!/usr/bin/env python
# coding: utf-8

# In[1]:


###################
# data processing #
###################
import pandas as pd
import fnmatch
import os
import math
import numpy as np
import sys

byte_size = 256*256

if len(sys.argv) == 2:
    byte_size = int(sys.argv[1]) * int(sys.argv[1]) 

base_dir = '/home/minkcho/proj/malware/'
filename = base_dir + 'vir_sample_' + str(byte_size) + '_kisa_2c.npz'
np_files = np.load(filename)
x_train = np_files['x_train']
y_train = np_files['y_train']
x_valid = np_files['x_valid']
y_valid = np_files['y_valid']
print(np_files.files)
print(sum(y_train))
print(sum(y_valid))
print(x_train.shape)
print(x_valid.shape)

file_cnt = x_train.shape[0]
input_data_size = x_train.shape[1]


# In[2]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[3]:


from keras.models import Sequential
from keras.layers import add, GlobalAveragePooling2D, Flatten, Embedding, Dense, Activation, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

from keras import regularizers
from keras import backend as K
K.set_image_dim_ordering('tf')

input_dim = 256
embedding_size = 8


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization( axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def create_model():
    # 2d conv model
    I0 = Input(batch_shape=(None, input_data_size, input_data_size, 1), name='Input')
    
    bn_axis = 3

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(I0)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1000, activation='relu', name='fc1000')(x)
    x = Dense(1, activation='sigmoid', name='fc1')(x)
    
    model = Model(inputs=[I0],outputs=[x])

    sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy']) #, weight_pred])
    
    return model, 'resnet50'


# In[4]:


import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

def evalute_model(model, threshold=0.9):      
    filename = base_dir + 'train_' + str(byte_size) + '_ms_2c.npz'
    test = np.load(filename)
    x_test = np.concatenate((test['x_train'], test['x_valid']), axis=0)
    y_test = np.concatenate((test['y_train'], test['y_valid']), axis=0)

    my_size = len(x_test)
    result_sum = 0
    result = model.predict(x=x_test)
    result_sum = np.sum(result > threshold)

    return result_sum - my_size    

def test_model(model, threshold=0.9):   
    filename = base_dir + 'renameAll_' + str(byte_size) + '_msorg_2c.npz'
    test = np.load(filename)
    x_test = np.concatenate((test['x_train'], test['x_valid']), axis=0)
    y_test = np.concatenate((test['y_train'], test['y_valid']), axis=0)

    my_size = len(x_test)
    result_sum = 0
    result = model.predict(x=x_test)
    result_sum = np.sum(result < threshold)
    
    return result_sum - my_size    


def check_roc(model, threshold=0.5):   
    filename = base_dir + 'train_' + str(byte_size) + '_ms_2c.npz'
    test = np.load(filename)
    x_test = np.concatenate((test['x_train'], test['x_valid']), axis=0)
    y_test = np.concatenate((test['y_train'], test['y_valid']), axis=0)

    filename = base_dir + 'renameAll_' + str(byte_size) + '_msorg_2c.npz'
    test = np.load(filename)
    x_test = np.concatenate((x_test, test['x_train'], test['x_valid']), axis=0)
    y_test = np.concatenate((y_test, test['y_train'], test['y_valid']), axis=0)
    
    my_size = len(x_test)
    result_sum = 0

    m_result = model.predict(x=x_test) 
    fpr, tpr, _ = roc_curve(y_test, m_result)
    roc_auc = auc(fpr, tpr)

    y_pred_bool = (m_result > threshold)    
    print(classification_report(y_test, y_pred_bool, digits=4))
    
    return roc_auc, fpr, tpr

def verify_the_trained_model(model, threshold=0.5):      
    filename = base_dir + 'vir_sample_' + str(byte_size) + '_kisa_2c.npz'
    test = np.load(filename)
    x_test = np.concatenate((test['x_train'], test['x_valid']), axis=0)
    y_test = np.concatenate((test['y_train'], test['y_valid']), axis=0)

    my_size = len(x_test)
    
    m_result = model.predict(x=x_test)
    
    fpr, tpr, _ = roc_curve(y_test, m_result)
    roc_auc = auc(fpr, tpr)
    
    y_pred_bool = (m_result > threshold)    
    print(classification_report(y_test, y_pred_bool, digits=4))

    return roc_auc, fpr, tpr        


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

seed = 7
numpy.random.seed(seed)

nb_epoch = 50
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model, model_name = create_model()
K.set_learning_phase(1)
model.fit(x_train, y_train, epochs=nb_epoch, validation_data=(x_valid, y_valid), batch_size=100, callbacks=[es])


# In[ ]:


print (model_name, '# of params', model.count_params())

K.set_learning_phase(0)

r_roc_auc, r_fpr, r_tpr = verify_the_trained_model(model, 0.5)
print(r_roc_auc)
'''
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
lw = 2
plt.plot(r_fpr, r_tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % r_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ' + model_name)
plt.legend(loc="lower right")
plt.show()
'''

# In[ ]:


print (model_name, '# of params', model.count_params())
K.set_learning_phase(0)

fail_e = evalute_model(model, 0.5)
fail_t = test_model(model, 0.5)
print (model_name, fail_e, fail_t)
print (model_name, (10868 + fail_e)/10868.0, (839 + fail_t) / 839.0)


# In[ ]:


roc_auc, fpr, tpr = check_roc(model)
print(roc_auc)
'''
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ' + model_name)
plt.legend(loc="lower right")
plt.show()
'''

