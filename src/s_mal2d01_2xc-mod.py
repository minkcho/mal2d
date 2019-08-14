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
filename = 'vir_sample_' + str(byte_size) + '_kisa_2c.npz'
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
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dropout, Dense, Conv1D, Conv2D, Activation, MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling2D, Input, Embedding, Multiply, MaxPooling1D,GlobalAveragePooling2D, GlobalAveragePooling1D

from keras.utils import plot_model
from keras import metrics
from keras.layers.advanced_activations import PReLU

from keras import regularizers
from keras import backend as K

K.set_image_dim_ordering('tf')

def create_model(x = 16, y = 64):   
    x2 = int(x/2)
    x4 = int(x/4)
    y2 = int(y/2)
    y4 = int(y/4)
    y8  = int(y/8)
    y16 = int(y/16)

    inp = Input(batch_shape=(None, input_data_size, input_data_size, 1), name='Input')
    
    c1  = Conv2D( filters= 64, kernel_size=(x, y), strides=(x2, y8), use_bias=True, activation='relu', padding='same')(inp)
    c2  = Conv2D( filters=128, kernel_size=(x, y), strides=(x2, y8), use_bias=True, activation='relu', padding='same')(c1)
    p1  = MaxPooling2D ( strides=(3,3), padding='same' )(c2)
    c3  = Conv2D( filters=192, kernel_size=(x2, y2), strides=(x, y4), use_bias=True, activation='relu', padding='same')(p1)
    c4  = Conv2D( filters=256, kernel_size=(x2, y2), strides=(x, y4), use_bias=True, activation='relu', padding='same')(c3)
    p2  = GlobalMaxPooling2D () (c4)    
    d1  = Dense(256, activation='relu')(p2)
    d2  = Dense(64,  activation='relu')(d1)
    outp= Dense(1, activation='sigmoid')(d2)
        
    basemodel = Model( inp, outp )
    
    basemodel.compile( loss='binary_crossentropy', optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True,decay=1e-6), metrics=['accuracy']) # , weight_pred] )
 
    return basemodel, "mal2d01_c"



# In[7]:


import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

def evalute_model(model, threshold=0.9):   
    filename = base_dir + 'train_' + str(byte_size) + '_ms_2c.npz'
    test = np.load(filename)
    #x_test = np.concatenate((test['x_train'], test['x_valid']), axis=0)
    #y_test = np.concatenate((test['y_train'], test['y_valid']), axis=0)

    my_size = len(test['x_train']) + len(test['x_valid'])
    
    result1 = model.predict(x=test['x_train'])
    result_sum = np.sum(result1 > threshold)
    
    result2 = model.predict(x=test['x_valid'])
    result_sum += np.sum(result2 > threshold)

    return result_sum - my_size    

def test_model(model, threshold=0.9):
    filename = base_dir + 'renameAll_' + str(byte_size) + '_msorg_2c.npz'
    ttest = np.load(filename)

    my_size = len(ttest['x_train']) # + len(ttest['x_valid'])
    print(my_size)
    
    result1 = model.predict(x=ttest['x_train'])
    result_sum = np.sum(result1 < threshold)
    
    return result_sum - my_size  


def check_roc(model, threshold=0.5):   
    filename = base_dir + 'train_' + str(byte_size) + '_ms_2c.npz'
    test = np.load(filename)

    result1 = model.predict(x=test['x_train'])
    result2 = model.predict(x=test['x_valid'])
    y_test1 = test['y_train']
    y_test2 = test['y_valid']

    filename = base_dir + 'renameAll_' + str(byte_size) + '_msorg_2c.npz'
    test = np.load(filename)
    result3 = model.predict(x=test['x_train'])
    y_test3 = test['y_train']

    m_result = np.concatenate((result1, result2, result3), axis=0)
    y_test   = np.concatenate((y_test1, y_test2, y_test3), axis=0)
    
    print('theshold', np.sort(m_result)[2586], np.sort(m_result)[-2586])
    
    my_size = len(m_result)

    fpr, tpr, _ = roc_curve(y_test, m_result)
    roc_auc = auc(fpr, tpr)

    y_pred_bool = (m_result > threshold)    
    print(classification_report(y_test, y_pred_bool, digits=4))
    
    return roc_auc, fpr, tpr

def verify_the_trained_model(model, threshold=0.5):      
    filename = base_dir + 'vir_sample_' + str(byte_size) + '_kisa_2c.npz'
    test = np.load(filename)

    y_test = np.concatenate((test['y_train'], test['y_valid']), axis=0)
   
    result1 = model.predict(x=test['x_train'])
    result2 = model.predict(x=test['x_valid'])
    
    m_result = np.concatenate((result1, result2), axis=0)
    
    fpr, tpr, _ = roc_curve(y_test, m_result)
    roc_auc = auc(fpr, tpr)
    
    y_pred_bool = (m_result > threshold)    
    print(classification_report(y_test, y_pred_bool, digits=4))

    return roc_auc, fpr, tpr        


# In[5]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

seed = 7
numpy.random.seed(seed)

nb_epoch = 50
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model, model_name = create_model(16,8)
K.set_learning_phase(1)
model.fit(x_train, y_train, epochs=nb_epoch, validation_data=(x_valid, y_valid), batch_size=100, callbacks=[es])


# In[8]:


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

# In[9]:


print (model_name, '# of params', model.count_params())
K.set_learning_phase(0)

fail_e = evalute_model(model, 0.5)
fail_t = test_model(model, 0.5)
print (model_name, fail_e, fail_t)
print (model_name, (10868 + fail_e)/10868.0, (839 + fail_t) / 839.0)


# In[10]:


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
