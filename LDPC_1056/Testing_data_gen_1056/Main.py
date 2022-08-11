# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:31:50 2021

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:17:48 2021

@author: Administrator
"""
import tensorflow as tf

import random
import numpy as np
import sys
import os
# Run as follows:
np.set_printoptions(precision=3)
import fill_matrix_info as Fill_matrix
import globalmap as GL
import data_generating as Data_gen
# python main.py 2.0 3.5 7 100 1000 wimax_1056_0.83.alist
#          1  2 3  4   5   6                                  
#command line arguments
GL.set_map('snr_lo', float(sys.argv[1]))
GL.set_map('snr_hi', float(sys.argv[2]))
GL.set_map('snr_num', int(sys.argv[3]))
GL.set_map('batch_size', int(sys.argv[4]))

GL.set_map('testing_batch_number', float(sys.argv[5]))
GL.set_map('H_filename', sys.argv[6])



# setting global parameters

GL.set_map('ALL_ZEROS_CODEWORD_TESTING', True)

GL.set_map('SIGMA_SCALING_TESTING', True)

GL.set_map('portion_dis', '0.01 0.01 0.01 0.03 0.06 0.1 0.1 0.25 0.25 0.5 0.5 0.75 1') #for generating testing data of various sizes

H_filename=GL.get_map('H_filename')


code = Fill_matrix.Code.load_code(H_filename)
GL.set_map('code_parameters', code)

#training setting
#retrieving global paramter values of the code
n = code.check_matrix_column

batch_size = GL.get_map('batch_size')
test_batch = GL.get_map('testing_batch_number')
snr_lo = GL.get_map('snr_lo')
snr_hi = GL.get_map('snr_hi')
snr_num = GL.get_map('snr_num')
SNRs = np.linspace(snr_lo,snr_hi, snr_num)
   
def get_tfrecords_example(feature, label):
    tfrecords_features = {}
    feat_shape = feature.shape
    tfrecords_features['feature'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    tfrecords_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
    return tf.train.Example(features = tf.train.Features(feature = tfrecords_features))
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
    feats, labels = data
    tfrecord_wrt = tf.io.TFRecordWriter(out_filename)
    ndatas = len(labels)
    for inx in range(ndatas):
        exmp = get_tfrecords_example(feats[inx], labels[inx])
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)
    tfrecord_wrt.close()
#create directory if not existence   
if not os.path.exists('./data/testing'):
    os.makedirs('./data/testing')

string_list = (GL.get_map('portion_dis')).split()
portion_list = np.array(string_list).astype(float) 
i = 0
for SNR in SNRs:
    nDatas_test = test_batch*batch_size      
    percentage = portion_list[i]
    # make test set
    max_frame = int(percentage*nDatas_test)
    test_data,test_labels = Data_gen.testing_data_generating(code, SNR,max_frame )
    data = (test_data,test_labels)
    make_tfrecord(data, './data/testing/test_'+str(round(SNR,2))+'dB.tfrecord')
    i +=1
print("Data for testing are generated successfully!")

