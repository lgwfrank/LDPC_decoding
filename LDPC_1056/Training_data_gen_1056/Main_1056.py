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
import numpy as np
import sys
import os
# Run as follows:
np.set_printoptions(precision=3)
import fill_matrix_info as Fill_matrix
import globalmap as GL
import data_generating as Data_gen
# python main.py 1.5 3 16 1e4   wimax_1056_0.83.alist            
#command line arguments

#sys.argv = "3.0 3.6 16 4000 wimax_1056_0.83.alist".split()
GL.set_map('snr_lo', float(sys.argv[1]))
GL.set_map('snr_hi', float(sys.argv[2]))
GL.set_map('batch_size', int(sys.argv[3]))
GL.set_map('training_batch_number', int(sys.argv[4]))
GL.set_map('H_filename', sys.argv[5])

# setting global parameters
H_filename=GL.get_map('H_filename')
batch_size = GL.get_map('batch_size')

code = Fill_matrix.Code.load_code(H_filename)
GL.set_map('code_parameters', code)

#training setting
#retrieving global paramters of the code
n = code.check_matrix_column
train_batch = GL.get_map('training_batch_number')

snr_lo = GL.get_map('snr_lo')
snr_hi = GL.get_map('snr_hi')
SNRs = [snr_lo,snr_hi]

    
def get_tfrecords_example(feature, label):
  tfrecords_features = {}
  feat_shape = feature.shape
  tfrecords_features['feature'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
  tfrecords_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
  tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
  return tf.train.Example(features = tf.train.Features(feature = tfrecords_features))
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
  feats,labels = data
  tfrecord_wrt = tf.io.TFRecordWriter(out_filename)
  ndatas = len(labels)
  for inx in range(ndatas):
    exmp = get_tfrecords_example(feats[inx], labels[inx])
    exmp_serial = exmp.SerializeToString()
    tfrecord_wrt.write(exmp_serial)
  tfrecord_wrt.close()
#create directory if not existence      
if not os.path.exists( './data/snr'+str(snr_lo)+'-'+str(snr_hi)):
  os.makedirs( './data/snr'+str(snr_lo)+'-'+str(snr_hi)) 
nDatas_train = train_batch*batch_size  
#generating training data
train_data,train_labels = Data_gen.training_data_generating(code, SNRs,nDatas_train)   
# make training set file
data = (train_data,train_labels)
file_dir = './data/snr'+str(snr_lo)+'-'+str(snr_hi)+'/ldpc-train.tfrecord'
make_tfrecord(data, out_filename=file_dir)
    
print("Data for training are generated successfully!")