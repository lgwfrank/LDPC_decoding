# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:11:35 2021

@author: Administrator
"""
import globalmap as GL
import numpy as np
def testing_data_generating(code,SNR,max_frame):
    #retrieving global paramters of the code
    n = code.check_matrix_column
    k = code.k    
    sigma = np.sqrt(1. / (2 * (np.float(k)/np.float(n)) * 10**(SNR/10)))
    testing_data_labels = np.zeros((max_frame,n),dtype=np.int64)
    #starting off the data generating process
    mean = 1     
    channel_information =  np.random.normal(mean,sigma,size=(max_frame,n))

    # Whether to use channel estimation as initialization of input(LLR)
    if GL.get_map('SIGMA_SCALING_TESTING'):
        testing_data= 2.0*channel_information/(sigma*sigma)
    else:
        testing_data= channel_information
    return testing_data,testing_data_labels
