# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:11:35 2021

@author: Administrator
"""
import numpy as np
from scipy import integrate

def f1(x): 
    y=2/(x**2)
    return y
def f2(x):
    y=4*(1/(x**2)+1/(x**4))
    return y

def training_data_generating(code,SNRs,max_frame):
    #retrieving global paramters of the code
    n = code.check_matrix_column
    k = code.k 
    training_data_labels = np.zeros((max_frame,n),dtype=np.int64)
    noise = np.random.randn(max_frame,n)
    #starting off the data generating process
    SNR1 = SNRs[0]
    SNR2 = SNRs[1]
    sigma1 =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR1/10)))
    sigma2 =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR2/10)))
    new_mean,err1 = integrate.quad(f1, sigma1,sigma2)/(sigma2-sigma1)
    new_variance,err2 = integrate.quad(f2, sigma1,sigma2)/(sigma2-sigma1)- new_mean**2
    # generate random codewords
    sigma = np.sqrt(new_variance)
    noise *= sigma 
    noise +=new_mean
    training_data = noise
    return training_data,training_data_labels
