# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:18:23 2022

@author: lgw
"""
import numpy as np
import math
from scipy import integrate

def f1(x,mid_sigma): 
  y=2/(x**2)*f_w(x,mid_sigma)
  return y
def f2(x,mid_sigma):
  y=4*(1/(x**2)+1/(x**4))*f_w(x,mid_sigma)
  return y
def f_w(x,mid_sigma):
  t = x-mid_sigma
  y= math.exp(-1e3*t)
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
    mid_SNR = (SNR1+SNR2)/2
    mid_sigma = np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(mid_SNR/10)))
  
    #weight_coefficient for valid density
    tmp,_ = integrate.quad(f_w,sigma1,sigma2,args=(mid_sigma))
    weight_coefficient = 1/tmp
    tmp_mean,_ = integrate.quad(f1, sigma1,sigma2,args=(mid_sigma))
    new_mean = weight_coefficient*tmp_mean
    tmp_variance,_ = integrate.quad(f2, sigma1,sigma2,args=(mid_sigma))
    new_variance  = weight_coefficient*tmp_variance- new_mean**2
    print(new_mean,new_variance)
    # generate random codewords
    sigma = np.sqrt(new_variance)
    noise *= sigma 
    noise +=new_mean
    training_data = noise
    return training_data,training_data_labels
