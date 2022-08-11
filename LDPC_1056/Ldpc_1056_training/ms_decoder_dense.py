# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
import pickle
import numpy as np

class Decodering_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = Decoder_Layer()
    def call(self,inputs,Lth_layer): 
        soft_output,loss = self.layer(inputs,Lth_layer)
        return soft_output,loss
    def acquire_failed_frames(self,FER_data):
        where_bool = tf.not_equal(FER_data, False)
        indices = tf.where(where_bool)
        return indices
    
    def get_eval(self,soft_output, labels):
        tmp = tf.cast((soft_output >= 0),tf.bool)
        label_bool = tf.cast(labels, tf.bool)
        err_batch = tf.equal(tmp, label_bool)
        FER_data = tf.reduce_any(err_batch,1)
        index = self.acquire_failed_frames(FER_data)
        BER_data = tf.reduce_sum(tf.cast(err_batch, tf.int64))
        FER = tf.math.count_nonzero(FER_data)/self.layer.train_batch_size
        BER = BER_data/(self.layer.train_batch_size*self.layer.code.check_matrix_column)
        return FER, BER,index    
    
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.L  = GL.get_map('weighted_L')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.train_batch_size = GL.get_map('train_batch_size')
        self.unit_batch_size = GL.get_map('unit_batch_size')
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape):       
        if GL.get_map('selected_decoder_type') in ['VDHD','VDHS','VDSL']:
          self.shared_bit_weight = []
          for i in range(self.num_iterations):
            self.shared_bit_weight.append(self.add_weight(name='decoder_input_weight_per_iteration'+str(i),shape=[self.code.check_matrix_column],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))    
        if GL.get_map('selected_decoder_type') in ['VSHD','VSHS','VSSL','VS']:
            self.shared_bit_weight = self.add_weight(name='decoder_input_weight',shape=[1,self.code.check_matrix_column],trainable=True,initializer=tf.keras.initializers.Constant(0.542))    
        if GL.get_map('selected_decoder_type') in ['SLHD','SLHS','SLSL']:
          self.shared_bit_weight = []
          for i in range(self.num_iterations):          
            self.shared_bit_weight.append(self.add_weight(name='decoder_input_weight_per_layer'+str(i),shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))                   
        if GL.get_map('selected_decoder_type') in ['VDHD','VSHD','SLHD']:
          self.shared_check_weight = []
          for i in range(self.num_iterations): 
            self.shared_check_weight.append(self.add_weight(name='decoder_check_weight_per_iteration'+str(i),shape=[self.code.check_matrix_row],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))    
        if GL.get_map('selected_decoder_type') in ['VDHS','VSHS','SLHS']:
            self.shared_check_weight = self.add_weight(name='decoder_check_weight',shape=[self.code.check_matrix_row,1],trainable=True,initializer=tf.keras.initializers.Constant(0.542))    
        if GL.get_map('selected_decoder_type') in ['VDSL','VSSL','SLSL','SNNMS']:
          self.shared_check_weight = []
          for i in range(self.num_iterations): 
            self.shared_check_weight.append(self.add_weight(name='decoder_check_weight_per_layer'+str(i),shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))   
        if GL.get_map('selected_decoder_type') == 'NMS':
            self.shared_check_weight = self.add_weight(name='decoder_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(-1.07))     
        
        if GL.get_map('selected_decoder_type') in ['SVMS']:
            self.shared_bit_weight = self.add_weight(name='decoder_variable_weight_per_block',shape=[1,self.code.max_chk_degree],trainable=True,initializer=tf.keras.initializers.Constant(0.542))    
            self.shared_check_weight = self.add_weight(name='decoder_check_weight_per_block',shape=[self.code.max_var_degree,1],trainable=True,initializer=tf.keras.initializers.Constant(0.542))    
        
        if GL.get_map('selected_decoder_type') in ['ANNMS']:
            self.bit_weight = []
            self.vc_matrix_weight = []
            self.cv_matrix_weight = []
            for i in range(self.num_iterations):
             self.bit_weight.append(self.add_weight(name='decoder_bit_weight_per_block'+str(i),shape=[self.code.check_matrix_column],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))   
             self.vc_matrix_weight.append(self.add_weight(name='decoder_vc_weight_per_block'+str(i),shape=[self.code.check_matrix_row,self.code.check_matrix_column],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))    
             self.cv_matrix_weight.append(self.add_weight(name='decoder_cv_weight_per_block'+str(i),shape=[self.code.check_matrix_row,self.code.check_matrix_column],trainable=True,initializer=tf.keras.initializers.Constant(0.542)))    

 
 # Code for model call (handles inputs and returns outputs)
    def call(self, inputs,Lth_layer):
        self.layer_iterations = Lth_layer+1
        soft_input = inputs[0]
        labels = inputs[1]     
        early_stopping_indicator = self.early_stopping(soft_input,labels)
        if early_stopping_indicator:
            return soft_input,self.calculation_loss_two(soft_input,labels)
        bp_result = self.belief_propagation_op(soft_input,labels)  
        return bp_result[1],bp_result[4] 
                
# builds a belief propagation TF graph
    def belief_propagation_op(self,soft_input, labels):

        if GL.get_map('selected_decoder_type') in ['SVMS']:
            shared_weight_list = []                         
            counter = 0  
            flag = self.code.H[0,0]                 
            if flag == 0:
                for j in range(self.code.check_matrix_row):
                    if self.code.H[0,j] != flag:
                        counter = counter+1
                    index = counter % self.code.max_chk_degree                        
                    shared_weight_list.append(self.shared_bit_weight[0,index])                          
            else:
                counter = -1
                for j in range(self.code.check_matrix_row):
                    if self.code.H[0,j] == flag:
                        counter = counter+1                      
                    shared_weight_list.append(self.shared_bit_weight[0,counter])     
            self.bit_weight = shared_weight_list
        if GL.get_map('selected_decoder_type') in ['SVMS']:
            shared_weight_list = []                         
            counter = 0  
            flag = self.code.H[0,0]                 
            if flag == 0:
                for j in range(self.code.check_matrix_column):
                    if self.code.H[j,0] != flag:
                        counter = counter+1
                    index = counter % self.code.max_var_degree                        
                    shared_weight_list.append(self.shared_check_weight[index,0])                          
            else:
                counter = -1
                for j in range(self.code.check_matrix_column):
                    if self.code.H[j,0] == flag:
                        counter = counter+1                      
                    shared_weight_list.append(self.shared_check_weight[counter,0])     
            self.check_weight = shared_weight_list

        return tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_input, # soft input for this iteration
                soft_input,  # soft output for this iteration
                tf.constant(0,dtype=tf.int32), # iteration number
                tf.zeros([self.unit_batch_size,self.code.check_matrix_row,self.code.check_matrix_column],dtype=tf.float32)    ,# cv_matrix
                tf.constant(0.0,dtype=tf.float32), # loss
                labels,
                False
            ]
            )
            
    # compute messages from variable nodes to check nodes
    def compute_vc(self,cv_matrix, soft_input,iteration):
        normalized_tensor = 1.0
        check_matrix_H = tf.cast(self.code.H,tf.float32)
        if GL.get_map('selected_decoder_type') in ['VDHD','VDHS','VDSL']:
          normalized_tensor = tf.nn.softplus(self.shared_bit_weight[iteration])        
        if GL.get_map('selected_decoder_type') in ['VSHD','VSHS','VSSL','VS']:
          normalized_tensor = tf.nn.softplus(self.shared_bit_weight)           
        if GL.get_map('selected_decoder_type') in ['SLHD','SLHS','SLSL']:
          normalized_tensor = tf.nn.softplus(self.shared_bit_weight[iteration]) 
        if GL.get_map('selected_decoder_type') == 'SVMS':
          normalized_tensor = tf.nn.softplus(self.bit_weight) 
        if GL.get_map('selected_decoder_type') in ['ANNMS']:
          normalized_tensor = tf.nn.softplus(self.bit_weight[iteration]) 
          normalized_tensor = tf.expand_dims(normalized_tensor,0)                            
          weight_matrix = check_matrix_H*tf.nn.softplus(self.vc_matrix_weight[iteration])
          expand_weight_matrix = tf.expand_dims(weight_matrix,0)
          cv_matrix = cv_matrix*expand_weight_matrix 

        soft_input_weighted = soft_input*normalized_tensor           
        temp = tf.reduce_sum(cv_matrix,1)                        
        temp = temp+soft_input_weighted
        temp = tf.expand_dims(temp,1)
        temp = temp*check_matrix_H
        vc_matrix = temp - cv_matrix
        return vc_matrix  
    # compute messages from check nodes to variable nodes
    def compute_cv(self,vc_matrix,iteration):
        normalized_tensor = 1.0
        check_matrix_H = self.code.H
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,-1e30-1,0.)
        back_matrix = tf.expand_dims(back_matrix,0)
        vc_matrix_abs = tf.abs(vc_matrix)
        vc_matrix_abs_clip = tf.clip_by_value(vc_matrix_abs, 0, 1e30)
        vc_matrix_abs_minus = -tf.abs(vc_matrix_abs_clip)
        vc_decision_matrix = vc_matrix_abs_minus+back_matrix
        min_submin_info = tf.nn.top_k(vc_decision_matrix,k=2)
        min_column_matrix = -min_submin_info[0][:,:,0]
        min_column_matrix = tf.expand_dims(min_column_matrix,2)
        min_column_matrix = min_column_matrix*check_matrix_H
        second_column_matrix = -min_submin_info[0][:,:,1]
        second_column_matrix = tf.expand_dims(second_column_matrix,2)
        second_column_matrix = second_column_matrix*check_matrix_H
        result_matrix = tf.where(vc_matrix_abs_clip>min_column_matrix,min_column_matrix,second_column_matrix)
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp1 = tf.reduce_prod(vc_matrix_sign,2)
        temp1 = tf.expand_dims(temp1,2)
        transition_sign_matrix = temp1*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        cv_no_weight =  result_matrix*tf.stop_gradient(result_sign_matrix)
        if GL.get_map('selected_decoder_type') in ['VDHD','VSHD','SLHD']:
          normalized_tensor = tf.nn.softplus(tf.reshape(self.shared_check_weight[:,iteration],[-1,1]))
        if GL.get_map('selected_decoder_type') in ['VDHS','VSHS','SLHS']:
          normalized_tensor = tf.nn.softplus(self.shared_check_weight)
        if GL.get_map('selected_decoder_type') in ['VDSL','VSSL','SLSL','SNNMS']:
          normalized_tensor = tf.nn.softplus(self.shared_check_weight[iteration])
        if GL.get_map('selected_decoder_type') == 'NMS':
          normalized_tensor = tf.nn.softplus(self.shared_check_weight)
        if GL.get_map('selected_decoder_type') == 'SVMS':
          normalized_tensor = tf.nn.softplus(self.check_weight) 
        if GL.get_map('selected_decoder_type') in ['ANNMS']:
          check_matrix_H = tf.cast(self.code.H,tf.float32) 
          weight_matrix = check_matrix_H*tf.nn.softplus(self.cv_matrix_weight[iteration]) 
          normalized_tensor = tf.expand_dims(weight_matrix,0)
        cv_matrix = cv_no_weight*normalized_tensor           
        return cv_matrix
  
    def calculation_loss_one(self,soft_output,labels,loss,iteration):
          #cross entroy
          labels = tf.cast(labels,tf.float32)
          CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) / self.num_iterations
          
          #minimum squared error 
          soft_prob = tf.sigmoid(-soft_output)
          MSE_loss = tf.reduce_mean(tf.square(soft_prob - labels)) / self.num_iterations
          new_loss = self.L * CE_loss + (1 - self.L) * MSE_loss*100
          #with open('./ckpts/current/'+'VSSL/'+'Output_data_first_iteration','ab') as f:
           # pickle.dump(soft_output.numpy(),f) 
          return loss + new_loss

    def calculation_loss_soft_max(self,soft_output,labels):
          #cross entroy
          labels = tf.cast(labels,tf.float32)
          cost_prob = tf.sigmoid(-soft_output)-labels
          abs_mag = tf.where(soft_output>=0,soft_output,-soft_output)  
          #CE_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cost_prob, labels=cost_prob)) 
          CE_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=cost_prob, labels=cost_prob))

          return CE_loss  

    def softmax(self,x):
        sum_raw = np.sum(np.exp(x), axis=-1)
        x1 = np.ones(np.shape(x))
        for i in range(np.shape(x)[0]):
            x1[i] = np.exp(x[i]) / sum_raw[i]
        return x1

    def calculation_loss_two(self,soft_output,labels):
          #cross entroy
          #CE_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,-soft_output)) 
          labels = tf.cast(labels,tf.float32)
          #CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
          CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels))
          #minimum squared error 
          soft_prob = tf.sigmoid(-soft_output)
          #MSE_loss = tf.reduce_mean(tf.square(soft_prob - labels))
          MSE_loss = tf.reduce_sum(tf.square(soft_prob - labels))
          #soft_syndrome_metric = self.syndrome_cal(soft_output,labels)
          #string_list = self.L.split()
          #portion_list = np.array(string_list).astype(float)
          #new_loss = portion_list[0]*CE_loss+portion_list[1]*MSE_loss*100+portion_list[2]*soft_syndrome_metric
          #new_loss = self.L*CE_loss+(1-self.L)*MSE_loss*100+5*soft_syndrome_metric
          new_loss = self.L*CE_loss+(1-self.L)*MSE_loss*100 
          return new_loss
    def syndrome_cal(self,soft_output,labels):
        check_matrix_H = self.code.H
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,101.,0.)
        soft_output_clip = tf.clip_by_value( soft_output, -100, 100)
        info_matrix = tf.expand_dims(soft_output_clip,1)*check_matrix_H
        decision_matrix = tf.abs(info_matrix)+tf.expand_dims(back_matrix,0)
        decision_matrix = tf.reduce_min(decision_matrix,2) 
        #operands sign processing       
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        sign_info = supplement_matrix +tf.sign(info_matrix)
        sign_matrix = tf.reduce_prod(sign_info,2)
        result_syndrome = 1-tf.stop_gradient(sign_matrix)*decision_matrix
        syndrome_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=result_syndrome,labels=tf.zeros_like(result_syndrome))
        return tf.reduce_mean(syndrome_entropy)
    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input):
        temp = tf.reduce_sum(cv_matrix,1)
        soft_output = temp+soft_input

        return soft_output  
    def early_stopping(self,soft_output,labels):
        ones = tf.ones_like(soft_output,dtype=tf.int64)
        zeros = tf.zeros_like(soft_output,dtype=tf.int64)
        temp = tf.where(soft_output>=0,zeros,ones)
        temp = tf.equal(temp,labels)  
        return tf.reduce_all(temp)
    
    def continue_condition(self,soft_input,soft_output,iteration, cv_matrix, loss, labels,early_stopping_indicator):
        condition = ((iteration < self.layer_iterations) and (not early_stopping_indicator))
        return condition
    
    def belief_propagation_iteration(self,soft_input, soft_output, iteration, cv_matrix, loss, labels,early_stopping_indicator):
        # compute vc
        vc_matrix = self.compute_vc(cv_matrix, soft_input,iteration)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix,iteration)      
        # get output for this iteration
        soft_output = self.marginalize(cv_matrix, soft_input)
        iteration += 1
        early_stopping_indicator = self.early_stopping(soft_output,labels)       
        if GL.get_map('Cumulative_loss_indicator'):
            loss = self.calculation_loss_one(soft_output,labels,loss,iteration)
        elif iteration == self.layer_iterations:
            if GL.get_map('Soft_max_indicator'):
              loss = self.calculation_loss_soft_max(soft_output,labels)
            else:
              loss = self.calculation_loss_two(soft_output,labels)
            
        return soft_input, soft_output, iteration, cv_matrix, loss, labels,early_stopping_indicator 
