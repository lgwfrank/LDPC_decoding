# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:47:08 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL

class Decodering_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = Decoder_Layer()
    def call(self,inputs): 
        bp_results = self.layer(inputs)
        return bp_results
    
    def get_eval(self,soft_output,early_stopping_indicator):
        tmp = tf.cast((soft_output < 0),tf.int64)
        BER_data = tf.reduce_sum(tmp)
        FER_data = tf.logical_not(early_stopping_indicator)
        index = tf.where(FER_data)
        FER =  tf.reduce_sum(tf.cast(FER_data,tf.int64))/self.layer.unit_batch_size
        BER = BER_data/(self.layer.unit_batch_size*self.layer.code.check_matrix_column)
        return FER,BER,index  
    def show_error_codeword(self,index,soft_output):
      for i in index.numpy().flatten():
        for j in range(len(soft_output[i].numpy())): 
          print(soft_output[i].numpy()[j],end=" ") 
    
class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        #self.list_weights = [0.124,0.663,0.965,0.482,1.055,-1.725,-1.485,-1.147,-1.058,-1.004]
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
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
    def call(self, inputs):
        soft_input = inputs[0]
        labels = inputs[1] 
        early_stopping_indicator = self.early_stopping(soft_input,labels)
        if tf.reduce_all(early_stopping_indicator):
          return soft_input,early_stopping_indicator
        yieldings = self.belief_propagation_op(soft_input,labels,early_stopping_indicator)    
        return yieldings[1],yieldings[5]
                
# builds a belief propagation TF graph
    def belief_propagation_op(self,soft_input,labels,early_stopping_indicator):
        #soft_input = tf.clip_by_value(soft_input, -100, 100)
        return tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_input, # soft input for this iteration
                soft_input,  # soft output for this iteration
                tf.constant(0,dtype=tf.int32), # iteration number
                tf.zeros([self.unit_batch_size,self.code.check_matrix_row,self.code.check_matrix_column],dtype=tf.float32)    ,# cv_matrix
                labels,
                early_stopping_indicator
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
      #normalized_tensor = 1.0
      #normalized_tensor = tf.nn.softplus(self.list_weights[iteration])
      if GL.get_map('selected_decoder_type')=='SPA':
        vc_matrix = tf.clip_by_value(vc_matrix, -10, 10)
        vc_matrix = tf.tanh(vc_matrix / 2.0) #tanh function applied 
        supple_matrix = 1 - self.code.H
        vc_matrix = vc_matrix+supple_matrix
        vc_matrix = tf.where(abs(vc_matrix)>0,vc_matrix,1e-10)
        temp = tf.reduce_prod(vc_matrix,2)                        
        temp = tf.expand_dims(temp,2)
        temp = temp*self.code.H
        cv_matrix = temp / vc_matrix
        cv_matrix = 2*tf.math.atanh(cv_matrix)
      else:
        normalized_tensor = 1.0
        check_matrix_H = self.code.H
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,-1e30-1.,0.)
        back_matrix = tf.expand_dims(back_matrix,0)
        vc_matrix_abs = tf.abs(vc_matrix)
        #if tf.reduce_any(vc_matrix_abs>1e20):
         # print("Horribly big figure appeared!")
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
        return tf.reduce_all(temp,1)
    
    def continue_condition(self,soft_input,soft_output,iteration,cv_matrix,labels,early_stopping_indicator):
      assert_judge = tf.reduce_all(early_stopping_indicator)
      if assert_judge:
        condition = False
      else:
        condition = (iteration < self.num_iterations)
      return condition

    def belief_propagation_iteration(self,soft_input,soft_output,iteration,cv_matrix,labels,early_stopping_indicator):
        # compute vc
        vc_matrix = self.compute_vc(cv_matrix,soft_input,iteration)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix,iteration) 
        # get output for this iteration
        soft_output = self.marginalize(cv_matrix,soft_input)
        iteration += 1
        early_stopping_indicator = self.early_stopping(soft_output,labels)   
            
        return soft_input,soft_output,iteration,cv_matrix,labels,early_stopping_indicator