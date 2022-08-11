# -*- coding: utf-8 -*-
import time
T1 = time.process_time()
import os
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import tensorflow  as tf
import sys
import fill_matrix_info as Fill_matrix
import globalmap as GL
import display_selection as Display
import ms_decoder_functions_tf2 as Decoder_module
import read_TFdata as Reading

tf.enable_eager_execution()
# Belief propagation using TensorFlow.Run as follows:
# python main.py 0 1.5 3 0.5 100 1000 0.85 10 LDPC_576_432.alist LDPC_576_432.gmat output_file SNNMS/NNMS  191 
#          1 2  3  4  5  6  7   8   9          10         11     12     13      
# setting global parameters
GL.set_map('ALL_ZEROS_CODEWORD_TRAINING', True)
GL.set_map('SIGMA_SCALING_TRAINING', True)
#command line arguments
GL.set_map('seed', int(sys.argv[1]))
GL.set_map('snr_lo', float(sys.argv[2]))
seed=GL.get_map('seed')
#np.random.seed(seed)
GL.set_map('snr_hi', float(sys.argv[3]))
GL.set_map('snr_step', float(sys.argv[4]))
GL.set_map('train_batch_size', int(sys.argv[5]))
GL.set_map('num_batch_train', int(sys.argv[6]))
GL.set_map('num_iterations', int(sys.argv[7]))
GL.set_map('H_filename', sys.argv[8])
GL.set_map('G_filename', sys.argv[9])
GL.set_map('selected_decoder_type', sys.argv[10])
GL.set_map('message_bit_number',int(sys.argv[11]))



# the training/testing paramters setting when selected_decoder_type="NNMS", "SNNMS"
GL.set_map('weighted_L', 0.2)  #balance between cross entroy and MSE(minimum squared error)
#GL.set_map('gamma',0.5)
GL.set_map('epochs',6)
GL.set_map('Cumulative_loss_indicator',False)
GL.set_map('initial_learning_rate', 0.002)
GL.set_map('decay_rate', 0.95)
GL.set_map('decay_step', 200)
GL.set_map('num_weight_train',5) #number of trainable parameters
train_batch_size = GL.get_map('train_batch_size')
data_dir = '../Data_gen_v1/data/batch-'+str(train_batch_size)
logdir = './logs'
ckpts_dir = './ckpts/'
ckpt_nm = 'ldpc-ckpt'
restore_step = ''

gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print(gpus, cpus)

Display.display_selection()

H_filename=GL.get_map('H_filename')
G_filename=GL.get_map('G_filename')
if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING') == True: 
    GL.set_map('G_filename', '')
else:
    print("Generator matrix has to be designated!!!")

code = Fill_matrix.Code.load_code(H_filename, G_filename)
GL.set_map('code_parameters', code)

#training setting
#retrieving global paramters of the code
#n = code.check_matrix_column

snr_lo = GL.get_map('snr_lo')
snr_hi = GL.get_map('snr_hi')
snr_step = GL.get_map('snr_step')
SNRs = np.arange(snr_lo, snr_hi+snr_step, snr_step) 

start_step = 0
train_steps = int(GL.get_map('num_batch_train'))*GL.get_map('epochs')

best_ber = 1  #initial ber placeholder for replacement

decay_rate = GL.get_map('decay_rate')
initial_learning_rate = GL.get_map('initial_learning_rate')
decay_steps = GL.get_map('decay_step')
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate,staircase=True)


# reading in training/validating data;make dataset iterator
dataset_train = Reading.data_handler(code.check_matrix_column,data_dir+'/ldpc-train.tfrecord',train_batch_size)
iterator = iter(dataset_train)

    
Model = Decoder_module.Decodering_model()

optimizer = tf.keras.optimizers.Adam(exponential_decay)



#fer,ber,index= get_eval(soft_output, tf_train_labels)

# summary
summary_writer = tf.summary.create_file_writer('./tensorboard')     # the parameter is the log folder we created
# tf.summary.trace_on(graph=True, profiler=True)  # Open Trace option, then the dataflow graph and profiling information can be recorded
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# whether to restore or not

checkpoint = tf.train.Checkpoint(myAwesomeModel=Model, myAwesomeOptimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, directory=ckpts_dir, checkpoint_name=ckpt_nm, max_to_keep=5)

 
if restore_step:
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
        if restore_step == 'latest':
            ckpt_f = tf.train.latest_checkpoint(ckpts_dir)
            start_step = int(ckpt_f.split('-')[-1]) + 1
        else:
            ckpt_f = ckpts_dir+ckpt_nm+'-'+restore_step
        print('Loading wgt file: '+ ckpt_f)
        checkpoint.restore(ckpt_f)
if (restore_step in ['','latest']):
    for batch_index in range(start_step, train_steps):
        inputs = next(iterator)
        with tf.GradientTape() as tape:
            soft_output,loss = Model(inputs)
            fer,ber,_= Model.get_eval(soft_output, inputs[1])
            with summary_writer.as_default():                               # the logger to be used
                 tf.summary.scalar("loss", loss, step=batch_index)
                 tf.summary.scalar("FER", fer, step=batch_index)  # you can also add other indicators below
                 tf.summary.scalar("BER", ber, step=batch_index)  # you can also add other indicators below
                 #tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=./trace_graph)    # Save Trace information to a file       

        grads = tape.gradient(loss,Model.variables) 
        grads_and_vars=zip(grads, Model.variables)
        capped_gradients = [(tf.clip_by_norm(grad,5), var) for grad, var in grads_and_vars if grad is not None]
        optimizer.apply_gradients(capped_gradients)
        # output_hard_decoding = tf.where(soft_output>=0,0,1)
        # output_hard_flat = tf.reshape(output_hard_decoding,shape=[-1,1])
        # labels_flat = tf.reshape(inputs[1],shape=[-1,1])
        # sparse_categorical_accuracy.update_state(labels_flat, output_hard_flat)
        
        # print("test ber accuracy: %f" % sparse_categorical_accuracy.result())    

        print("Step%4d: learning rate:%.4f current_loss:%.4f current_FER:%.4f current_BER:%.4f"%\
                  (batch_index,exponential_decay(batch_index),loss.numpy(), fer.numpy(), ber.numpy()) ) 
        # log to stdout 
        if batch_index % 100 == 0 or batch_index == train_steps-1:
            manager.save(checkpoint_number=batch_index) 
            if ber< best_ber:
                best_ber = ber
                best_ber_step = batch_index   
        with open(ckpts_dir+'/best_ber','w') as f:
            f.write('best step is %d(ber)\n'%best_ber_step)
        print('best step is %d(ber)\n'%best_ber_step)
T2 =time.process_time()
print('程序运行时间:%s秒'%(T2 - T1))

#%load_ext tensorboard
#%tensorboard --logdir='/content/drive/MyDrive/LDPC_decoding/Test_zone/current/logs/LDPC_DL'

