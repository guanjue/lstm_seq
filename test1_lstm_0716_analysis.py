import os,sys
from PIL import Image
from tensorflow.python.ops import rnn
import tensorflow as tf
import numpy as np
import math
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm

import shutil
import sys
sys.path.insert(0, '/Users/universe/Documents/2016_BG/Pugh_lab/DeepShape/bin')
import plot_curves
#import mutation
sess = tf.InteractiveSession()
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def one_conv_layer_net(sec_d,thr_d,for_d,first_filter_out,sec_filter_out,full_cn_out,filter1_size1,filter1_size2,filter2_size1,filter2_size2,iter_num,batch_size,train_pos,train_neg,max_pool1,max_pool2,vkmer,vkmer2,training_speed,seq_num,shape_num,file_list_all,file_list_random):
	filter1_size1_0=1
	######### get data
	### read data (shuffled seq)
	input_allpks_files=[]
	input_allpks_files_label=[]
	file_list_allpks=open(file_list_all,'r')
	for records in file_list_allpks:
		input_allpks_files.append(records.split()[0])
		input_allpks_files_label.append(records.split()[1])

	#print(input_allpks_files)
	all_pks_data={}
	for i in range(0,seq_num+shape_num):
		all_pks_data[i]=[]
		data=open(input_allpks_files[i],'r')
		for records in data:
			#all_pks_data[i].append([np.float32(x.strip()) for x in records.split()[3:len(records.split())-3]]) 24:77
			#all_pks_data[i].append([np.float32(x.strip()) for x in records.split()[27-filter1_size1/2:len(records.split())-27+filter1_size1/2]])
			#all_pks_data[i].append([np.float32(x.strip()) for x in records.split()[len(records.split())/2-sec_d/2-filter1_size1_0/2:len(records.split())/2+sec_d/2+filter1_size1_0/2]])
			all_pks_data[i].append([np.float32(x.strip()) for x in records.split()[len(records.split())/2-sec_d/2:len(records.split())/2+sec_d/2]])


	### read negative control (shuffled seq)		
	input_randompks_files=[]
	file_list_randompks=open(file_list_random,'r')
	for records in file_list_randompks:
		input_randompks_files.append(records.split()[0])

	#print(input_allpks_files)
	random_pks_data={}
	for i in range(0,seq_num+shape_num):
		random_pks_data[i]=[]
		data=open(input_randompks_files[i],'r')
		for records in data:
			#random_pks_data[i].append([np.float32(x.strip()) for x in records.split()[3:len(records.split())-3]])
			#random_pks_data[i].append([np.float32(x.strip()) for x in records.split()[27-filter1_size1/2:len(records.split())-27+filter1_size1/2]])
			#random_pks_data[i].append([np.float32(x.strip()) for x in records.split()[len(records.split())/2-sec_d/2-filter1_size1_0/2:len(records.split())/2+sec_d/2+filter1_size1_0/2]])
			random_pks_data[i].append([np.float32(x.strip()) for x in records.split()[len(records.split())/2-sec_d/2:len(records.split())/2+sec_d/2]])

	### normalization
	for i in range(0,seq_num+shape_num):
		random_pks_data[i]=random_pks_data[i]/np.max([np.max(np.absolute(all_pks_data[i])),np.max(np.absolute(random_pks_data[i]))])
		#print(all_pks_data[i][0])
		all_pks_data[i]=all_pks_data[i]/np.max([np.max(np.absolute(all_pks_data[i])),np.max(np.absolute(random_pks_data[i]))])
		#print('lalallal')
		#print(all_pks_data[i][0])

	matrix_K_L_M=[]
	for i in range(0,len(all_pks_data[0])):
		tmp_matrix_K_L_M=[]
		for j in range(0,len(all_pks_data[0][i])):	
			for k in range(0,len(all_pks_data)):
				tmp_matrix_K_L_M=tmp_matrix_K_L_M+[all_pks_data[k][i][j]]
		matrix_K_L_M.append(tmp_matrix_K_L_M)

	matrix_K_L_M=np.array(matrix_K_L_M)
	print(matrix_K_L_M.shape)
	#print(matrix_K_L_M[0])

	### Extract motif information and shape information
	#matrix_K_L_M_matrix=matrix_K_L_M.reshape([-1,sec_d+filter1_size1-1,thr_d,for_d])
	matrix_K_L_M_matrix=matrix_K_L_M.reshape([-1,sec_d,thr_d,for_d])
	print((matrix_K_L_M_matrix.mean(axis=0)[:,0,:]).shape)
	matrix_K_L_M_matrix_mean=matrix_K_L_M_matrix.mean(axis=0)[:,0,:]
	matrix_K_L_M_matrix_max=matrix_K_L_M_matrix.max(axis=0)[:,0,:]
	matrix_K_L_M_matrix_min=matrix_K_L_M_matrix.min(axis=0)[:,0,:]
	#print(matrix_K_L_M_matrix_min[:,4])
	#print(min(matrix_K_L_M_matrix_min[:,4]))
	#print(np.random.uniform(min(matrix_K_L_M_matrix_min[:,4]), max(matrix_K_L_M_matrix_max[:,4]), size=1)[0])

	if seq_num!=0:
		matrix_K_L_M_matrix_pwm=matrix_K_L_M_matrix_mean[:,range(0,seq_num)]
		np.savetxt('matrix_K_L_M_matrix_pwm.txt', matrix_K_L_M_matrix_pwm,delimiter='\t')

	matrix_K_L_M_matrix_shape={}
	if shape_num!=0:
		for i in range(seq_num,seq_num+shape_num):
			matrix_K_L_M_matrix_shape[i]=matrix_K_L_M_matrix_mean[:,i]

	matrix_K_L_M_random=[]
	for i in range(0,len(random_pks_data[0])):
		tmp_matrix_K_L_M_random=[]
		for j in range(0,len(random_pks_data[0][i])):	
			for k in range(0,len(random_pks_data)):
				tmp_matrix_K_L_M_random=tmp_matrix_K_L_M_random+[random_pks_data[k][i][j]]
		matrix_K_L_M_random.append(tmp_matrix_K_L_M_random)

	matrix_K_L_M_random=np.array(matrix_K_L_M_random)

	### Extract motif information and shape information
	#matrix_K_L_M_random_matrix=matrix_K_L_M_random.reshape([-1,sec_d+filter1_size1-1,thr_d,for_d])
	matrix_K_L_M_random_matrix=matrix_K_L_M_random.reshape([-1,sec_d,thr_d,for_d])
	matrix_K_L_M_random_matrix_mean=matrix_K_L_M_random_matrix.mean(axis=0)[:,0,:]
	matrix_K_L_M_random_matrix_max=matrix_K_L_M_random_matrix.max(axis=0)[:,0,:]
	matrix_K_L_M_random_matrix_min=matrix_K_L_M_random_matrix.min(axis=0)[:,0,:]

	if seq_num!=0:
		matrix_K_L_M_random_matrix_pwm=matrix_K_L_M_random_matrix_mean[:,range(0,seq_num)]
		np.savetxt('matrix_K_L_M_random_matrix_pwm.txt', matrix_K_L_M_random_matrix_pwm,delimiter='\t')

	matrix_K_L_M_random_matrix_shape={}
	if shape_num!=0:
		for i in range(seq_num,seq_num+shape_num):
			matrix_K_L_M_random_matrix_shape[i]=matrix_K_L_M_random_matrix_mean[:,i]
	

	############
	### data normalization

	######
	np.random.seed(seed=2017)
	index_array_matrix_K_L_M=np.arange(matrix_K_L_M.shape[0])
	np.random.shuffle(index_array_matrix_K_L_M)
	index_array_matrix_K_L_M_random=np.arange(matrix_K_L_M_random.shape[0])
	np.random.shuffle(index_array_matrix_K_L_M_random)

	matrix_K_L_M=matrix_K_L_M[index_array_matrix_K_L_M]
	matrix_K_L_M_random=matrix_K_L_M_random[index_array_matrix_K_L_M_random]

	matrix_K_L_M_test=np.concatenate((matrix_K_L_M[train_pos:],matrix_K_L_M_random[train_neg:]))
	ys_test=np.repeat([[np.float32(1),np.float32(0)],[np.float32(0),np.float32(1)]], [matrix_K_L_M[train_pos:].shape[0],matrix_K_L_M_random[train_neg:].shape[0]], axis=0)

	np.savetxt('matrix_K_L_M_test.txt', matrix_K_L_M_test)
	print(matrix_K_L_M_test.shape)

	matrix_K_L_M_visual_p=matrix_K_L_M[train_pos:]
	ys_visual_p=np.repeat([[np.float32(1),np.float32(0)]], matrix_K_L_M[train_pos:].shape[0], axis=0)

	matrix_K_L_M_visual_n=matrix_K_L_M_random[train_neg:]
	ys_visual_n=np.repeat([[np.float32(0),np.float32(1)]], matrix_K_L_M_random[train_neg:].shape[0], axis=0)

	######### Tensorflow model
	x = tf.placeholder(tf.float32, shape=[None, matrix_K_L_M_test.shape[1]])
	y_ = tf.placeholder(tf.float32, shape=[None, ys_test.shape[1]])
	### Weight/bias Initialization
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	### Convolution and Pooling
	def conv2d(x, W):
		#return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x1(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],strides=[1, 2, 1, 1], padding='SAME')

	def max_pool_4x1(x):
		return tf.nn.max_pool(x, ksize=[1, 4, 1, 1],strides=[1, 4, 1, 1], padding='SAME')

	def max_pool_5x1(x):
		return tf.nn.max_pool(x, ksize=[1, 5, 1, 1],strides=[1, 5, 1, 1], padding='SAME')

	def max_pool_WTA(x,layers):
		return tf.nn.max_pool(x, ksize=[1, 1, 1, layers],strides=[1, 1, 1, layers], padding='SAME')

	def weight_matrix2human(fd,input,output):
		### write human readable weight matrix (first convolution layer) 
		i0=fd
		W_1=open(input,'r')
		W_2=[]
		for records in W_1:
			W_2.append([float(x.strip()) for x in records.split()])
		W_2=np.array(W_2).transpose() ### transpose the weight matrix to a np array with n (number of channel) rows
		W_result=open(output,'w')
		i=0
		for records in W_2:
			for record in records:
				W_result.write(str(record)+'\t')
				i+=1
				if i%i0==0:
					W_result.write('\n')	
			W_result.write('\n')
		W_result.close()
		W_1.close()


	### input layer
	x_image = tf.reshape(x, [-1,sec_d,thr_d,for_d])

	### First Convolutional Layer
	### The First layer will have 32 features for each 6x1 patch.
	W_conv1 = weight_variable([filter1_size1, filter1_size2, for_d, first_filter_out])
	b_conv1 = bias_variable([first_filter_out])
	### 
	#pool_shape1=sec_d*thr_d/max_pool1
	#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_conv1 = tf.nn.softsign(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1))
	#h_conv1 = h_conv1/abs(h_conv1)
	print(h_conv1.shape)
	#h_conv1_activate_0 = tf.transpose(tf.nn.softsign([h_conv1[:,:,:,0]]),[1, 2, 3, 0])
	#h_conv1_activate_1 = tf.transpose(tf.nn.softsign([h_conv1[:,:,:,1]]),[1, 2, 3, 0])
	#print(h_conv1_activate_0.shape)
	#print(h_conv1_activate_1.shape)
	#h_conv1_activate=tf.concat([h_conv1_activate_0,h_conv1_activate_1],3)
	#keep_prob1 = tf.placeholder(tf.float32)
	#h_conv1_drop = tf.nn.dropout(h_conv1, keep_prob1)
	h_pool1_wta = tf.reduce_max(h_conv1, reduction_indices=[3], keep_dims=True)


	# Permuting batch_size and n_steps
	#h_conv1_out = tf.transpose(h_pool1, [1, 0, 2, 3])
	# Reshaping to (n_steps*batch_size, n_input)
	x_transpose = tf.transpose(h_pool1_wta, [1, 0, 2, 3])
	#x_reshape = tf.reshape(x_transpose, [-1, first_filter_out])
	x_reshape = tf.reshape(x_transpose, [-1, 1])
	print('reshape:')
	print(x_reshape.shape)
	# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	rnn_input = tf.split(x_reshape,axis=0, num_or_size_splits=sec_d)
	print('reshape:')
	print(len(rnn_input))
	print(rnn_input[0:3])
	#lstm_cell = tf.contrib.rnn.LSTMCell(1, forget_bias=1.0)
	#lstm_cell = tf.contrib.rnn.LSTMCell(1)
	lstm_cell = tf.contrib.rnn.GRUCell(1,'tanh')

	outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
	'''
	with tf.variable_scope('rnn_cell'):
		W = tf.get_variable('W', [num_classes + state_size, state_size])
		b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

	def rnn_cell(rnn_input, state):
		with tf.variable_scope('rnn_cell', reuse=True):
			W = tf.get_variable('W', [num_classes + state_size, state_size])
			b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
		return tf.tanh(tf.matmul(tf.concat(1, [rnn_input, state]), W) + b)

	state = tf.constant_initializer(0.0)
	rnn_outputs = []
	for rnn_input in h_conv1_out:
	    state = rnn_cell(rnn_input, state)
	    rnn_outputs.append(state)
	final_state = rnn_outputs[-1]
	'''
	y_conv = tf.matmul(outputs[-1], weight_variable([1, 2]) )# + bias_variable([2]) )
	#print((outputs[-1]))
	#print(tf.concat(1,outputs))
	#W_atten=tf.nn.softmax(tf.concat(1,outputs),dim=-1)
	#print(W_atten)
	#print(tf.concat(1,outputs))
	#print(outputs)
	#W_atten=weight_variable([sec_d, 2])
	#b_atten=bias_variable([2])
	#focus_out = tf.matmul(tf.concat(1,outputs), W_atten)
	#atten1=W_atten * tf.concat(1,outputs) 
	#print(atten1)
	#y_conv = outputs[-2:]#tf.matmul(tf.concat(1,outputs), W_atten)# + b_atten
	#print(y_conv)
	### Densely Connected Layer
	### a fully-connected layer with 1024 neurons to allow processing on the entire seq.

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)
	train_step = tf.train.AdamOptimizer(training_speed).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	keep_prob1 = tf.placeholder(tf.float32)
	keep_prob2 = tf.placeholder(tf.float32)
	'''
	W_fc1 = weight_variable([pool_shape1 * first_filter_out, full_cn_out])
	b_fc1 = bias_variable([full_cn_out])

	###
	h_pool_flat = tf.reshape(h_pool1, [-1, pool_shape1 * first_filter_out])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

	### Dropout
	### To reduce overfitting, we will apply dropout before the readout layer. 
	### We create a placeholder for the probability that a neuron's output is kept during dropout
	keep_prob2 = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob2)

	### Readout Layer
	### Finally, we add a softmax layer, just like for the one layer softmax regression above.
	W_fc2 = weight_variable([full_cn_out, ys_test.shape[1]])
	b_fc2 = bias_variable([ys_test.shape[1]])
	### 
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	'''
	### Evaluate the Model
	#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	#cross_entropy = tf.nn.softmax_cross_entropy_with_logits((tf.matmul(h_fc1_drop, W_fc2) + b_fc2), y_)
	#train_step = tf.train.AdamOptimizer(training_speed).minimize(cross_entropy)
	#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	### initial parameter
	sess.run(tf.initialize_all_variables())

	### write initial net
	'''
	with file('test_pre.txt', 'w') as outfile:
		for slice_1d in sess.run(W_conv1):
			for slice_2d in slice_1d:
				np.savetxt(outfile, slice_2d)
	'''

	print('Start!!! LSTM')
	saver = tf.train.Saver()
	saver.restore(sess, "trained_rnn.ckpt")

	accuracy_array=np.array(sess.run(accuracy, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))
	print('accuracy_array')
	print(accuracy_array)

	#ys_test_array=np.array(sess.run(ys_test, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))
	#print('ys_test_array')
	#print(ys_test_array.shape)
	#np.savetxt('ys_test_array.txt', ys_test_array)
	def write2d_array(array,output):
		r1=open(output,'w')
		for records in array:
			for i in range(0,len(records)-1):
				r1.write(str(records[i])+'\t')
			r1.write(str(records[len(records)-1])+'\n')
		r1.close()

	W_conv1_layer=np.array(sess.run(W_conv1, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))
	print('W_conv1_layer')
	print(W_conv1_layer.shape)
	#np.savetxt('W_conv1_layer.txt', W_conv1_layer)

	rnn_hidden_state=np.array(sess.run(outputs, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))
	print('rnn_hidden_state')
	print(np.transpose(np.array(rnn_hidden_state[:,0:2347,:],dtype=float),(1,0,2)).shape)
	rnn_hidden=np.transpose(np.array(rnn_hidden_state[:,0:2347,:],dtype=float),(1,0,2))[:,:,0]
	write2d_array(rnn_hidden,'rnn_hidden_state_pos0.txt')
	#np.savetxt('rnn_hidden_state_pos0.txt', np.transpose(rnn_hidden_state[:,0:2347,:],(1,0,2)) )

	h_pool1_wta_array=np.array(sess.run(h_pool1_wta, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))
	print('h_pool1_wta_array')
	print(h_pool1_wta_array.shape)
	#np.savetxt('h_conv1_array.txt', h_conv1_array)

	h_conv1_array_pos=np.array(sess.run(h_conv1, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))
	print('h_pool1_wta_array')
	print(h_conv1_array_pos.shape)

	y_conv_array=np.array(sess.run(y_conv, feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))[0:2347,:]
	print('y_conv_array')
	print(y_conv_array.shape)

	order=np.argsort(-y_conv_array[:,0])
	print(y_conv_array[order,0][0:10])
	print(y_conv_array[order,0][-10:])

	write2d_array(rnn_hidden[order,:],'rnn_hidden_state_pos0.txt')
	write2d_array(np.array(h_pool1_wta_array[order,:,0,0],dtype=float),'h_pool1_wta_array.txt')

	write2d_array(np.array(h_conv1_array_pos[order,:,0,0],dtype=float),'h_conv1_array_pos0.txt')
	write2d_array(np.array(h_conv1_array_pos[order,:,0,1],dtype=float),'h_conv1_array_pos1.txt')
	write2d_array(np.array(h_conv1_array_pos[order,:,0,2],dtype=float),'h_conv1_array_pos2.txt')
	write2d_array(np.array(h_conv1_array_pos[order,:,0,3],dtype=float),'h_conv1_array_pos3.txt')
	#write2d_array(np.array(h_conv1_array_pos[order,:,0,4],dtype=float),'h_conv1_array_pos4.txt')
	#write2d_array(np.array(h_conv1_array_pos[order,:,0,5],dtype=float),'h_conv1_array_pos5.txt')
	#write2d_array(np.array(h_conv1_array_pos[order,:,0,6],dtype=float),'h_conv1_array_pos6.txt')
	#write2d_array(np.array(h_conv1_array_pos[order,:,0,7],dtype=float),'h_conv1_array_pos7.txt')

	#np.savetxt('h_conv1_array_pos0.txt', np.array(h_conv1_array_pos[0:2347,:,0],dtype=float))
	#np.savetxt('h_conv1_array_pos1.txt', np.array(h_conv1_array_pos[0:2347,:,1],dtype=float))


	#time python test1_lstm_0716_analysis.py -s 200 -t 1 -f 4 -i 2 -e 16 -u 16 -l 5 -o 1 -r 8 -v 1 -a 10000 -b 100 -p 4000 -n 4000 -x 1 -y 2 -k 2 -m 6 -d 0.0001 -j 4 -q 0 -w pos_list.txt -z neg_list.txt
	'''
	W_conv1_data=sess.run(W_conv1)
	print(W_conv1_data.shape)
	W_fc1_data=np.transpose(sess.run(W_fc1))
	print(W_fc1_data.shape)
	W_fc1_matrix_expand=np.repeat(np.transpose(W_fc1_data[0,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1)
	#W_fc1_matrix_expand=np.repeat(np.transpose(W_fc1_data[m,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1)
	print(W_fc1_matrix_expand.shape)
	W_fc2_data=np.transpose(sess.run(W_fc2))
	print(W_fc2_data[0])
	W_fc2fc1_matrix_expand=np.repeat(np.transpose(np.dot(W_fc2_data,W_fc1_data)[0,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1)
	print(W_fc2fc1_matrix_expand.shape)
	print(np.dot(W_fc2_data[0,:],W_fc1_data).shape)
	print(np.dot(W_fc2_data,W_fc1_data).shape)
	print(W_fc1_data.shape)
	h_conv1_data_n=sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	print(  (np.transpose(np.mean(np.greater(h_conv1_data_n[:,:,0,:],0),axis=0))*W_fc1_matrix_expand).shape   )
	print(  (np.transpose(np.mean(np.greater(h_conv1_data_n[:,:,0,:],0),axis=0))*W_fc1_matrix_expand)   )
	print(W_fc1_matrix_expand)
	'''

	'''
	###################################
	### visualize k-mer mutation effects (deconvolute the model)
	print('hahahahahaha'+str(vkmer)+'-mer mutation')
	#print(matrix_K_L_M_test[0])
	accuracy_table=open('accuracy_table.'+str(vkmer)+'mer.mutate.txt','w')
	accuracy_table_data=[]
	for i in range(0,sec_d*thr_d*1-vkmer+1):
		tmp_all={}
		tmp_seq={}
		tmp_shape={}
		for seq_num_i in range(0,seq_num):
			tmp_seq[seq_num_i]=np.copy(matrix_K_L_M_test)
		for shape_num_i in range(0,shape_num):
			tmp_shape[shape_num_i]=np.copy(matrix_K_L_M_test)
		for all_i in range(0,3):
			tmp_all[all_i]=np.copy(matrix_K_L_M_test)

		for j in range(0,matrix_K_L_M_test.shape[0]): ### mutation l-mer (set l mer data to 0 in all pos/neg seqs)	
			for seq_channel_i in range(0,seq_num):
				for l_mut in range(0,vkmer):
					tmp_seq[seq_channel_i][j][i*for_d+l_mut*for_d+seq_channel_i]=np.mean(matrix_K_L_M_matrix_mean[:,seq_channel_i])	### mutation l-mer (seq A channel)

			for shape_channel_i in range(0,shape_num):
				shape_min=min(np.append(matrix_K_L_M_matrix_min[:,seq_num+shape_channel_i],matrix_K_L_M_random_matrix_min[:,seq_num+shape_channel_i]))
				shape_max=max(np.append(matrix_K_L_M_matrix_max[:,seq_num+shape_channel_i],matrix_K_L_M_random_matrix_max[:,seq_num+shape_channel_i]))
				for l_mut in range(0,vkmer):
					tmp_shape[shape_channel_i][j][i*for_d+l_mut*for_d+seq_num+shape_channel_i]=np.random.uniform(shape_min, shape_max, size=1)[0]
					### mutation l-mer (shape channel) set it equals to mean of that channel

			for all_channel_i in range(0,3):
				if (all_channel_i==0 or all_channel_i==1):
					for all_channel_num_seq in range(0,seq_num):
						for l_mut in range(0,vkmer):
							tmp_all[all_channel_i][j][i*for_d+l_mut*for_d+all_channel_num_seq]=np.mean(matrix_K_L_M_matrix_mean[:,seq_channel_i])
				if (all_channel_i==0 or all_channel_i==2):
					for all_channel_num_shape in range(seq_num,seq_num+shape_num):
						shape_min=min(np.append(matrix_K_L_M_matrix_min[:,all_channel_num_shape],matrix_K_L_M_random_matrix_min[:,all_channel_num_shape]))
						shape_max=max(np.append(matrix_K_L_M_matrix_max[:,all_channel_num_shape],matrix_K_L_M_random_matrix_max[:,all_channel_num_shape]))						
						for l_mut in range(0,vkmer):
							tmp_all[all_channel_i][j][i*for_d+l_mut*for_d+all_channel_num_shape]=np.random.uniform(shape_min, shape_max, size=1)[0]
							### mutation l-mer (shape channel) set it equals to mean of that channel

		accuracy_position=[]
		for i_all in range(0,len(tmp_all)):
			accuracy_position=accuracy_position+[ accuracy.eval(feed_dict={x: tmp_all[i_all], y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}) ]
		for i_seq in range(0,len(tmp_seq)):
			accuracy_position=accuracy_position+[ accuracy.eval(feed_dict={x: tmp_seq[i_seq], y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}) ]
		for i_shape in range(0,len(tmp_shape)):
			accuracy_position=accuracy_position+[ accuracy.eval(feed_dict={x: tmp_shape[i_shape], y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}) ]

		print(i,accuracy_position)
		accuracy_table_data.append(accuracy_position)
		tmp_table=accuracy_position

		for accu in tmp_table:
			accuracy_table.write(str(accu)+'\t')
		accuracy_table.write('\n')
	accuracy_table.write('\n'+str(accuracy.eval(feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))+'\n')
	accuracy_table.close()


	
	### visualize SECOND k-mer mutation effects (deconvolute the model)
	print('hahahahahaha'+str(vkmer2)+'-mer mutation')
	#print(matrix_K_L_M_test[0])
	accuracy_table_vkmer2=open('accuracy_table.'+str(vkmer2)+'mer.mutate.txt','w')
	accuracy_table_data_vkmer2=[]
	for i in range(0,sec_d*thr_d*1-vkmer2+1):
		tmp_all={}
		tmp_seq={}
		tmp_shape={}
		for seq_num_i in range(0,seq_num):
			tmp_seq[seq_num_i]=np.copy(matrix_K_L_M_test)
		for shape_num_i in range(0,shape_num):
			tmp_shape[shape_num_i]=np.copy(matrix_K_L_M_test)
		for all_i in range(0,3):
			tmp_all[all_i]=np.copy(matrix_K_L_M_test)

		for j in range(0,matrix_K_L_M_test.shape[0]): ### mutation l-mer (set l mer data to 0 in all pos/neg seqs)	
			for seq_channel_i in range(0,seq_num):
				for l_mut in range(0,vkmer2):
					tmp_seq[seq_channel_i][j][i*for_d+l_mut*for_d+seq_channel_i]=np.mean(matrix_K_L_M_matrix_mean[:,seq_channel_i])	### mutation l-mer (seq A channel)

			for shape_channel_i in range(0,shape_num):
				shape_min=min(np.append(matrix_K_L_M_matrix_min[:,seq_num+shape_channel_i],matrix_K_L_M_random_matrix_min[:,seq_num+shape_channel_i]))
				shape_max=max(np.append(matrix_K_L_M_matrix_max[:,seq_num+shape_channel_i],matrix_K_L_M_random_matrix_max[:,seq_num+shape_channel_i]))
				for l_mut in range(0,vkmer2):
					tmp_shape[shape_channel_i][j][i*for_d+l_mut*for_d+seq_num+shape_channel_i]=np.random.uniform(shape_min, shape_max, size=1)[0]
					### mutation l-mer (shape channel) set it equals to mean of that channel

			for all_channel_i in range(0,3):
				if (all_channel_i==0 or all_channel_i==1):
					for all_channel_num_seq in range(0,seq_num):
						for l_mut in range(0,vkmer2):
							tmp_all[all_channel_i][j][i*for_d+l_mut*for_d+all_channel_num_seq]=np.mean(matrix_K_L_M_matrix_mean[:,seq_channel_i])
				if (all_channel_i==0 or all_channel_i==2):
					for all_channel_num_shape in range(seq_num,seq_num+shape_num):
						shape_min=min(np.append(matrix_K_L_M_matrix_min[:,seq_num+shape_channel_i],matrix_K_L_M_random_matrix_min[:,all_channel_num_shape]))
						shape_max=max(np.append(matrix_K_L_M_matrix_max[:,seq_num+shape_channel_i],matrix_K_L_M_random_matrix_max[:,all_channel_num_shape]))
						for l_mut in range(0,vkmer2):
							tmp_all[all_channel_i][j][i*for_d+l_mut*for_d+all_channel_num_shape]=np.random.uniform(shape_min, shape_max, size=1)[0]		
							### mutation l-mer (shape channel) set it equals to mean of that channel

		accuracy_position=[]
		for i_all in range(0,len(tmp_all)):
			accuracy_position=accuracy_position+[ accuracy.eval(feed_dict={x: tmp_all[i_all], y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}) ]
		for i_seq in range(0,len(tmp_seq)):
			accuracy_position=accuracy_position+[ accuracy.eval(feed_dict={x: tmp_seq[i_seq], y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}) ]
		for i_shape in range(0,len(tmp_shape)):
			accuracy_position=accuracy_position+[ accuracy.eval(feed_dict={x: tmp_shape[i_shape], y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}) ]

		print(i,accuracy_position)
		accuracy_table_data_vkmer2.append(accuracy_position)
		tmp_table=accuracy_position

		for accu in tmp_table:
			accuracy_table_vkmer2.write(str(accu)+'\t')
		accuracy_table_vkmer2.write('\n')
	accuracy_table_vkmer2.write('\n'+str(accuracy.eval(feed_dict={x: matrix_K_L_M_test, y_: ys_test, keep_prob1: 1.0, keep_prob2: 1.0}))+'\n')
	accuracy_table_vkmer2.close()
	'''

	
	### save model
	### save first convolutional layer weight matrix
	subdirectory='W_conv1_data'
	if os.path.exists('W_conv1_data'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)
	W_conv1_data=sess.run(W_conv1)
	#print(W_conv1_data.shape)
	for i in range(0,W_conv1_data.shape[3]):
		if seq_num!=0:
			### plot seq heatmap in each convolutional filter
			fig,ax=plt.subplots()
			heatmap=sns.clustermap(W_conv1_data[:,0,range(0,seq_num),i],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='bwr')
			filename0p=os.path.join(subdirectory,'W_conv1.'+str(i)+'.png')
			plt.savefig(filename0p,format='png')
			plt.clf()
			### plot shape curve in each convolutional filter
		if shape_num!=0:
			fig,ax=plt.subplots()
			color=['k','c','m','b','r','g','y']
			### set y_limit
			axes = plt.gca()
			ymax=np.max(W_conv1_data[:,0,range(seq_num,seq_num+shape_num),:])
			ymin=np.min(W_conv1_data[:,0,range(seq_num,seq_num+shape_num),:])
			axes.set_ylim([ymin-0.01,ymax+0.01])
			for j in range(seq_num,seq_num+shape_num):
				### plot each shape in same plot
				plt.plot(W_conv1_data[:,0,[j],i],color[j-seq_num]+'o') ### MGW
				plt.plot(W_conv1_data[:,0,[j],i],color[j-seq_num])
			filename0p=os.path.join(subdirectory,'W_conv1_shape.'+str(i)+'.png')
			plt.savefig(filename0p,format='png')
			plt.clf()
			### plot shape curve seperately
			color=['k','c','m','b','r','g','y']
			for j in range(seq_num,seq_num+shape_num):
				fig,ax=plt.subplots()
				### set y_limit
				axes = plt.gca()
				ymax=np.max(W_conv1_data[:,0,[j],:])
				ymin=np.min(W_conv1_data[:,0,[j],:])
				axes.set_ylim([ymin-0.01,ymax+0.01])
				### plot each shape in seperate plot
				plt.plot(W_conv1_data[:,0,[j],i],color[j-seq_num]+'o') ### MGW
				plt.plot(W_conv1_data[:,0,[j],i],color[j-seq_num])
				filename0p=os.path.join(subdirectory,'W_conv1_shape.'+str(i)+'.'+input_allpks_files_label[j]+'.png')
				plt.savefig(filename0p,format='png')
				plt.clf()
	### save first fully connected layer weight matrix
	subdirectory='W_fc1_data'
	if os.path.exists('W_fc1_data'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)
	W_fc1_data=np.transpose(sess.run(W_fc1))
	#print(np.transpose(W_fc1_data[0,].reshape([pool_shape1,first_filter_out])))
	#print(np.repeat(np.transpose(W_fc1_data[0,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1))
	#print(W_fc1_data.shape)

	for i in range(0,W_fc1_data.shape[0]):
		fig,ax=plt.subplots()
		W_fc1_matrix_expand=np.repeat(np.transpose(W_fc1_data[i,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1)
		heatmap=sns.clustermap(W_fc1_matrix_expand,col_cluster=False,row_cluster=True,method='complete',metric='euclidean',cmap='bwr',figsize=(20, 10))
		filename0p=os.path.join(subdirectory,'W_fc1.'+str(i)+'.png')
		plt.savefig(filename0p,format='png')
		plt.clf()

	W_conv1_data_merge=np.sum(W_conv1_data,axis=3)
	fig,ax=plt.subplots()
	axes = plt.gca()
	ymax=np.max(W_conv1_data_merge)
	ymin=np.min(W_conv1_data_merge)
	axes.set_ylim([ymin-0.01,ymax+0.01])
	### plot each shape in seperate plot
	for j in range(seq_num,seq_num+shape_num):
		plt.plot(W_conv1_data_merge[:,0,[j]],color[j-seq_num]+'o') ### MGW
		plt.plot(W_conv1_data_merge[:,0,[j]],color[j-seq_num])
	filename0p=os.path.join(subdirectory,'W_conv1_shape.'+str('merged')+'.'+input_allpks_files_label[j]+'.png')
	plt.savefig(filename0p,format='png')
	plt.clf()

	### get hidden layer activation states
	h_conv1_data_n=sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	h_conv1_data_p=sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0})
	h_conv1_data_n=sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	h_pool1_data_p=sess.run(h_pool1,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0})
	h_pool1_data_n=sess.run(h_pool1,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	h_fc1_data_p=sess.run(h_fc1,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0})
	h_fc1_data_n=sess.run(h_fc1,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	y_conv_data_p=sess.run(y_conv,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0})
	y_conv_data_n=sess.run(y_conv,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	classification=sess.run(y_conv, feed_dict={x: matrix_K_L_M_test, keep_prob1: 1.0, keep_prob2: 1.0})

	###################################################################
	### save 1st conv layer & 1st fully connected layer weight matrix
	subdirectory='W_conv1fc1_data_filtered_p'
	if os.path.exists('W_conv1fc1_data_filtered_p'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)
	W_conv1_data=sess.run(W_conv1)
	W_fc1_data=np.transpose(sess.run(W_fc1))
	W_fc2_data=np.transpose(sess.run(W_fc2))
	#print(W_fc2_data)
	#print(W_fc2_data[0])
	#h_conv1_data_p=sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0})
	### initialize dict for storing model shape
	merge_shape_allmodel={}
	for k in range(0,for_d):
		merge_shape_allmodel[k]={}
	### get model shape
	for m in range(0,W_fc1_data.shape[0]):
		W_fc1_matrix_expand=np.repeat(np.transpose(W_fc1_data[m,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1)
		#W_conv1fc1_matrix=np.dot(W_conv1_data,W_fc1_matrix_expand)
		W_fc1_matrix_expand_filter=np.transpose(np.mean(np.greater(h_conv1_data_p[:,:,0,:],0),axis=0)) * W_fc1_matrix_expand
		W_conv1fc1_matrix_filter=np.dot(W_conv1_data,W_fc1_matrix_expand_filter)
		###
		#print(W_conv1fc1_matrix_filter.shape)
		#print(W_conv1fc1_matrix_filter[:,0,1,0])

		### get merged model shape (matrix diagonal sum) 
		for k in range(0,W_conv1fc1_matrix_filter.shape[2]):
			merge_shape_allmodel[k][m]=np.repeat(0.0,W_conv1fc1_matrix_filter.shape[3]+filter1_size1-1) ### float repeat!!!
			for l in range(0,W_conv1fc1_matrix_filter.shape[3]):
				tmp=merge_shape_allmodel[k][m][l:(l+filter1_size1)]+W_conv1fc1_matrix_filter[:,0,k,l]
				merge_shape_allmodel[k][m][l:(l+filter1_size1)]=tmp	
		### 
		### plot each position's model shape
		for i in range(0,W_conv1fc1_matrix_filter.shape[3]):
			if seq_num!=0:
				### plot seq heatmap in each convolutional filter
				'''
				fig,ax=plt.subplots()
				heatmap=sns.clustermap(np.transpose(W_conv1fc1_matrix_filter[:,0,range(0,seq_num),i]),col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='bwr')
				filename0p=os.path.join(subdirectory,'Model_'+str(m)+'.W_conv1fc1_filter.'+str(i)+'.png')
				plt.savefig(filename0p,format='png')
				plt.clf()
				'''
				### plot shape curve in each convolutional filter
			if shape_num!=0:
				### plot shape curve seperately
				'''
				color=['k','c','m','b','r','g','y']
				for j in range(seq_num,seq_num+shape_num):
					fig,ax=plt.subplots()
					### set y_limit
					axes = plt.gca()
					ymax=np.max(W_conv1fc1_matrix_filter[:,0,[j],:])
					ymin=np.min(W_conv1fc1_matrix_filter[:,0,[j],:])
					axes.set_ylim([ymin-0.01,ymax+0.01])
					### plot each shape in seperate plot
					plt.plot(W_conv1fc1_matrix_filter[:,0,[j],i],color[j-seq_num]+'o') ### MGW
					plt.plot(W_conv1fc1_matrix_filter[:,0,[j],i],color[j-seq_num])
					filename0p=os.path.join(subdirectory,input_allpks_files_label[j]+'.Model_'+str(m)+'.W_conv1fc1_shape_filter.'+str(i)+'.png')
					plt.savefig(filename0p,format='png')
					plt.clf()
				'''
	### get universal y limits
	merge_shape_allmodel_shape_fc2filtered_heatmap_p=[np.repeat(0.0,len(merge_shape_allmodel[k][0]))]
	for k in range(0,W_conv1fc1_matrix_filter.shape[2]):
		ylower=np.min([np.min(merge_shape_allmodel[k][x]) for x in merge_shape_allmodel[k] ])-0.01
		yupper=np.max([np.max(merge_shape_allmodel[k][x]) for x in merge_shape_allmodel[k] ])+0.01
		ylowerfc=np.min([np.min(merge_shape_allmodel[k][x]*W_fc2_data[0][x]) for x in merge_shape_allmodel[k] ])-0.01
		yupperfc=np.max([np.max(merge_shape_allmodel[k][x]*W_fc2_data[0][x]) for x in merge_shape_allmodel[k] ])+0.01
		### plot conv1, fc1, fc2 all merge one model shape
		merge_shape_allmodel_shape=np.repeat(0.0,len(merge_shape_allmodel[k][0]))
		merge_shape_allmodel_shape_fc2filtered=np.repeat(0.0,len(merge_shape_allmodel[k][0]))
		### plot entire model shape
		for m in range(0,W_fc1_data.shape[0]):
			#print([ylower,yupper])
			#plot_curves.plot_DNAshape_curve_model(sec_d,thr_d,merge_shape_allmodel[k][m],ylower,yupper,input_allpks_files_label[k]+'.W_conv1fc1_data_filtered_p.'+str(m),filter1_size1,subdirectory)
			#plot_curves.plot_DNAshape_curve_model(sec_d,thr_d,merge_shape_allmodel[k][m]*W_fc2_data[0][m],ylowerfc,yupperfc,input_allpks_files_label[k]+'.W_conv1fc1fc2_data_filtered_p.'+str(m),filter1_size1,subdirectory)
			merge_shape_allmodel_shape=merge_shape_allmodel_shape+merge_shape_allmodel[k][m]*W_fc2_data[0][m]
			merge_shape_allmodel_shape_fc2filtered=merge_shape_allmodel_shape_fc2filtered+merge_shape_allmodel[k][m]*W_fc2_data[0][m]*np.mean(h_fc1_data_p,axis=0)[m]
		### plot all merged model shape
		plot_curves.plot_DNAshape_curve(sec_d,thr_d,merge_shape_allmodel_shape,input_allpks_files_label[k]+'.W_conv1fc1fc2_allmerged_p.',filter1_size1)
		plot_curves.plot_DNAshape_curve(sec_d,thr_d,merge_shape_allmodel_shape_fc2filtered,input_allpks_files_label[k]+'.W_conv1fc1fc2_allmerged_p_fc2filtered.',filter1_size1)
		merge_shape_allmodel_shape_fc2filtered_heatmap_p=np.append(merge_shape_allmodel_shape_fc2filtered_heatmap_p, [merge_shape_allmodel_shape_fc2filtered], axis=0)
	#print(merge_shape_allmodel_shape_fc2filtered_heatmap_p[1:])
	### plot grident heatmap
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(merge_shape_allmodel_shape_fc2filtered_heatmap_p[1:],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='bwr',figsize=(20, 10))
	filename0p=os.path.join(subdirectory,'allchannel.W_conv1fc1fc2_allmerged_p_fc2filtered_heatmap.pdf')
	plt.savefig(filename0p,format='pdf')
	plt.clf()
	### save model_weight_matrix_table
	filename0p=os.path.join(subdirectory,'allchannel.W_conv1fc1fc2_allmerged_p_fc2filtered_heatmap.txt')
	weight_model_matrix=open(filename0p,'w')
	for gradients in merge_shape_allmodel_shape_fc2filtered_heatmap_p[1:]:
		#print(gradients)
		for gradient_2d in gradients:
			#print(gradient_2d)
			weight_model_matrix.write(str(gradient_2d)+'\t')
		weight_model_matrix.write('\n')
	weight_model_matrix.close()

	###################################################################
	### save 1st conv layer & 1st fully connected layer weight matrix
	subdirectory='W_conv1fc1_data_filtered_n'
	if os.path.exists('W_conv1fc1_data_filtered_n'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)
	W_conv1_data=sess.run(W_conv1)
	W_fc1_data=np.transpose(sess.run(W_fc1))
	W_fc2_data=np.transpose(sess.run(W_fc2))
	#h_conv1_data_n=sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})
	### initialize dict for storing model shape
	merge_shape_allmodel={}
	for k in range(0,for_d):
		merge_shape_allmodel[k]={}
	### get model shape
	for m in range(0,W_fc1_data.shape[0]):
		W_fc1_matrix_expand=np.repeat(np.transpose(W_fc1_data[m,].reshape([pool_shape1,first_filter_out])),max_pool1,axis=1)
		#W_conv1fc1_matrix=np.dot(W_conv1_data,W_fc1_matrix_expand)
		W_fc1_matrix_expand_filter=np.transpose(np.mean(np.greater(h_conv1_data_n[:,:,0,:],0),axis=0)) * W_fc1_matrix_expand
		W_conv1fc1_matrix_filter=np.dot(W_conv1_data,W_fc1_matrix_expand_filter)
		###
		#print(W_conv1fc1_matrix_filter.shape)
		#print(W_conv1fc1_matrix_filter[:,0,1,0])

		### get merged model shape (matrix diagonal sum) 
		for k in range(0,W_conv1fc1_matrix_filter.shape[2]):
			merge_shape_allmodel[k][m]=np.repeat(0.0,W_conv1fc1_matrix_filter.shape[3]+filter1_size1-1) ### float repeat!!!
			for l in range(0,W_conv1fc1_matrix_filter.shape[3]):
				tmp=merge_shape_allmodel[k][m][l:(l+filter1_size1)]+W_conv1fc1_matrix_filter[:,0,k,l]
				merge_shape_allmodel[k][m][l:(l+filter1_size1)]=tmp	
		### 
		### plot each position's model shape
		for i in range(0,W_conv1fc1_matrix_filter.shape[3]):
			if seq_num!=0:
				### plot seq heatmap in each convolutional filter
				'''
				fig,ax=plt.subplots()
				heatmap=sns.clustermap(np.transpose(W_conv1fc1_matrix_filter[:,0,range(0,seq_num),i]),col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='bwr')
				filename0p=os.path.join(subdirectory,'Model_'+str(m)+'.W_conv1fc1_filter.'+str(i)+'.png')
				plt.savefig(filename0p,format='png')
				plt.clf()
				'''
				### plot shape curve in each convolutional filter
			if shape_num!=0:
				### plot shape curve seperately
				'''
				color=['k','c','m','b','r','g','y']
				for j in range(seq_num,seq_num+shape_num):
					fig,ax=plt.subplots()
					### set y_limit
					axes = plt.gca()
					ymax=np.max(W_conv1fc1_matrix_filter[:,0,[j],:])
					ymin=np.min(W_conv1fc1_matrix_filter[:,0,[j],:])
					axes.set_ylim([ymin-0.01,ymax+0.01])
					### plot each shape in seperate plot
					plt.plot(W_conv1fc1_matrix_filter[:,0,[j],i],color[j-seq_num]+'o') ### MGW
					plt.plot(W_conv1fc1_matrix_filter[:,0,[j],i],color[j-seq_num])
					filename0p=os.path.join(subdirectory,input_allpks_files_label[j]+'.Model_'+str(m)+'.W_conv1fc1_shape_filter.'+str(i)+'.png')
					plt.savefig(filename0p,format='png')
					plt.clf()
				'''
	### get universal y limits
	merge_shape_allmodel_shape_fc2filtered_heatmap_n=[np.repeat(0.0,len(merge_shape_allmodel[k][0]))]
	for k in range(0,W_conv1fc1_matrix_filter.shape[2]):
		ylower=np.min([np.min(merge_shape_allmodel[k][x]) for x in merge_shape_allmodel[k] ])-0.01
		yupper=np.max([np.max(merge_shape_allmodel[k][x]) for x in merge_shape_allmodel[k] ])+0.01
		ylowerfc=np.min([np.min(merge_shape_allmodel[k][x]*W_fc2_data[1][x]) for x in merge_shape_allmodel[k] ])-0.01
		yupperfc=np.max([np.max(merge_shape_allmodel[k][x]*W_fc2_data[1][x]) for x in merge_shape_allmodel[k] ])+0.01
		### plot conv1, fc1, fc2 all merge one model shape
		merge_shape_allmodel_shape=np.repeat(0.0,len(merge_shape_allmodel[k][0]))
		merge_shape_allmodel_shape_fc2filtered=np.repeat(0.0,len(merge_shape_allmodel[k][0]))
		### plot entire model shape
		for m in range(0,W_fc1_data.shape[0]):
			#print([ylower,yupper])
			#plot_curves.plot_DNAshape_curve_model(sec_d,thr_d,merge_shape_allmodel[k][m],ylower,yupper,input_allpks_files_label[k]+'.W_conv1fc1_data_filtered_n.'+str(m),filter1_size1,subdirectory)
			#plot_curves.plot_DNAshape_curve_model(sec_d,thr_d,merge_shape_allmodel[k][m]*W_fc2_data[1][m],ylowerfc,yupperfc,input_allpks_files_label[k]+'.W_conv1fc1fc2_data_filtered_n.'+str(m),filter1_size1,subdirectory)
			merge_shape_allmodel_shape=merge_shape_allmodel_shape+merge_shape_allmodel[k][m]*W_fc2_data[1][m]
			merge_shape_allmodel_shape_fc2filtered=merge_shape_allmodel_shape_fc2filtered+merge_shape_allmodel[k][m]*W_fc2_data[1][m]*np.mean(h_fc1_data_n,axis=0)[m]
		### plot all merged model shape
		plot_curves.plot_DNAshape_curve(sec_d,thr_d,merge_shape_allmodel_shape,input_allpks_files_label[k]+'.W_conv1fc1fc2_allmerged_n.',filter1_size1)
		plot_curves.plot_DNAshape_curve(sec_d,thr_d,merge_shape_allmodel_shape_fc2filtered,input_allpks_files_label[k]+'.W_conv1fc1fc2_allmerged_n_fc2filtered.',filter1_size1)
		merge_shape_allmodel_shape_fc2filtered_heatmap_n=np.append(merge_shape_allmodel_shape_fc2filtered_heatmap_n, [merge_shape_allmodel_shape_fc2filtered], axis=0)
	#print(merge_shape_allmodel_shape_fc2filtered_heatmap_n[1:])
	### plot grident heatmap
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(merge_shape_allmodel_shape_fc2filtered_heatmap_n[1:],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='bwr',figsize=(20, 10))
	filename0p=os.path.join(subdirectory,'allchannel.W_conv1fc1fc2_allmerged_n_fc2filtered_heatmap.pdf')
	plt.savefig(filename0p,format='pdf')
	plt.clf()
	### save model_weight_matrix_table
	filename0p=os.path.join(subdirectory,'allchannel.W_conv1fc1fc2_allmerged_p_fc2filtered_heatmap.txt')
	weight_model_matrix=open(filename0p,'w')
	for gradients in merge_shape_allmodel_shape_fc2filtered_heatmap_p[1:]:
		#print(gradients)
		for gradient_2d in gradients:
			#print(gradient_2d)
			weight_model_matrix.write(str(gradient_2d)+'\t')
		weight_model_matrix.write('\n')
	weight_model_matrix.close()

	#################################
	### plot relative grident heatmap
	merge_shape_allmodel_shape_fc2filtered_heatmap_relative_gradient=merge_shape_allmodel_shape_fc2filtered_heatmap_p-merge_shape_allmodel_shape_fc2filtered_heatmap_n
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(merge_shape_allmodel_shape_fc2filtered_heatmap_relative_gradient[1:],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='bwr',figsize=(20, 10))
	filename0p=os.path.join('allchannel.W_conv1fc1fc2_allmerged_fc2filtered_heatmap_relative_gradient.pdf')
	plt.savefig(filename0p,format='pdf')
	plt.clf()
	filename0p=os.path.join('allchannel.W_conv1fc1fc2_allmerged_fc2filtered_heatmap_relative_gradient.txt')
	weight_model_matrix=open(filename0p,'w')
	for gradients in merge_shape_allmodel_shape_fc2filtered_heatmap_relative_gradient[1:]:
		#print(gradients)
		for gradient_2d in gradients:
			#print(gradient_2d)
			weight_model_matrix.write(str(gradient_2d)+'\t')
		weight_model_matrix.write('\n')
	weight_model_matrix.close()
	###################################################################		
	### save second fully connected layer weight matrix
	subdirectory='W_fc2_data'
	if os.path.exists('W_fc2_data'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)	
	W_fc2_data=np.transpose(sess.run(W_fc2))
	print(W_fc2_data.shape)
	for i in range(0,W_fc2_data.shape[0]):
		fig,ax=plt.subplots()
		plt.plot(W_fc2_data[i,],'bo')
		plt.plot(W_fc2_data[i,],'b')
		plt.ylabel('2nd fully-connected layer weight')
		plt.xlabel('Number of previous layer scanner')
		filename0p=os.path.join(subdirectory,'W_fc2.'+str(i)+'.png')
		plt.savefig(filename0p,format='png')
		plt.clf()

	with file('W_conv1.txt', 'w') as outfile:
		for slice_1d in sess.run(W_conv1):
			for slice_2d in slice_1d:
				np.savetxt(outfile, slice_2d,delimiter='\t')
	bias1=open('Bias_conv1.txt', 'w')
	for slice_1d in sess.run(b_conv1):
		#print(slice_1d)
		bias1.write(str(slice_1d)+'\n')
	bias1.close()
	#print(sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual, y_: ys_visual, keep_prob1: 1.0, keep_prob2: 1.0})[0])
	#print(sess.run(h_conv1,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0}).shape)
	### save the activation state of the first convolutional layer
	#x_image_p=sess.run(x_image,feed_dict={x: matrix_K_L_M_visual_p, y_: ys_visual_p, keep_prob1: 1.0, keep_prob2: 1.0})
	#x_image_n=sess.run(x_image,feed_dict={x: matrix_K_L_M_visual_n, y_: ys_visual_n, keep_prob1: 1.0, keep_prob2: 1.0})

	with file('h_conv1_p.txt', 'w') as outfile:		
		for slice_1d in h_conv1_data_p:
			for slice_2d in slice_1d:
				np.savetxt(outfile, slice_2d,delimiter='\t')

	#np.savetxt('W_fc0.txt', sess.run(W_fc0),delimiter='\t')
	np.savetxt('W_fc1.txt', sess.run(W_fc1),delimiter='\t')
	np.savetxt('W_fc2.txt', sess.run(W_fc2),delimiter='\t')

	### classification result on test data
	#y_conv_result=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
	#classification=sess.run(y_conv_result, feed_dict={x: matrix_K_L_M_test, keep_prob1: 1.0, keep_prob2: 1.0})
	#np.savetxt('classification_y_conv_result.txt', classification,delimiter='\t')

	### write human readable weight matrix (first convolution layer) 
	weight_matrix2human(filter1_size1,'W_conv1.txt','W_conv_1_h.txt')
	### write human readable weight matrix (first fully connected layer) 
	weight_matrix2human(pool_shape1,'W_fc1.txt','W_fc1_h.txt')
	### write human readable weight matrix (second fully connected layer) 
	weight_matrix2human(full_cn_out,'W_fc2.txt','W_fc2_h.txt')

	### plot accuracy heatmap
	accuracy_table_data=np.array(accuracy_table_data)
	plot_curves.plot_accuracy_curve(accuracy_table_data,sec_d,thr_d,for_d,vkmer,seq_num,shape_num,input_allpks_files_label)
	#accuracy_table_data_vkmer2=np.array(accuracy_table_data_vkmer2)
	#plot_curves.plot_accuracy_curve(accuracy_table_data_vkmer2,sec_d,thr_d,for_d,vkmer2,seq_num,shape_num,input_allpks_files_label)

	### plot DNA shape curve
	if shape_num!=0:
		for i in range(seq_num,seq_num+shape_num):
			plot_curves.plot_DNAshape_curve(sec_d,thr_d,matrix_K_L_M_matrix_shape[i],input_allpks_files_label[i],filter1_size1)
			plot_curves.plot_DNAshape_curve(sec_d,thr_d,matrix_K_L_M_random_matrix_shape[i],input_allpks_files_label[i],filter1_size1)
			plot_curves.plot_DNAshape_curve_both(sec_d,thr_d,matrix_K_L_M_matrix_shape[i],matrix_K_L_M_random_matrix_shape[i],input_allpks_files_label[i],filter1_size1)

	### plot the activation state of the first convolutional layer
	print('hohoho')
	subdirectory='h_conv1_curve_pn'
	if os.path.exists('h_conv1_curve_pn'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)	

	print('lolololololo')

	for i in range(0,first_filter_out):
		############### heatmap
		#print(h_conv1_data_p[:,:,:,:].shape)
		if np.sum(h_conv1_data_p[:,:,0,i])!=0:
			fig,ax=plt.subplots()
			heatmap=sns.clustermap(h_conv1_data_p[:,:,0,i],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='OrRd',vmin=0, vmax=np.max([np.max(h_conv1_data_p),np.max(h_conv1_data_n)]))
			filename0p=os.path.join(subdirectory,'accuracy_table_p.'+str(i)+'.png')
			plt.savefig(filename0p,format='png')
			plt.clf()

		############### heatmap
		#print(h_conv1_data_p[:,:,:,i].shape)
		if np.sum(h_conv1_data_n[:,:,0,i])!=0:
			fig,ax=plt.subplots()
			heatmap=sns.clustermap(h_conv1_data_n[:,:,0,i],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='OrRd',vmin=0, vmax=np.max([np.max(h_conv1_data_p),np.max(h_conv1_data_n)]))
			filename0n=os.path.join(subdirectory,'accuracy_table_n.'+str(i)+'.png')
			plt.savefig(filename0n,format='png')
			plt.clf()

		activate_state_p=np.transpose(np.mean(h_conv1_data_p[:,:,0,i],axis=0))
		activate_state_n=np.transpose(np.mean(h_conv1_data_n[:,:,0,i],axis=0))
		activate_state_pn=activate_state_p-activate_state_n
		#print(activate_state_pn)
		plot_curves.plot_hidden_layer_activate_curve(subdirectory,'h_conv1_curve_p.',sec_d,thr_d,activate_state_p,i)
		plot_curves.plot_hidden_layer_activate_curve(subdirectory,'h_conv1_curve_n.',sec_d,thr_d,activate_state_n,i)
		plot_curves.plot_hidden_layer_activate_curve(subdirectory,'h_conv1_curve_pn.',sec_d,thr_d,activate_state_pn,i)

	
	active_heatmap=np.append(np.transpose(np.mean(h_conv1_data_p[:,:,0,:],axis=0)),np.transpose(np.mean(h_conv1_data_n[:,:,0,:],axis=0)), axis=0)
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(active_heatmap,col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='OrRd',vmin=0, vmax=np.max(active_heatmap))
	filename0n=os.path.join(subdirectory,'hconv1_node_active_heatmap.png')
	plt.savefig(filename0n,format='png')
	plt.clf()
	
	active_heatmap=np.transpose(np.mean(h_conv1_data_p[:,:,0,:],axis=0))-np.transpose(np.mean(h_conv1_data_n[:,:,0,:],axis=0))
	#print(active_heatmap)
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(active_heatmap,col_cluster=False,row_cluster=True,method='complete',metric='euclidean',cmap='bwr')
	filename0n=os.path.join(subdirectory,'hconv1_node_relative_active_heatmap.png')
	plt.savefig(filename0n,format='png')
	plt.clf()

	### plot the activation state of the first pooling layer
	print('hohoho')
	subdirectory='h_pool1_curve_pn'
	if os.path.exists('h_pool1_curve_pn'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)	

	print('lolololololo')

	for i in range(0,first_filter_out):
		############### heatmap
		#print(h_conv1_data_p[:,:,:,:].shape)
		if np.sum(h_pool1_data_p[:,:,0,i])!=0:
			fig,ax=plt.subplots()
			heatmap=sns.clustermap(h_pool1_data_p[:,:,0,i],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='OrRd',vmin=0, vmax=np.max([np.max(h_pool1_data_p),np.max(h_pool1_data_n)]))
			filename0p=os.path.join(subdirectory,'accuracy_table_p.'+str(i)+'.png')
			plt.savefig(filename0p,format='png')
			plt.clf()
		############### heatmap
		#print(h_conv1_data_p[:,:,:,i].shape)
		if np.sum(h_pool1_data_n[:,:,0,i])!=0:
			fig,ax=plt.subplots()
			heatmap=sns.clustermap(h_pool1_data_n[:,:,0,i],col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='OrRd',vmin=0, vmax=np.max([np.max(h_pool1_data_p),np.max(h_pool1_data_n)]))
			filename0n=os.path.join(subdirectory,'accuracy_table_n.'+str(i)+'.png')
			plt.savefig(filename0n,format='png')
			plt.clf()

		filename=os.path.join(subdirectory,'h_pool1_curve_pn.'+str(i)+'.pdf')
		fig,ax=plt.subplots()
		activate_state_p=np.transpose(np.mean(h_pool1_data_p[:,:,0,i],axis=0))
		#print(activate_state_p)
		activate_state_n=np.transpose(np.mean(h_pool1_data_n[:,:,0,i],axis=0))
		#print(activate_state_n)
		#activate_state_p=np.mean(h_conv1_data_p,axis=0)[:,:,i]
		#activate_state_n=np.mean(h_conv1_data_n,axis=0)[:,:,i]
		activate_state_pn=activate_state_p-activate_state_n
		axes = plt.gca()
		axes.set_xlim([-(sec_d/max_pool1+2)/2,(sec_d/max_pool1+2)/2])
		plt.grid(True)
		if (sec_d*thr_d*1/max_pool1)%2==0:
			x=range( -int((sec_d*thr_d*1/max_pool1)/2-1), int((sec_d*thr_d*1/max_pool1)/2+1) ) ### start from 1
		else:
			x=range( -int((sec_d*thr_d*1/max_pool1)/2-1), int((sec_d*thr_d*1/max_pool1)/2+2) ) ### start from 1
		axes.set_ylim(   np.min(np.mean(h_pool1_data_p[:,:,0,:],axis=0)-np.mean(h_pool1_data_n[:,:,0,:],axis=0))-0.01,np.max(np.mean(h_pool1_data_p[:,:,0,:],axis=0)-np.mean(h_pool1_data_n[:,:,0,:],axis=0))+0.01  )
		#plt.plot(x,activate_state_n,'ro')
		#plt.plot(x,activate_state_n,'r')
		plt.plot(x,activate_state_pn,'bo')
		plt.plot(x,activate_state_pn,'b')
		plt.ylabel('1st max pooling layer activation states positive & negative')
		plt.xlabel('Distance from Motif Center')
		plt.savefig(filename,format='pdf')
		plt.clf()

	active_heatmap=np.append(np.transpose(np.mean(h_pool1_data_p[:,:,0,:],axis=0)),np.transpose(np.mean(h_pool1_data_n[:,:,0,:],axis=0)), axis=0)
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(active_heatmap,col_cluster=False,row_cluster=False,method='complete',metric='euclidean',cmap='OrRd',vmin=0, vmax=np.max(active_heatmap))
	filename0n=os.path.join(subdirectory,'hpool1_node_active_heatmap.png')
	plt.savefig(filename0n,format='png')
	plt.clf()

	active_heatmap=np.transpose(np.mean(h_pool1_data_p[:,:,0,:],axis=0))-np.transpose(np.mean(h_pool1_data_n[:,:,0,:],axis=0))
	fig,ax=plt.subplots()
	heatmap=sns.clustermap(active_heatmap,col_cluster=False,row_cluster=True,method='complete',metric='euclidean',cmap='bwr')
	filename0n=os.path.join(subdirectory,'hpool1_node_relative_active_heatmap.png')
	plt.savefig(filename0n,format='png')
	plt.clf()

	### plot the activation state of the first fully connected layer
	print('hohoho')
	print(h_fc1_data_p.shape)
	subdirectory='h_fc1_curve_pn'
	if os.path.exists('h_fc1_curve_pn'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)

	filename=os.path.join(subdirectory,'h_fc1_curve_pn.'+str(i)+'.pdf')
	activate_state_p=np.mean(h_fc1_data_p,axis=0)
	activate_state_n=np.mean(h_fc1_data_n,axis=0)
	activate_state_pn=activate_state_p-activate_state_n
	axes = plt.gca()
	plt.grid(True)
	axes.set_ylim([-1.5,1.5])
	plt.plot(activate_state_pn,'bo')
	plt.plot(activate_state_pn,'b')
	plt.ylabel('1st fully connected layer activation states positive')
	plt.xlabel('Distance from Motif Center')
	plt.savefig(filename,format='pdf')
	plt.clf()
	### positive pks
	filename=os.path.join(subdirectory,'h_fc1_curve_p.'+str(i)+'.pdf')
	activate_state_p=np.mean(h_fc1_data_p,axis=0)
	axes = plt.gca()
	plt.grid(True)
	axes.set_ylim([-1.5,1.5])
	plt.plot(activate_state_p,'bo')
	plt.plot(activate_state_p,'b')
	plt.ylabel('1st fully connected layer activation states positive')
	plt.xlabel('Distance from Motif Center')
	plt.savefig(filename,format='pdf')
	plt.clf()
	### negative pks
	filename=os.path.join(subdirectory,'h_fc1_curve_n.'+str(i)+'.pdf')
	activate_state_n=np.mean(h_fc1_data_n,axis=0)
	axes = plt.gca()
	plt.grid(True)
	axes.set_ylim([-1.5,1.5])
	plt.plot(activate_state_n,'bo')
	plt.plot(activate_state_n,'b')
	plt.ylabel('1st fully connected layer activation states positive')
	plt.xlabel('Distance from Motif Center')
	plt.savefig(filename,format='pdf')
	plt.clf()

	### plot the activation state of the first fully connected layer
	print('hohoho')
	subdirectory='y_conv_curve_pn'
	if os.path.exists('y_conv_curve_pn'):
		shutil.rmtree(subdirectory)
	os.mkdir(subdirectory)

	print(np.mean(y_conv_data_p,axis=0))
	filename=os.path.join(subdirectory,'y_conv_curve_pn.'+str(i)+'.pdf')
	activate_state_p=np.mean(y_conv_data_p,axis=0)
	activate_state_n=np.mean(y_conv_data_n,axis=0)
	#activate_state_pn=activate_state_p-activate_state_n
	axes = plt.gca()
	plt.grid(True)
	#axes.set_xlim([-int(full_cn_out/2+2),int(full_cn_out/2+2)])
	axes.set_xlim([-0.5,1.5])
	#x=range( -int(full_cn_out/2), int(full_cn_out/2) )
	#plt.plot(activate_state_n,'ro')
	#plt.plot(activate_state_n,'r')
	plt.plot(activate_state_p,'bo')
	#plt.plot(activate_state_p,'b')
	plt.plot(activate_state_n,'ro')
	#plt.plot(activate_state_n,'r')
	plt.ylabel('1st fully connected layer activation states positive')
	plt.xlabel('Distance from Motif Center')
	plt.savefig(filename,format='pdf')
	plt.clf()
############################################################################
import getopt
import sys
def main(argv):
	try:
		opts, args = getopt.getopt(argv,"hs:t:f:i:e:u:l:o:r:v:a:b:p:n:x:y:k:m:d:j:q:w:z:")
	except getopt.GetoptError:
		print 'one_conv_layer_net.py -s <image d2> -t <image d3> -f <image d4> -i <filter1 out> -e <filter2 out> -u <full connect1 out> -l <filter1 d1> -o <filter1 d2> -r <filter2 d1> -v <filter2 d1> -a iter_num -b batch_size -p train_pos_num -n train_neg_num -x max_pooling_1 -y max_pooling_2 -k visual_k-mer_mutation -m visual_Second_k-mer_mutation -d training_speed -j seq_num -q shape_num -w file_list_all -z file_list_random'
		sys.exit(2)

	for opt,arg in opts:
		if opt=="-h":
			print 'one_conv_layer_net.py -s <image d2> -t <image d3> -f <image d4> -i <filter1 out> -e <filter2 out> -u <full connect1 out> -l <filter1 d1> -o <filter1 d2> -r <filter2 d1> -v <filter2 d1> -a iter_num -b batch_size -p train_pos_num -n train_neg_num -x max_pooling_1 -y max_pooling_2 -k visual_k-mer_mutation -m visual_Second_k-mer_mutation -d training_speed -j seq_num -q shape_num -w file_list_all -z file_list_random'
			sys.exit()
		elif opt=="-s":
			sec_d=int(arg.strip())
		elif opt=="-t":
			thr_d=int(arg.strip())
		elif opt=="-f":
			for_d=int(arg.strip())
		elif opt=="-i":
			first_filter_out=int(arg.strip())
		elif opt=="-e":
			sec_filter_out=int(arg.strip())
		elif opt=="-u":
			full_cn_out=int(arg.strip())
		elif opt=="-l":
			filter1_size1=int(arg.strip())
		elif opt=="-o":
			filter1_size2=int(arg.strip())
		elif opt=="-r":
			filter2_size1=int(arg.strip())
		elif opt=="-v":
			filter2_size2=int(arg.strip())
		elif opt=="-a":
			iter_num=int(arg.strip())
		elif opt=="-b":
			batch_size=int(arg.strip())
		elif opt=="-p":
			train_pos=int(arg.strip())
		elif opt=="-n":
			train_neg=int(arg.strip())
		elif opt=="-x":
			max_pool1=int(arg.strip())
		elif opt=="-y":
			max_pool2=int(arg.strip())
		elif opt=="-k":
			vkmer=int(arg.strip())
		elif opt=="-m":
			vkmer2=int(arg.strip())
		elif opt=="-d":
			training_speed=float(arg.strip())	
		elif opt=="-j":
			seq_num=int(arg.strip())
		elif opt=="-q":
			shape_num=int(arg.strip())	
		elif opt=="-w":
			file_list_all=str(arg.strip())
		elif opt=="-z":
			file_list_random=str(arg.strip())
	one_conv_layer_net(sec_d,thr_d,for_d,first_filter_out,sec_filter_out,full_cn_out,filter1_size1,filter1_size2,filter2_size1,filter2_size2,iter_num,batch_size,train_pos,train_neg,max_pool1,max_pool2,vkmer,vkmer2,training_speed,seq_num,shape_num,file_list_all,file_list_random)

if __name__=="__main__":
	main(sys.argv[1:])


# python /Users/universe/Documents/2016_BG/Pugh_lab/0412_pk_pair/rap1/one_conv_layer_net.py -s 96 -t 1 -f 6 -i 8 -e 16 -u 16 -l 6 -o 1 -r 8 -v 1 -a 300 -b 100 -p 990 -n 2938 -x 2 -y 2 -k 2 -m 5 -d 0.0001 -j 4 -q 2 -w /Users/universe/Documents/2016_BG/Pugh_lab/DeepShape/bin/file_list_allpks6.txt -z /Users/universe/Documents/2016_BG/Pugh_lab/DeepShape/bin/file_list_randompks6.txt
