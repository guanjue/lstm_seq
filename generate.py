	### matrix_K_L_M_all & ys_all
	matrix_K_L_M_all=np.concatenate((matrix_K_L_M,matrix_K_L_M_random))
	ys_all=np.repeat([[np.float32(1),np.float32(0)],[np.float32(0),np.float32(1)]], [matrix_K_L_M.shape[0],matrix_K_L_M_random.shape[0]], axis=0)

	### h_conv1_array_pos_label
	h_conv1_array_all=np.array(sess.run(h_conv1, feed_dict={x: matrix_K_L_M_all, y_: ys_all, keep_prob1: 1.0, keep_prob2: 1.0}))
	y_label_array_all=np.array(sess.run(y_, feed_dict={x: matrix_K_L_M_all, y_: ys_all, keep_prob1: 1.0, keep_prob2: 1.0}))

	h_conv1_array_all_label=[]
	for i in range(0,h_conv1_array_all.shape[0]):
		sample_matrix=np.array(h_conv1_array_all[i,:,0,:])
		sample_label_vector=[]
		#print(sample_matrix.shape)
		for j in range(0,h_conv1_array_all.shape[1]-1):
			#print(np.nonzero(sample_matrix[i,:]))
			tmp_dimer_label=np.nonzero(sample_matrix[j,:])[0]
			sample_label_vector.append(tmp_dimer_label[0])
		h_conv1_array_all_label.append(sample_label_vector)
	write2d_array(h_conv1_array_all_label,'h_conv1_array_all_label.txt')

	### y_label
	write2d_array(y_label_array_all,'y_label_array_all.txt')
