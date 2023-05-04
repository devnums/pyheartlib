import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *





def print_layer(layer):
	'''Prints layer output dim and its name or class name'''
	l_out_shape = tf.keras.backend.shape(layer)._inferred_value
	l_name = layer.name
	#l_in_shape = layer.input_shape
	#l_out_shape = layer.output_shape
	#print('\nLayer: {} --> Input shape: {}, Output shape: {}'.
	#		format(str(l_name), str(l_in_shape) , str(l_out_shape))) 
	print('\nLayer: {} -->  Output shape: {}'.
			format(str(l_name).upper(), str(l_out_shape))) 


def reg():
	return tf.keras.regularizers.l2(l=0.01)


def conv1d_block(inp, name=None, filters=64, kernel_size=64, strides=1, 
							bn=True, drate=0.30, pool_size=0,flatten=True,regularizer=None):

	#print('{}:'.format(name))
	output = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, 
									padding='valid', activation=None, 
									kernel_regularizer = regularizer)(inp)
	if bn:
		output = BatchNormalization(axis=-1)(output)

	output = relu(output)

	if drate>0:
		output = Dropout(drate)(output)
	if pool_size>0:
		output = MaxPool1D(pool_size=pool_size)(output)
	if flatten:
		output = Flatten()(output)
	
	#print(tf.keras.backend.shape(output)._inferred_value)
	return output 


def model_arch(params_model):
	x_input_dim = int(params_model['x_input_dim'])
	num_classes = int(params_model['num_classes'])
	#input_layer = Input(shape=(x_input_dim), name='x_input_dim')

	input1 = Input(shape=(x_input_dim),name='input_waveform')
	input2 = Input(shape=(20),name='input_rri')
	input3 = Input(shape=(9),name='input_rri_feats')
	input_layer = [input1,input2,input3]


	#import numpy as np
	#input_layer = np.asarray(input_layer[1]).astype(np.float)
	#tf.convert_to_tensor(input_layer[1], dtype=tf.float32)
	
	#input: (None,seq_len)  --> (None,seq_len,1) 
	#out1 = tf.expand_dims(input_layer[0], axis=-1)
	#out1 = BatchNormalization(axis=-1)(out1)
	#print_layer(out1)


	#out = MaxPool1D(pool_size=2)(out)
	#print_layer(out)

	#out1 = conv1d_block(out1, filters=16, kernel_size=28, strides=1,bn=True, drate=0.2, pool_size=4,flatten=False)
	#print_layer(out1)
	#out1 = conv1d_block(out1, filters=16, kernel_size=28, strides=1,bn=True, drate=0.2, pool_size=4,flatten=False)
	#print_layer(out1)
	#out1 = conv1d_block(out1, filters=16, kernel_size=28, strides=1,bn=True, drate=0.2, pool_size=4,flatten=False)
	#print_layer(out1)
	#out1 = conv1d_block(out1, filters=16, kernel_size=28, strides=1,bn=True, drate=0.2, pool_size=4,flatten=False)
	#print_layer(out1)

	#out1 = Flatten()(out1)
	#print_layer(out1)

	#sqr = out**2
	#out = concatenate([out,sqr],axis=-1)
	#print_layer(out)


	#out1 = AveragePooling1D(pool_size=28)(out)
	#print_layer(out1)

	#out2 = MaxPool1D(pool_size=28)(out)
	#print_layer(out2)

	#out = concatenate([out1,out2],axis=-1)
	#print_layer(out)


	out2 = tf.expand_dims(input_layer[2], axis=-1)
	out2 = BatchNormalization(axis=-1)(out2)
	print_layer(out2)
	#tf.keras.backend.print_tensor(out, message='out:')

	#out2 = Masking(mask_value=0.)(out2)
	#tf.keras.backend.print_tensor(out, message='mask:')

	#out = BatchNormalization(axis=-1)(out)
	#out2 = Bidirectional(LSTM(16, return_sequences=False))(out2)
	#out2 = Dropout(0.2)(out2)
	#print_layer(out2)


	#out = BatchNormalization(axis=-1)(out)
	#out = Flatten()(out)
	#print_layer(out)
	#out = Dropout(0.2)(out)

	#out = concatenate([out1,out2],axis=-1)
	#print_layer(out)
	out = out2
	out = Flatten()(out)

	#out = BatchNormalization(axis=-1)(out)
	out = Dense(2048,activation='relu')(out)
	out = Dropout(0.5)(out)
	print_layer(out)

	out = Dense(512,activation='relu')(out)
	out = Dropout(0.5)(out)
	print_layer(out)

	out = Dense(256,activation='relu')(out)
	print_layer(out)

	out = Dropout(0.5)(out)
	out = Dense(num_classes, activation="softmax")(out)
	print_layer(out)
	
	return tf.keras.Model(inputs=input_layer, outputs=out, name='Model_Conv1d_ARR')






