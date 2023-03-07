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

def conv1d_block(inp, name=None, filters=64, kernel_size=64, bn=True, drate=0.30, pool_size=0,flatten=True,regularizer=None):
	#print('{}:'.format(name))
	output = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, 
									padding='same', activation=None, 
									kernel_regularizer = regularizer)(inp)
	if bn:
		output = BatchNormalization(axis=-1)(output)
	output = relu(output)  #RELU
	if drate>0:
		output = Dropout(drate)(output)
	if pool_size>0:
		output = MaxPool1D(pool_size=pool_size)(output)
	if flatten:
		output = Flatten()(output)
	
	#print(tf.keras.backend.shape(output)._inferred_value)
	print_layer(output)
	return output 



def model_arch(params_model):
	x_input_dim = int(params_model['x_input_dim'])
	num_classes = int(params_model['num_classes'])
	out_seq_len = int(params_model['out_seq_len'])

	input1_layer = Input(shape=(x_input_dim), name='x_input_dim')
	#input1_layer = Input(shape=(150,14), name='x_input_dim')
	print_layer(input1_layer)

	#input: (None,seq_len)  --> (None,seq_len,1) 
	out = tf.expand_dims(input1_layer, axis=-1)
	print_layer(out)

	out = BatchNormalization()(out)
	print_layer(out)

	#out1 = conv1d_block(out,name='block1', filters=8, kernel_size=64, bn=True, drate=0.5, pool_size=72,flatten=False)
	#out2 = conv1d_block(out,name='block2', filters=8, kernel_size=32, bn=True, drate=0.5, pool_size=72,flatten=False)
	#out3 = conv1d_block(out,name='block3', filters=8, kernel_size=16, bn=True, drate=0.5, pool_size=72,flatten=False)
	#out4 = conv1d_block(out,name='block4', filters=8, kernel_size=8, bn=True, drate=0.5, pool_size=72,flatten=False)
	out = conv1d_block(out,name='block5', filters=16, kernel_size=36, bn=True, drate=0.5, pool_size=144,flatten=False)
	#out = concatenate([out1,out2,out3,out4,out5],axis=-1)
	#out = Dropout(0.30)(out)
	print_layer(out)


	#out = conv1d_block(out,name='block2', filters=8, kernel_size=38, bn=True, drate=0.2, pool_size=4,flatten=False)
	#out = conv1d_block(out,name='block3', filters=8, kernel_size=38, bn=True, drate=0.2, pool_size=4,flatten=False)
	#out = conv1d_block(out,name='block4', filters=1, kernel_size=3, bn=True, drate=0.2, pool_size=4,flatten=True)

	#squared
	#sqr = tf.math.square(out)

	#avg_sqr = AveragePooling1D(pool_size=36)(sqr)
	#print_layer(out)
	#max_sqr = MaxPool1D(pool_size=36)(sqr)
	#print_layer(out)

	out = Bidirectional(LSTM(128, return_sequences=True))(out)
	out = Dropout(0.5)(out)
	print_layer(out)
	#out = Bidirectional(LSTM(64, return_sequences=True))(out)
	#out = Dropout(0.2)(out)
	#print_layer(out)

	#out =  LSTM(128, return_sequences=True)(out)
	#print_layer(out)


	#out = Dense(out_seq_len, activation=None)(out)
	#print_layer(out)

	#out = RepeatVector(out_seq_len)(out)
	#print_layer(out)

	#conv1d = Conv1D(filters=8, 
	#				kernel_size=5, strides=1, padding='same')
	#dens = Dense(256,activation='relu')
	#out = TimeDistributed(dens)(out)
	#print_layer(out)
	

	#dns = Dense(num_classes, activation="sigmoid")
	#out = TimeDistributed(dns)(out)
	#print_layer(out)


	#out = Dense(1, activation="sigmoid")(out)
	#print_layer(out)

	out = Dense(64,activation='relu')(out)
	out = Dropout(0.5)(out)
	
	out = Dense(num_classes, activation="softmax")(out)
	print_layer(out)
	
	return tf.keras.Model(inputs=input1_layer, outputs=out, name='Model_Conv1d_Rpeak')






