import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import *
tf.get_logger().setLevel('ERROR')


##########################################################
def print_layer(layer):
	'''Prints layer output dim and its name or class name'''
	l_out_shape = tf.keras.backend.shape(layer)._inferred_value
	l_name = layer.name
	print('\nLayer: {} -->  Output shape: {}'.
			format(str(l_name).upper(), str(l_out_shape))) 

def reg():
	return tf.keras.regularizers.l2(l=0.01)

#Conv Block
def conv1d_block(
	inp, 
	name=None, 
	filters=64, 
	kernel_size=64, 
	bn=True, 
	drate=0.30, 
	pool_size=0,
	flatten=True,
	regularizer=None):
	
	print('{}:'.format(name))
	output = Conv1D(
		filters=filters, 
		kernel_size=kernel_size, 
		strides=1, 
		padding='valid', 
		activation=None, 
		kernel_regularizer = regularizer)(inp)
	
	if bn:
		output = BatchNormalization(axis=-1)(output)
	output = tf.keras.activations.relu(output)
	if drate>0:
		output = Dropout(drate)(output)
	if pool_size>0:
		output = MaxPool1D(pool_size=pool_size)(output)
	if flatten:
		output = Flatten()(output)

	print_layer(output)
	return output


#Model Architecture
def model_arch(params_model):
	x_input_dim = int(params_model['x_input_dim'])
	r_input_dim = int(params_model['r_input_dim'])
	num_classes = int(params_model['num_classes'])

	input1_layer = Input(shape=(x_input_dim), name='x_input')
	input2_layer = Input(shape=(r_input_dim), name='r_input')

	#input: (None,seq_len)  --> (None,seq_len,1) 
	out = tf.expand_dims(input1_layer, axis=-1)
	out = BatchNormalization(axis=-1)(out)

	out1 = conv1d_block(out,name='block1', filters=2, kernel_size=64, bn=True, drate=0.5, pool_size=4,flatten=True)
	out2 = conv1d_block(out,name='block2', filters=2, kernel_size=32, bn=True, drate=0.5, pool_size=4,flatten=True)
	out3 = conv1d_block(out,name='block3', filters=2, kernel_size=16, bn=True, drate=0.5, pool_size=4,flatten=True)
	out4 = conv1d_block(out,name='block4', filters=2, kernel_size=8, bn=True, drate=0.5, pool_size=4,flatten=True)
	out5 = conv1d_block(out,name='block5', filters=2, kernel_size=4, bn=True, drate=0.5, pool_size=4,flatten=True)
	
	out = concatenate([out1,out2,out3,out4,out5],axis=-1)
	out = Dropout(0.30)(out)
	print_layer(out)
	
	out = Dense(512,activation='relu')(out)
	out = Dropout(0.30)(out)
	print_layer(out)

	out = Dense(10,activation='relu')(out)
	out = concatenate([out,input2_layer],axis=-1)
	out = Dropout(0.30)(out)
	print_layer(out)

	out = Dense(num_classes, activation="softmax")(out)
	
	return tf.keras.Model(inputs=[input1_layer, input2_layer], outputs=out, name='Model_Conv1d_RR')






