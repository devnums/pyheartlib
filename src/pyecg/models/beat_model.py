import tensorflow as tf
from tensorflow.keras.layers import *



def reg():
	return tf.keras.regularizers.l2(l=0.01)

def conv2d_block(inp, name=None, filters=16, kernel_size=(3, 3), bn=True, drate=0.30, pool_size=(0, 0),flatten=True,regularizer=None):
	print('{}:'.format(name))
	output = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), 
									padding='valid', activation=None, 
									kernel_regularizer = regularizer)(inp)
	print(tf.keras.backend.shape(output)._inferred_value)
	if bn:
		output = BatchNormalization(axis=-1)(output)
	output = tf.keras.activations.relu(output)
	if drate>0:
		output = Dropout(drate)(output)
	if any(pool_size):
		output = MaxPool2D(pool_size=pool_size)(output)
	if flatten:
		output = Flatten()(output)
	
	print(tf.keras.backend.shape(output)._inferred_value)

	return output



def model_arch(params_model):
	x_input_dim = int(params_model['x_input_dim'])
	r_input_dim = int(params_model['r_input_dim'])
	num_classes = int(params_model['num_classes'])

	input1_layer = Input(shape=(x_input_dim), name='x_input')
	input2_layer = Input(shape=(r_input_dim), name='r_input')

	#input: (None,d1,d2)
	out = tf.expand_dims(input2_layer, axis=-1)
	out = BatchNormalization(axis=-1)(out)
	out = Flatten()(out)

	out = Dense(256,activation='relu')(out)
	out = Dropout(0.50)(out)

	out = Dense(128,activation='relu')(out)

	out = Dropout(0.30)(out)
	out = Dense(num_classes, activation="softmax")(out)
	
	return tf.keras.Model(inputs=[input1_layer,input2_layer], outputs=out, name='Model_Beat')






