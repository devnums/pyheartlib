import os
import tensorflow as tf


#FCN model arch (Anomaly)
def model_arch(params_model):
	x_input_dim = int(params_model['x_input_dim'])
	#r_input_dim = int(params_model['r_input_dim'])
	input1_layer = tf.keras.layers.Input(shape=(x_input_dim), name='x_input')
	#input2_layer = tf.keras.layers.Input(shape=(r_input_dim), name='r_input')
	
	out1 = input1_layer
	#out2 = input2_layer
	
	out1 = tf.keras.layers.BatchNormalization()(out1)
	#out2 = tf.keras.layers.BatchNormalization()(out2)  

	out1 = tf.keras.layers.Dropout(0.5)(out1)


	#out = tf.keras.layers.Dense(32, activation= 'relu')(input1_layer)
	#out = tf.keras.layers.Dropout(0.1)(out)
	#out = tf.keras.layers.Dense(8, activation= 'relu')(out)
	#out = tf.keras.layers.Dense(32, activation= 'relu')(out)
	#out = tf.keras.layers.Dropout(0.1)(out)

	dim_latent = params_model['dim_latent']
	out1 = tf.keras.layers.Dense(dim_latent, activation= 'sigmoid', name='latent')(out1)
	#out2 = tf.keras.layers.Dense(2, activation= 'relu')(out2)
	out1 = tf.keras.layers.BatchNormalization()(out1)
	#out2 = tf.keras.layers.BatchNormalization()(out2)
	
	out1 = tf.keras.layers.Dropout(0.5)(out1)
	#out2 = tf.keras.layers.Dropout(0.1)(out2)

	#out_concat = tf.keras.layers.concatenate([out1, out2])

	#out1 = tf.keras.layers.Dense(x_input_dim, activation= None)(out_concat)
	#out2 = tf.keras.layers.Dense(r_input_dim, activation= None)(out_concat)

	out1 = tf.keras.layers.Dense(x_input_dim, activation= None)(out1)
	#out2 = tf.keras.layers.Dense(r_input_dim, activation= None)(out2)

	return tf.keras.Model(inputs= input1_layer, outputs=out1, name='Model_Anomaly_FCN')


