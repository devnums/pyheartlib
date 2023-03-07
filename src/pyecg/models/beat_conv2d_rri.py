import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

##########################################################
def reg():
	return tf.keras.regularizers.l2(l=0.01)

def conv2d_block(inp, name=None, filters=16, kernel_size=(3, 3), bn=True, drate=0.30, pool_size=(0, 0),flatten=True,regularizer=None):
	print('{}:'.format(name))
	output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), 
									padding='valid', activation=None, 
									kernel_regularizer = regularizer)(inp)
	print(tf.keras.backend.shape(output)._inferred_value)
	if bn:
		output = tf.keras.layers.BatchNormalization(axis=-1)(output)
	output = tf.keras.activations.relu(output)
	if drate>0:
		output = tf.keras.layers.Dropout(drate)(output)
	if any(pool_size):
		output = tf.keras.layers.MaxPool2D(pool_size=pool_size)(output)
	if flatten:
		output = tf.keras.layers.Flatten()(output)
	
	print(tf.keras.backend.shape(output)._inferred_value)

	return output



def model_arch(params_model):
	conv_input_dim = int(params_model['conv_input_dim'])
	r_input_dim = int(params_model['conv_input_dim'])
	num_classes = int(params_model['num_classes'])
	regularizer = int(params_model['regularizer'])
	input1_layer = tf.keras.layers.Input(shape=(conv_input_dim), name='conv_input')
	input2_layer = tf.keras.layers.Input(shape=(beatinfo_input_dim), name='r_input')

	print('\ninput1:{}\ninput2:{}:'.format(tf.keras.backend.shape(input1_layer)._inferred_value,
												tf.keras.backend.shape(input2_layer)._inferred_value))
	#input: (None,d1,d2)
	out = tf.expand_dims(input1_layer, axis=-1)
	out = tf.keras.layers.BatchNormalization(axis=-1)(out)

	out1 = conv2d_block(out, name='block1', filters=32, kernel_size=(3,3), bn=True, drate=0.5, pool_size=(4,4),flatten=True)
	out2 = conv2d_block(out, name='block2', filters=32, kernel_size=(5,5), bn=True, drate=0.5, pool_size=(4,4),flatten=True)
	out3 = conv2d_block(out, name='block3', filters=32, kernel_size=(11,11), bn=True, drate=0.5, pool_size=(4,4),flatten=True)

	#out = out1
	out = tf.keras.layers.concatenate([out1,out2,out3],axis=-1)
	print('\n'+ str(tf.keras.backend.shape(out)._inferred_value))

	out = tf.keras.layers.Dense(60,activation='relu')(out)
	out = tf.keras.layers.concatenate([out,input2_layer],axis=-1)
	print('\n'+ str(tf.keras.backend.shape(out)._inferred_value))

	out = tf.keras.layers.Dense(1024,activation='relu')(out)
	out = tf.keras.layers.Dropout(0.30)(out)
	print('\n'+ str(tf.keras.backend.shape(out)._inferred_value))

	out = tf.keras.layers.Dense(512,activation='relu')(out)
	print('\n'+ str(tf.keras.backend.shape(out)._inferred_value))

	out = tf.keras.layers.Dropout(0.30)(out)
	out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
	
	return tf.keras.Model(inputs=[input1_layer,input2_layer], outputs=out, name='Model_Conv2d_RR')






