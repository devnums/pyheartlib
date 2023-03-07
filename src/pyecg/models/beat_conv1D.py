import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import *

tf.get_logger().setLevel('ERROR')

# num_threads = 1
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.config.set_soft_device_placement(True)


##########################################################
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
	output = tf.keras.activations.relu(output)
	if drate>0:
		output = Dropout(drate)(output)
	if pool_size>0:
		output = MaxPool1D(pool_size=pool_size)(output)
	if flatten:
		output = Flatten()(output)
	#print(tf.keras.backend.shape(output)._inferred_value)
	print_layer(output)
	return output 

#Model Architecture
def model_arch(params_model):
	x_input_dim = int(params_model['x_input_dim'])
	#r_input_dim = int(params_model['r_input_dim'])
	num_classes = int(params_model['num_classes'])

	input_layer = Input(shape=(x_input_dim))
	#input: (None,seq_len)  --> (None,seq_len,1) 
	out = tf.expand_dims(input_layer, axis=-1)
	out = BatchNormalization(axis=-1)(out)

	out = conv1d_block(out,name='block1', filters=8, kernel_size=5, bn=True, drate=0.2, pool_size=4,flatten=False)

	out = Flatten()(out)
	print_layer(out)
	out = Dropout(0.30)(out)
	out = Dense(25,activation='relu')(out)
	print_layer(out)
	out = Dropout(0.30)(out)
	out = Dense(num_classes, activation="softmax")(out)
	return tf.keras.Model(inputs=input_layer, outputs=out)














############################################################
class Conv1DClassifier(tf.keras.Model):
  def __init__(self, seq_len, num_classes, drate=0.15):
    super(Conv1DClassifier, self).__init__()
    
    self.ln = LayerNormalization(axis=-2)
    self.bn = BatchNormalization(axis=-2)
    self.conv1D1 = Conv1D(filters=32, kernel_size=3, strides=1,
                                         padding='valid', activation='relu')
    self.conv1D2 = Conv1D(filters=64, kernel_size=3, strides=1,
                                      padding='valid', activation='relu')
    self.maxpooling = MaxPool1D(pool_size=2)
    self.flatten = Flatten()
    self.dropout = Dropout(drate)
    self.dropout1 = Dropout(drate)
    self.dense1 = Dense(128,activation='relu')
    self.final_out = Dense(num_classes, activation="softmax")
    
  def call(self, x, training):
    out =  tf.expand_dims(x, axis=-1) #(batch_size,seq_len) ---> (batch_size,seq_len,1)
    out = self.bn(out)
    out = self.conv1D1(out)
    out = self.conv1D2(out) ###  
    out = self.dropout1(out)
    out = self.maxpooling(out)
    out = self.flatten(out)
    out = self.dropout(out)
    out = self.dense1(out)
    out = self.final_out(out)
    return out

'''
batch_size = 1;seq_len = 256;num_classes = 15
cls = Conv1DClassifier(seq_len, num_classes, drate=0.15)
x=tf.random.uniform((batch_size,seq_len))
##x=tf.ones(((batch_size,seq_len,1)))
print(x)
cls.compile()
cls(x)
'''