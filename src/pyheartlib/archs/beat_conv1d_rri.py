import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

num_threads = 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.set_soft_device_placement(True)


##########################################################
def reg():
	return tf.keras.regularizers.l2(l=0.01)


def model_arch(conv_input_dim=256,beatinfo_input_dim=4, num_classes=5, regularizer=None):
	input1_layer = tf.keras.layers.Input(shape=(conv_input_dim), name='conv_input')
	input2_layer = tf.keras.layers.Input(shape=(beatinfo_input_dim), name='beatinfo_input')

	#input: (None,seq_len)  --> (None,seq_len,1) 
	out = tf.expand_dims(input1_layer, axis=-1)
	out = tf.keras.layers.BatchNormalization(axis=-1)(out)
	out = tf.keras.layers.Conv1D(filters=32, kernel_size=32, strides=1, 
									padding='valid', activation=None, 
									kernel_regularizer = regularizer)(out)
	out = tf.keras.layers.BatchNormalization(axis=-1)(out)
	out = tf.keras.activations.relu(out)
	#out = tf.keras.layers.Dropout(0.30)(out)
	#out = tf.keras.layers.MaxPool1D(pool_size=2)(out)

	print(tf.keras.backend.shape(out))
	out = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, 
									padding='valid', activation=None,
									kernel_regularizer = regularizer)(out)
	out = tf.keras.layers.BatchNormalization(axis=-1)(out)
	out = tf.keras.activations.relu(out)
	#out = tf.keras.layers.Dropout(0.30)(out)
	#out = tf.keras.layers.MaxPool1D(pool_size=2)(out)

	print('\n')
	print(tf.keras.backend.shape(out))

	#out = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, 
	#								padding='valid', activation=None,
	#								kernel_regularizer = regularizer)(out)
	#out = tf.keras.layers.BatchNormalization(axis=-1)(out)
	#out = tf.keras.activations.relu(out)
	#out = tf.keras.layers.Dropout(0.30)(out)
	#out = tf.keras.layers.MaxPool1D(pool_size=2)(out)


	#print('\n')
	#print(tf.keras.backend.shape(out))

	#out = tf.keras.layers.Conv1D(filters=32, kernel_size=16, strides=1, 
	#								padding='valid', activation=None,
	#								kernel_regularizer = regularizer)(out)
	#out = tf.keras.layers.BatchNormalization(axis=-1)(out)
	#out = tf.keras.activations.relu(out)
	#out = tf.keras.layers.Dropout(0.30)(out)

	#print('\n')
	#print(tf.keras.backend.shape(out))
	
	out = tf.keras.layers.MaxPool1D(pool_size=4)(out)
	out = tf.keras.layers.Flatten()(out)
	print('\n')
	print(tf.keras.backend.shape(out))	
	out = tf.keras.layers.Dropout(0.30)(out)
	
	out = tf.keras.layers.Dense(4,activation='relu')(out)
	out = tf.keras.layers.concatenate([out,input2_layer],axis=-1)
	print('\n')
	print(tf.keras.backend.shape(out))	

	out = tf.keras.layers.Dense(1024,activation='relu')(out)
	out = tf.keras.layers.Dropout(0.30)(out)
	print('\n')
	print(tf.keras.backend.shape(out))

	out = tf.keras.layers.Dense(512,activation='relu')(out)
	print('\n')
	print(tf.keras.backend.shape(out))

	out = tf.keras.layers.Dropout(0.30)(out)
	out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)
	
	return tf.keras.Model(inputs=[input1_layer,input2_layer], outputs=out)














############################################################
class Conv1DClassifier(tf.keras.Model):
  def __init__(self, seq_len, num_classes, drate=0.15):
    super(Conv1DClassifier, self).__init__()
    
    self.ln = tf.keras.layers.LayerNormalization(axis=-2)
    self.bn = tf.keras.layers.BatchNormalization(axis=-2)
    self.conv1D1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1,
                                         padding='valid', activation='relu')
    self.conv1D2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1,
                                      padding='valid', activation='relu')
    self.maxpooling = tf.keras.layers.MaxPool1D(pool_size=2)
    self.flatten = tf.keras.layers.Flatten()
    self.dropout = tf.keras.layers.Dropout(drate)
    self.dropout1 = tf.keras.layers.Dropout(drate)
    self.dense1 = tf.keras.layers.Dense(128,activation='relu')
    self.final_out = tf.keras.layers.Dense(num_classes, activation="softmax")
    
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