import os
import tensorflow as tf




def conv1d_block(input_feat, num_filters, name=None): 
    print('{}:'.format(name))
    out = tf.keras.layers.Conv1D(num_filters, 32, padding="same")(input_feat)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.MaxPooling1D(2)(out)  
    out = tf.keras.layers.Dropout(0.2)(out) 
    print(tf.keras.backend.shape(out)._inferred_value)
    return out

def conv1dT_block(input_feat, num_filters, name=None): 
    print('{}:'.format(name))
    out = tf.keras.layers.Conv1DTranspose(num_filters, 32, strides=2, padding="same")(input_feat)
    out = tf.keras.layers.Conv1D(num_filters, 32, padding="same")(out)
    #out = tf.keras.layers.concatenate([convT2, conv2])
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    print(tf.keras.backend.shape(out)._inferred_value)
    return out

#conv1d model arch (Anomaly)
def model_arch(params_model):
    x_input_dim = int(params_model['x_input_dim'])
    r_input_dim = int(params_model['r_input_dim'])
    input1_layer = tf.keras.layers.Input(shape=(x_input_dim), name='x_input')
    input2_layer = tf.keras.layers.Input(shape=(r_input_dim), name='r_input')
    
    out = tf.expand_dims(input1_layer, axis=-1)
    out = tf.keras.layers.BatchNormalization()(out)  
    out = conv1d_block(out, 4, name='block1')
    out = conv1d_block(out, 8, name='block2')
    out = conv1dT_block(out, 8, name='block1T')
    out = conv1dT_block(out, 4, name='block2T')
    out = tf.keras.layers.Conv1D(1, 1, padding="same", activation= None)(out)
    print(tf.keras.backend.shape(out)._inferred_value)
    out = tf.keras.backend.squeeze(out, axis=-1)
    print(tf.keras.backend.shape(out)._inferred_value)

    return tf.keras.Model(inputs=[input1_layer,input2_layer], outputs=out, name='Model_Anomaly_CONV1D')



