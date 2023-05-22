import os
import tensorflow as tf




def conv2d_block(input_feat,name, num_filters): 
    print('{}:'.format(name))
    print(tf.keras.backend.shape(input_feat)._inferred_value)
    out = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(input_feat)
    print(tf.keras.backend.shape(out)._inferred_value)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.MaxPooling2D((2, 2))(out)
    print(tf.keras.backend.shape(out)._inferred_value)  
    out = tf.keras.layers.Dropout(0.2)(out) 
    print(tf.keras.backend.shape(out)._inferred_value)
    return out

def conv2dT_block(input_feat,name, num_filters): 
    print('{}:'.format(name))
    print(tf.keras.backend.shape(input_feat)._inferred_value)
    out = tf.keras.layers.Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding="same")(input_feat)
    print('after convT')
    print(tf.keras.backend.shape(out)._inferred_value)
    out = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same")(out)
    print(tf.keras.backend.shape(out)._inferred_value)
    #out = tf.keras.layers.concatenate([convT2, conv2])
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation('relu')(out)
    out = tf.keras.layers.Dropout(0.2)(out)
    print(tf.keras.backend.shape(out)._inferred_value)
    return out



#conv2d model arch (Anomaly)
def model_arch(params_model):
    x_input_dim = (params_model['x_input_dim'])
    r_input_dim = int(params_model['r_input_dim'])
    input1_layer = tf.keras.layers.Input(shape=(x_input_dim), name='x_input')
    input2_layer = tf.keras.layers.Input(shape=(r_input_dim), name='r_input')
    out1 = tf.expand_dims(input1_layer,axis=-1)

    out1 = tf.keras.layers.BatchNormalization()(out1)  
    out1 = conv2d_block(out1,'conv2d_block1', 8)
    out1 = conv2d_block(out1,'conv2d_block2', 2)
    out1 = conv2dT_block(out1,'conv2dT_block1', 2)
    out1 = conv2dT_block(out1,'conv2dT_block2', 8)
    out1 = tf.keras.layers.Conv2D(1, (1,1), padding="same", activation= None)(out1)
    print(tf.keras.backend.shape(out1)._inferred_value)
    return tf.keras.Model(inputs=[input1_layer,input2_layer], outputs=out1, name='Model_Anomaly_CONV2D')



