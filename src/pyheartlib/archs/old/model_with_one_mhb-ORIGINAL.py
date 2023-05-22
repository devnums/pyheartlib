#Modified https://www.tensorflow.org/text/tutorials/transformer
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


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model):
    super(MultiHeadAttention, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % num_heads == 0, "d_model must be integer divisible by num_heads"
    self.depth = d_model // num_heads

    self.WQ = tf.keras.layers.Dense(d_model,use_bias=False)
    self.WK = tf.keras.layers.Dense(d_model,use_bias=False)
    self.WV = tf.keras.layers.Dense(d_model,use_bias=False)
    self.WO = tf.keras.layers.Dense(d_model, use_bias=False)

  def dotProductAttention(self,q,k,v):
    qk_prod = tf.matmul(q,k,transpose_b= True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attn_score = qk_prod/tf.math.sqrt(dk)
    attn_weights = tf.nn.softmax(scaled_attn_score, axis=-1)
    output = tf.matmul(attn_weights, v)
    return output, attn_weights

  def splitHeads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, q, k, v):
    batch_size = tf.shape(q)[0]

    q = self.WQ(q)  # (batch_size, seq_len, d_model)
    k = self.WK(k)  # (batch_size, seq_len, d_model)
    v = self.WV(v)  # (batch_size, seq_len, d_model)

    q = self.splitHeads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.splitHeads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.splitHeads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = self.dotProductAttention(q, k, v)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.WO(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

#x=tf.random.uniform((1,2,4))
#MultiHeadAttention(2,6)(x,x,x)


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EmbeddingLayer, self).__init__()

    def build(self,input_shape):
      self.seq_len = input_shape[-2]
      self.emded_dim = input_shape[-1]
      self.position_emb = tf.keras.layers.Embedding(input_dim=self.seq_len, output_dim=self.emded_dim)

    def call(self, x):
      # x is the input to the network (...,seq_len,emded_dim) 
      positions_emb = self.position_emb(tf.range(0,self.seq_len))
      return x + positions_emb
#EmbeddingLayer()(tf.zeros((2, 3, 4)))



class MultiHeadAttentionClassifier(tf.keras.Model):
    def __init__(self, seq_len, d_model, num_heads, num_classes, drate=0.15, name="Model3"):
        super(MultiHeadAttentionClassifier, self).__init__()
        self.d_model = d_model
        self.emb = EmbeddingLayer()
        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)
        self.mha3 = MultiHeadAttention(num_heads, d_model)
        self.mha4 = MultiHeadAttention(num_heads, d_model)
        self.mha5 = MultiHeadAttention(num_heads, d_model)
        
        self.layernorm1a = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-5)

        self.layernorm1 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-5) #along d_model axis
        self.layernorm2 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-5)
        self.layernorm3 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-5)
        self.layernorm4 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-5)
        self.layernorm5 = tf.keras.layers.LayerNormalization(axis = -1, epsilon=1e-5)


        self.dropout1 = tf.keras.layers.Dropout(drate)
        self.dropout2 = tf.keras.layers.Dropout(drate)
        self.dropout3 = tf.keras.layers.Dropout(drate)
        self.dropout4 = tf.keras.layers.Dropout(drate)
        self.dropout5 = tf.keras.layers.Dropout(drate)
        self.dropout6 = tf.keras.layers.Dropout(drate)

        self.bh = tf.keras.layers.BatchNormalization()
        
        #self.ffn = tf.keras.Sequential(
         #   [
          #    tf.keras.layers.Dense(10, activation='relu'),
           #   tf.keras.layers.Dense(d_model)
            #])        

        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')

        self.flatten = tf.keras.layers.Flatten()


        self.final_out = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, x, training):
        assert x.shape[-1] == self.d_model, 'embed_dim of the input must be equal to d_model'
        
        x = self.layernorm1a(x)

        emb_out = self.emb(x)     #(batch_size,seq_len,emded_dim)  ---> (batch_size,seq_len,emded_dim)
       
        attn_out1,_ = self.mha1(emb_out,emb_out,emb_out)   #each input (batch_size,seq_len,emded_dim)  ---> (batch_size, seq_len_q, d_model)
        attn_out1 = self.dropout1(attn_out1, training=training)
        norm_out1 = self.layernorm1(attn_out1 + x)    #(batch_size, seq_len, d_model)

        #attn_out2,_ = self.mha2(emb_out,emb_out,emb_out)   #each input (batch_size,seq_len,emded_dim)  ---> (batch_size, seq_len_q, d_model)
        #attn_out2 = self.dropout2(attn_out2, training=training)
        #norm_out2 = self.layernorm2(attn_out2 + x)    #(batch_size, seq_len, d_model)

        #attn_out3,_ = self.mha3(emb_out,emb_out,emb_out)   #each input (batch_size,seq_len,emded_dim)  ---> (batch_size, seq_len_q, d_model)
        #attn_out3 = self.dropout3(attn_out3, training=training)
        #norm_out3 = self.layernorm3(attn_out3 + x)    #(batch_size, seq_len, d_model)


        #attn_out4,_ = self.mha4(norm_out1, norm_out2, norm_out3)#each input (batch_size,seq_len,emded_dim)-->(batch_size, seq_len_q,d_model)
        #attn_out4 = self.dropout4(attn_out4, training=training)
        #norm_out4 = self.layernorm4(attn_out4)    #(batch_size, seq_len, d_model)
        
        #attn_out5,_ = self.mha5(norm_out4, norm_out4, norm_out4)#each input (batch_size,seq_len,emded_dim)-->(batch_size, seq_len_q,d_model)
        #attn_out5 = self.dropout5(attn_out5, training=training)
        #norm_out5 = self.layernorm5(attn_out5)    #(batch_size, seq_len, d_model)
        
        #norm_out5 = tf.keras.layers.Reshape((x.shape[-2], x.shape[-1]))(norm_out5)
        #--------------------------------------------------------------------------
        norm_out1 = self.flatten(norm_out1)   #(batch_size, seq_len*d_model)
        #norm_out2 = self.flatten(norm_out2)   #(batch_size, seq_len*d_model)
        #norm_out3 = self.flatten(norm_out3)   #(batch_size, seq_len*d_model)
        #print(norm_out1.shape)
        
        #concat_out = tf.keras.layers.concatenate([norm_out1,norm_out2,norm_out3], axis=-1) #axis -1   (batch_size, 3*seq_len*d_model)
        #print(concat_out.shape)

        #--------------------------------------------------------------------------
        #out = self.dense1(norm_out4)     #(batch_size, seq_len, 1024)
        #out = self.dropout5(out, training=training)
        #out = self.layernorm5(out)
        out = norm_out1 #self.flatten(norm_out5)   #(batch_size, seq_len*d_model)
        #out = tf.math.reduce_mean(norm_out4, axis=-2) # (batch_size, d_model)
        #out = self.dense2(out)    # (batch_size, 128)
        out = self.dropout6(out, training=training) 
        output = self.final_out(out)  #(batch_size, num_classes) 


        #print('It is 5C!')

        return output
        
        
"""
batch_size = 2;seq_len = 7;d_model = 33;num_heads = 3;num_classes = 15
cls = MultiHeadAttentionClassifier(seq_len, d_model, num_heads, num_classes, drate=0.15)
x=tf.random.uniform((batch_size,seq_len,d_model))
cls.compile()
cls(x)
"""

