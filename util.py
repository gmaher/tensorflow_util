import tensorflow as tf
#tensorflow utility functions
def weight_variable(shape, std):
    return tf.random_normal(shape=shape, stddev=std)

def bias_variable(shape, init):
    return tf.Variable(tf.constant(init, shape=shape))

def fc(x,w,b):
    return tf.matmul(x,w)+b   

def fc_weights(x, insize, outsize, std=0.01):
    w = weight_variable([insize, outsize], std)
    b = bias_variable([outsize], std)
    
    out = fc(x,w,b)
    
    return (out, w, b)

def conv2d_3x3(x,w,b, padding="VALID"):
   out = tf.nn.conv2d(x, w, [1,1,1,1], padding=padding)
   out = tf.nn.bias_add(out, b)
   out = tf.nn.relu(out)
   
   return out

def conv2d_3x3_weights(x, Nchannels=3, Nfilters=32, std=0.01, padding="VALID"):
    w = weight_variable([3,3,Nchannels, Nfilters], std)
    b = bias_variable([Nfilters], std)
    
    out = conv2d_3x3(x,w,b,padding=padding)
    
    return (out, w, b)

def deconv2d_3x3(x,w,b, outshape, padding="VALID"):
    out = tf.nn.conv2d_transpose(x, w, outshape, [1,1,1,1], padding=padding)
    out = tf.nn.bias_add(out, b)
    out = tf.nn.relu(out)
    
    return out

def deconv2d_3x3_weights(x, outshape, out_channels=3, Nfilters=32, std=0.01, padding="VALID"):
    w = weight_variable([3, 3, out_channels, Nfilters], std)
    b = bias_variable([out_channels], std)
    
    out = deconv2d_3x3(x,w,b,outshape,padding=padding)
    
    return (out, w, b)

def conv3d_3x3(x,w,b, padding="VALID"):
    out = tf.nn.conv3d(x,w, [1,1,1,1,1], padding=padding)
    out = tf.nn.bias_add(out, b)
    out = tf.nn.relu(out)

    return out

def conv3d_3x3_weights(x, Nchannels=3, Nfilters=32, std=0.01, padding="VALID"):
    w = weight_variable([3,3,3,Nchannels,Nfilters], std)
    b = bias_variable([Nfilters], std)

    out = conv3d_3x3(x,w,b, padding=padding)

    return (out,w,b)
