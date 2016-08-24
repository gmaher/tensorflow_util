import tensorflow as tf
#tensorflow utility functions
def weight_variable(shape, std):
    return tf.Variable(tf.random_normal(shape=shape, stddev=std))

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

   return out

def conv2d_3x3_weights(x, Nchannels=3, Nfilters=32, std=0.01, padding="VALID"):
    w = weight_variable([3,3,Nchannels, Nfilters], std)
    b = bias_variable([Nfilters], std)
    
    out = conv2d_3x3(x,w,b,padding=padding)
    
    return (out, w, b)

def deconv2d_3x3(x,w,b, outshape, padding="VALID"):
    out = tf.nn.conv2d_transpose(x, w, outshape, [1,1,1,1], padding=padding)
    out = tf.nn.bias_add(out, b)
    
    return out

def deconv2d_3x3_weights(x, outshape, out_channels=3, Nfilters=32, std=0.01, padding="VALID"):
    w = weight_variable([3, 3, out_channels, Nfilters], std)
    b = bias_variable([out_channels], std)
    
    out = deconv2d_3x3(x,w,b,outshape,padding=padding)
    
    return (out, w, b)

def conv3d_3x3(x,w,b, padding="VALID"):
    out = tf.nn.conv3d(x,w, [1,1,1,1,1], padding=padding)
    out = tf.nn.bias_add(out, b)

    return out

def conv3d_3x3_weights(x, Nchannels=3, Nfilters=32, std=0.01, padding="VALID"):
    w = weight_variable([3,3,3,Nchannels,Nfilters], std)
    b = bias_variable([Nfilters], std)

    out = conv3d_3x3(x,w,b, padding=padding)

    return (out,w,b)

def batch_norm_train(x, mu, sig, gamma, beta):
    xnorm = tf.div((x-mu), sig)

    return tf.mul(xnorm,gamma)+beta

def batch_norm(x, mode, inshape, mom=0.9, axes=[0]):
    '''
    returns a conditional operator that either performs batch
    normalization with exponentially weighted running statistics
    for mean and variance during training, or during test simply 
    applies batch normalization

    inputs:
    - x: Input tensor
    - mode: A boolean placeholder, if True indicates training mode
    - mom: exponentially weighted averaging coefficient
    - inshape: input shape of x without the batch index
    '''
    eps = 1e-6

    gamma = tf.Variable(tf.ones(inshape))
    beta = tf.Variable(tf.zeros(inshape))

    batch_mean, batch_var = tf.nn.moments(x,axes)

    ema = tf.train.ExponentialMovingAverage(mom)

    def collect_average():
        with tf.control_dependencies([
            ema.apply([batch_mean, batch_var])
            ]):
            return (tf.identity(batch_mean), tf.identity(batch_var))

    def get_run_stats():
        return (ema.average(batch_mean), ema.average(batch_var))

    mu, var = tf.cond(mode, 
        collect_average,
        get_run_stats)

    result = tf.nn.batch_normalization(x, mu, var, beta, gamma, 1e-4)

    return (result, gamma, beta)

