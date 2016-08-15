#tensorflow utility functions
def conv2d_3x3(x, Nchannels=3, Nfilters=32, std=0.01, padding="VALID"):
    
    w = tf.random_normal(shape=(3,3, Nchannels, Nfilters), stddev=std)
    b = tf.Variable(tf.constant(std, shape=[Nfilters]))
    
    out = tf.nn.conv2d(x, w, [1,1,1,1], padding=padding)
    out = tf.nn.bias_add(out, b)
    out = tf.nn.relu(out)
    
    return (out, [w,b])

def deconv2d_3x3(x, outshape, out_channels=3, Nfilters=32, std=0.01, padding="VALID"):


    w = tf.random_normal(shape=(3,3, out_channels, Nfilters), stddev=std)
    b = tf.Variable(tf.constant(std, shape=[out_channels]))
    
    out = tf.nn.conv2d_transpose(x, w, outshape, [1,1,1,1], padding=padding)
    out = tf.nn.bias_add(out, b)
    out = tf.nn.relu(out)
    
    return (out, [w,b])

def fc(x, insize, outsize, std=0.01):
    
    w = tf.Variable(tf.random_normal([insize, outsize], stddev=std))
    b = tf.Variable(tf.constant(std, shape=[outsize]))
    
    out = tf.matmul(x,w)+b
    
    return (out,[w,b])
