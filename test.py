import util
import tensorflow as tf
import numpy as np

###############################################
# Weight variable test
###############################################
w = util.weight_variable([2,3,4,5], 0.01)
print w

b = util.bias_variable([2], 0.01)
print b

###############################################
# Fully connected layer test
###############################################
x_fc = tf.placeholder(dtype=np.float32,
	shape = [100,10])

fc_out, w_fc, b_fc = util.fc_weights(x_fc, 10,1000)
print fc_out

fc_out_2 = util.fc(x_fc,w_fc,b_fc)
print fc_out_2

###############################################
# Convolution 2d layers test
###############################################
x_conv = tf.placeholder(dtype=np.float32,
	shape=[10, 32, 32, 3])

conv_out, w_conv, b_conv = util.conv2d_3x3_weights(
	x_conv,
	Nchannels=3,
	Nfilters=32,
	std=0.01,
	padding="SAME")
print conv_out

conv_out2 = util.conv2d_3x3(x_conv, w_conv, b_conv,
	padding="SAME")
print conv_out2

#################################################
# Deconvolution test
#################################################
deconv_out, w_deconv, b_deconv = \
	util.deconv2d_3x3_weights(x_conv,
		outshape=[10,34,34,32],
		out_channels=32,
		Nfilters=3)
print deconv_out

deconv_out2 = util.deconv2d_3x3(x_conv, w_deconv, b_deconv,
	outshape=[10,34,34,32])
print deconv_out2

################################################
# Convolution 3d 
################################################
x_conv3d = tf.placeholder(dtype=np.float32,
	shape=[1,32,32,32,3])

conv3d_out, w_conv3d, b_conv3d = util.conv3d_3x3_weights(
	x_conv3d,
	Nchannels=3,
	Nfilters=32,
	std=0.01,
	padding="SAME")
print conv3d_out

################################################
# Batch normalization
################################################
mode = tf.placeholder(bool)
x_fc_norm, gamma_fc, beta_fc, mean_fc, var_fc = util.batch_norm(x_fc, mode, [10], mom=0.5)
print x_fc_norm
print gamma_fc
print beta_fc

x_conv_norm, gamma_conv, beta_conv, mean_conv, var_conv = util.batch_norm(x_conv, mode, [10, 32, 32, 3])
print x_conv_norm
print gamma_conv
print beta_conv

init = tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#test batch norm train
for i in range(0,5000):
	x= np.random.randn(100,10)+10.0
	sess.run(x_fc_norm, feed_dict={x_fc:x,mode:True})

print sess.run(mean_fc, feed_dict={x_fc:x,mode:True})
print sess.run(var_fc, feed_dict={x_fc:x,mode:True})