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
	shape=[1000, 32, 32, 3])

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
		outshape=[100,34,34,32],
		out_channels=32,
		Nfilters=3)
print deconv_out

deconv_out2 = util.deconv2d_3x3(x_conv, w_deconv, b_deconv,
	outshape=[100,34,34,32])
print deconv_out2

################################################
# Convolution 3d 
################################################
x_conv3d = tf.placeholder(dtype=np.float32,
	shape=[1000,32,32,32,3])

conv3d_out, w_conv3d, b_conv3d = util.conv3d_3x3_weights(
	x_conv3d,
	Nchannels=3,
	Nfilters=32,
	std=0.01,
	padding="SAME")
print conv3d_out