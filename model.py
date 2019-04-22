"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    #for visualize big epsilon adversarial images
    self.adv_test_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.adv_test_img = tf.reshape(self.adv_test_input, [-1, 28, 28, 1])

    self.nat_test_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.nat_test_img = tf.reshape(self.nat_test_input, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
    h_pool1 = self._max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self._max_pool_2x2(h_conv2)

    # first fully connected layer
    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _avg_pool_2x2( x):
      return tf.nn.avg_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
class GSOPcModel(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    #for visualize big epsilon adversarial images
    self.adv_test_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.adv_test_img = tf.reshape(self.adv_test_input, [-1, 28, 28, 1])

    self.nat_test_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.nat_test_img = tf.reshape(self.nat_test_input, [-1, 28, 28, 1])
    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)

    #second order pooling
    W_prev_sop0 = self._weight_variable([1,1,32,16])
    b_prev_sop0 = self._bias_variable([16])
    sop_conv0 = tf.nn.relu(self._conv2d(h_conv1, W_prev_sop0) + b_prev_sop0)

    sop1 = tf.reshape(sop_conv0, [-1, 28*28, 16])
    tmp_sop1 = tf.transpose(sop1, [0, 2, 1])
    sop1 = tf.matmul(tmp_sop1, sop1)
    sop1 = tf.nn.l2_normalize(sop1, axis=2)
    sop1 = tf.reshape(sop1, [-1, 1, 16, 16])

    W_sop1 = self._weight_variable([4, 16, 16])
    b_sop1 = self._bias_variable([16*4])

    sop1 = tf.reduce_mean(sop1 * W_sop1, axis=3)
    sop1 = tf.nn.leaky_relu(tf.reshape(sop1, [-1, 4*16]) + b_sop1)
    
    W_sop1_ = self._weight_variable([4*16, 32])
    b_sop1_ = self._bias_variable([32])

    sop1 = tf.matmul(sop1, W_sop1_) + b_sop1_
    sop1 = tf.sigmoid(tf.reshape(sop1, [-1, 1, 1, 32]))
    sop_out1 = h_conv1 * sop1 

    # second convolutional layer
    h_pool1 = self._max_pool_2x2(sop_out1)

    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)

    #second order pooling
    W_prev_sop1 = self._weight_variable([1,1,64,32])
    b_prev_sop1 = self._bias_variable([32])
    sop_conv1 = tf.nn.relu(self._conv2d(h_conv2, W_prev_sop1) + b_prev_sop1)

    sop2 = tf.reshape(sop_conv1, [-1, 14*14, 32])
    tmp_sop2 = tf.transpose(sop2, [0, 2, 1])
    sop2 = tf.matmul(tmp_sop2, sop2)
    sop2 = tf.nn.l2_normalize(sop2, axis=2)
    sop2 = tf.reshape(sop2, [-1, 1, 32, 32])

    W_sop2 = self._weight_variable([4, 32, 32])
    b_sop2 = self._bias_variable([32*4])

    sop2 = tf.reduce_mean(sop2 * W_sop2, axis=3)
    sop2 = tf.nn.leaky_relu(tf.reshape(sop2, [-1, 4*32]) + b_sop2)
    
    W_sop2_ = self._weight_variable([4*32, 64])
    b_sop2_ = self._bias_variable([64])

    sop2 = tf.matmul(sop2, W_sop2_) + b_sop2_
    sop2 = tf.sigmoid(tf.reshape(sop2, [-1, 1, 1, 64]))
    sop_out2 = h_conv2 * sop2 

    # first fully connected layer
    h_pool2 = self._max_pool_2x2(sop_out2)

    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _avg_pool_2x2( x):
      return tf.nn.avg_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

class SEModel(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])

    #for visualize big epsilon adversarial images
    self.adv_test_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.adv_test_img = tf.reshape(self.adv_test_input, [-1, 28, 28, 1])

    self.nat_test_input = tf.placeholder(tf.float32, shape = [None, 784])
    self.nat_test_img = tf.reshape(self.nat_test_input, [-1, 28, 28, 1])
    # first convolutional layer
    W_conv1 = self._weight_variable([5,5,1,32])
    b_conv1 = self._bias_variable([32])

    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)

    #SE block
    self.r = 16
    squeeze1 = self._squeeze(h_conv1)
    squeeze_flat1 = tf.reshape(squeeze1, [-1, 32]) 

    W_ex11 = self._weight_variable([32, int(32/self.r)])
    b_ex11 = self._bias_variable([int(32/self.r)])
    
    ex_mid1 = tf.nn.relu(tf.matmul(squeeze_flat1, W_ex11) + b_ex11)

    W_ex12 = self._weight_variable([int(32/self.r), 32])
    b_ex12 = self._bias_variable([32])

    ex_scale1 = tf.sigmoid(tf.matmul(ex_mid1, W_ex12) + b_ex12)
    ex_scale1 = tf.reshape(ex_scale1, [-1, 1, 1, 32])
    se_out1 = ex_scale1 * h_conv1

    # second convolutional layer
    h_pool1 = self._max_pool_2x2(se_out1)

    W_conv2 = self._weight_variable([5,5,32,64])
    b_conv2 = self._bias_variable([64])

    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)

    #SE block
    squeeze2 = self._squeeze(h_conv2)
    squeeze_flat2 = tf.reshape(squeeze2, [-1, 64]) 

    W_ex21 = self._weight_variable([64, int(64/self.r)])
    b_ex21 = self._bias_variable([int(64/self.r)])
    
    ex_mid2 = tf.nn.relu(tf.matmul(squeeze_flat2, W_ex21) + b_ex21)

    W_ex22 = self._weight_variable([int(64/self.r), 64])
    b_ex22 = self._bias_variable([64])

    ex_scale2 = tf.sigmoid(tf.matmul(ex_mid2, W_ex22) + b_ex22)
    ex_scale2 = tf.reshape(ex_scale2, [-1, 1, 1, 64])
    se_out2 = ex_scale2 * h_conv2


    # first fully connected layer
    h_pool2 = self._max_pool_2x2(se_out2)

    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self._bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # output layer
    W_fc2 = self._weight_variable([1024,10])
    b_fc2 = self._bias_variable([10])

    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _squeeze( x):
      return tf.nn.avg_pool(x,
                            ksize = [1,x.shape[1],x.shape[2],1],
                            strides=[1,1,1,1],
                            padding='VALID')
  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

#class CSEModel(object):
#  def __init__(self):
#    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.y_input = tf.placeholder(tf.int64, shape = [None])
#
#    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
#
#    #for visualize big epsilon adversarial images
#    self.adv_test_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.adv_test_img = tf.reshape(self.adv_test_input, [-1, 28, 28, 1])
#
#    self.nat_test_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.nat_test_img = tf.reshape(self.nat_test_input, [-1, 28, 28, 1])
#    # first convolutional layer
#    W_conv1 = self._weight_variable([5,5,1,32])
#    b_conv1 = self._bias_variable([32])
#
#    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
#
#    #SE block
#    squeeze1 = self._squeeze(h_conv1)
#    squeeze_flat1 = tf.reshape(squeeze1, [-1, 32]) 
#
#    W_ex11 = self._weight_variable([32, 32])
#    b_ex11 = self._bias_variable([32])
#    
#    ex_scale1 = tf.sigmoid(tf.matmul(squeeze_flat1, W_ex11) + b_ex11)
#    ex_scale1 = tf.reshape(ex_scale1, [-1, 1, 1, 32])
#    se_out1 = ex_scale1 * h_conv1
#
#    # second convolutional layer
#    h_pool1 = self._max_pool_2x2(se_out1)
#
#    W_conv2 = self._weight_variable([5,5,32,64])
#    b_conv2 = self._bias_variable([64])
#
#    h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
#
#    #SE block
#    squeeze2 = self._squeeze(h_conv2)
#    squeeze_flat2 = tf.reshape(squeeze2, [-1, 64]) 
#
#    W_ex21 = self._weight_variable([64,64])
#    b_ex21 = self._bias_variable([64])
#    
#    ex_scale2 = tf.sigmoid(tf.matmul(squeeze_flat2, W_ex21) + b_ex21)
#    ex_scale2 = tf.reshape(ex_scale2, [-1, 1, 1, 64])
#    se_out2 = ex_scale2 * h_conv2
#
#
#    # first fully connected layer
#    h_pool2 = self._max_pool_2x2(se_out2)
#
#    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
#    b_fc1 = self._bias_variable([1024])
#
#    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#    # output layer
#    W_fc2 = self._weight_variable([1024,10])
#    b_fc2 = self._bias_variable([10])
#
#    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
#
#    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#        labels=self.y_input, logits=self.pre_softmax)
#
#    self.xent = tf.reduce_sum(y_xent)
#
#    self.y_pred = tf.argmax(self.pre_softmax, 1)
#
#    correct_prediction = tf.equal(self.y_pred, self.y_input)
#
#    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
#    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#  @staticmethod
#  def _weight_variable(shape):
#      initial = tf.truncated_normal(shape, stddev=0.1)
#      return tf.Variable(initial)
#
#  @staticmethod
#  def _bias_variable(shape):
#      initial = tf.constant(0.1, shape = shape)
#      return tf.Variable(initial)
#
#  @staticmethod
#  def _conv2d(x, W):
#      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
#  @staticmethod
#  def _avg_pool_2x2( x):
#      return tf.nn.avg_pool(x,
#                            ksize = [1,2,2,1],
#                            strides=[1,2,2,1],
#                            padding='SAME')
#  @staticmethod
#  def _squeeze( x):
#      return tf.nn.avg_pool(x,
#                            ksize = [1,x.shape[1],x.shape[2],1],
#                            strides=[1,1,1,1],
#                            padding='VALID')
#  @staticmethod
#  def _max_pool_2x2( x):
#      return tf.nn.max_pool(x,
#                            ksize = [1,2,2,1],
#                            strides=[1,2,2,1],
#                            padding='SAME')

#class GSOPwhModel(object):
#  def __init__(self):
#    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.y_input = tf.placeholder(tf.int64, shape = [None])
#
#    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
#
#    #for visualize big epsilon adversarial images
#    self.adv_test_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.adv_test_img = tf.reshape(self.adv_test_input, [-1, 28, 28, 1])
#
#    self.nat_test_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.nat_test_img = tf.reshape(self.nat_test_input, [-1, 28, 28, 1])
#    # first convolutional layer
#    W_conv1 = self._weight_variable([5,5,1,32])
#    b_conv1 = self._bias_variable([32])
#
#    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
#    h_pool1 = self._avg_pool_2x2(h_conv1)
#
#    #second order pooling
#    W_prev_sop0 = self._weight_variable([1,1,32,16])
#    b_prev_sop0 = self._bias_variable([16])
#    sop_conv0 = tf.nn.relu(self._conv2d(h_pool1, W_prev_sop0) + b_prev_sop0)
#
#    sop1 = tf.reshape(sop_conv0, [-1, 14*14, 16])
#    tmp_sop1 = tf.transpose(sop1, [0, 2, 1])
#    sop1 = tf.matmul(sop1, tmp_sop1)
#    sop1 = tf.nn.l2_normalize(sop1, axis=2)
#    sop1 = tf.reshape(sop1, [-1, 1, 14*14, 14*14])
#
#    W_sop1 = self._weight_variable([4, 14*14, 14*14])
#    b_sop1 = self._bias_variable([14*14*4])
#
#    sop1 = tf.reduce_mean(sop1 * W_sop1, axis=3)
#    sop1 = tf.nn.leaky_relu(tf.reshape(sop1, [-1, 4*14*14]) + b_sop1)
#    
#    W_sop1_ = self._weight_variable([4*14*14, 14*14])
#    b_sop1_ = self._bias_variable([14*14])
#
#    sop1 = tf.matmul(sop1, W_sop1_) + b_sop1_
#    sop1 = tf.sigmoid(tf.reshape(sop1, [-1, 14, 14, 1]))
#    sop_out1 = h_pool1 * sop1 
#
#    # second convolutional layer
#    W_conv2 = self._weight_variable([5,5,32,64])
#    b_conv2 = self._bias_variable([64])
#
#    h_conv2 = tf.nn.relu(self._conv2d(sop_out1, W_conv2) + b_conv2)
#    h_pool2 = self._avg_pool_2x2(h_conv2)
#
#    #second order pooling
#    W_prev_sop1 = self._weight_variable([1,1,64,32])
#    b_prev_sop1 = self._bias_variable([32])
#    sop_conv1 = tf.nn.relu(self._conv2d(h_pool2, W_prev_sop1) + b_prev_sop1)
#
#    sop2 = tf.reshape(sop_conv1, [-1, 7*7, 32])
#    tmp_sop2 = tf.transpose(sop2, [0, 2, 1])
#    sop2 = tf.matmul(sop2, tmp_sop2)
#    sop2 = tf.nn.l2_normalize(sop2, axis=2)
#    sop2 = tf.reshape(sop2, [-1, 1, 7*7, 7*7])
#
#    W_sop2 = self._weight_variable([4, 7*7, 7*7])
#    b_sop2 = self._bias_variable([7*7*4])
#
#    sop2 = tf.reduce_mean(sop2 * W_sop2, axis=3)
#    sop2 = tf.nn.leaky_relu(tf.reshape(sop2, [-1, 4*7*7]) + b_sop2)
#    
#    W_sop2_ = self._weight_variable([4*7*7, 7*7])
#    b_sop2_ = self._bias_variable([7*7])
#
#    sop2 = tf.matmul(sop2, W_sop2_) + b_sop2_
#    sop2 = tf.sigmoid(tf.reshape(sop2, [-1, 7, 7, 1]))
#    sop_out2 = h_pool2 * sop2 
#
#    # first fully connected layer
#    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
#    b_fc1 = self._bias_variable([1024])
#
#    h_pool2_flat = tf.reshape(sop_out2, [-1, 7 * 7 * 64])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#    # output layer
#    W_fc2 = self._weight_variable([1024,10])
#    b_fc2 = self._bias_variable([10])
#
#    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
#
#    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#        labels=self.y_input, logits=self.pre_softmax)
#
#    self.xent = tf.reduce_sum(y_xent)
#
#    self.y_pred = tf.argmax(self.pre_softmax, 1)
#
#    correct_prediction = tf.equal(self.y_pred, self.y_input)
#
#    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
#    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#  @staticmethod
#  def _weight_variable(shape):
#      initial = tf.truncated_normal(shape, stddev=0.1)
#      return tf.Variable(initial)
#
#  @staticmethod
#  def _bias_variable(shape):
#      initial = tf.constant(0.1, shape = shape)
#      return tf.Variable(initial)
#
#  @staticmethod
#  def _conv2d(x, W):
#      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
#  @staticmethod
#  def _avg_pool_2x2( x):
#      return tf.nn.avg_pool(x,
#                            ksize = [1,2,2,1],
#                            strides=[1,2,2,1],
#                            padding='SAME')
#  @staticmethod
#  def _max_pool_2x2( x):
#      return tf.nn.max_pool(x,
#                            ksize = [1,2,2,1],
#                            strides=[1,2,2,1],
#                            padding='SAME')
#
#class CompactGSOPwhModel(object):
#  def __init__(self):
#    self.x_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.y_input = tf.placeholder(tf.int64, shape = [None])
#
#    self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
#
#    #for visualize big epsilon adversarial images
#    self.adv_test_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.adv_test_img = tf.reshape(self.adv_test_input, [-1, 28, 28, 1])
#
#    self.nat_test_input = tf.placeholder(tf.float32, shape = [None, 784])
#    self.nat_test_img = tf.reshape(self.nat_test_input, [-1, 28, 28, 1])
#    # first convolutional layer
#    W_conv1 = self._weight_variable([5,5,1,32])
#    b_conv1 = self._bias_variable([32])
#
#    h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)
#    h_pool1 = self._avg_pool_2x2(h_conv1)
#
#    #second order pooling
#    W_prev_sop0 = self._weight_variable([1,1,32,16])
#    b_prev_sop0 = self._bias_variable([16])
#    sop_conv0 = tf.nn.relu(self._conv2d(h_pool1, W_prev_sop0) + b_prev_sop0)
#
#    sop1 = tf.reshape(sop_conv0, [-1, 14*14, 16])
#    tmp_sop1 = tf.transpose(sop1, [0, 2, 1])
#    sop1 = tf.matmul(sop1, tmp_sop1)
#    sop1 = tf.nn.l2_normalize(sop1, axis=2)
#    sop1 = tf.reshape(sop1, [-1, 14*14, 14*14])
#
#    W_sop1 = self._weight_variable([14*14, 14*14])
#    b_sop1 = self._bias_variable([14*14])
#
#    sop1 = tf.reduce_mean(sop1 * W_sop1, axis=2) + b_sop1
#    sop1 = tf.sigmoid(tf.reshape(sop1, [-1, 14, 14, 1]))
#    sop_out1 = h_pool1 * sop1 
#
#    # second convolutional layer
#    W_conv2 = self._weight_variable([5,5,32,64])
#    b_conv2 = self._bias_variable([64])
#
#    h_conv2 = tf.nn.relu(self._conv2d(sop_out1, W_conv2) + b_conv2)
#    h_pool2 = self._avg_pool_2x2(h_conv2)
#
#    #second order pooling
#    W_prev_sop1 = self._weight_variable([1,1,64,32])
#    b_prev_sop1 = self._bias_variable([32])
#    sop_conv1 = tf.nn.relu(self._conv2d(h_pool2, W_prev_sop1) + b_prev_sop1)
#
#    sop2 = tf.reshape(sop_conv1, [-1, 7*7, 32])
#    tmp_sop2 = tf.transpose(sop2, [0, 2, 1])
#    sop2 = tf.matmul(sop2, tmp_sop2)
#    sop2 = tf.nn.l2_normalize(sop2, axis=2)
#    sop2 = tf.reshape(sop2, [-1, 7*7, 7*7])
#
#    W_sop2 = self._weight_variable([7*7, 7*7])
#    b_sop2 = self._bias_variable([7*7])
#
#    sop2 = tf.reduce_mean(sop2 * W_sop2, axis=2) + b_sop2
#    sop2 = tf.sigmoid(tf.reshape(sop2, [-1, 7, 7, 1]))
#    sop_out2 = h_pool2 * sop2 
#
#    # first fully connected layer
#    W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
#    b_fc1 = self._bias_variable([1024])
#
#    h_pool2_flat = tf.reshape(sop_out2, [-1, 7 * 7 * 64])
#    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#    # output layer
#    W_fc2 = self._weight_variable([1024,10])
#    b_fc2 = self._bias_variable([10])
#
#    self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
#
#    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
#        labels=self.y_input, logits=self.pre_softmax)
#
#    self.xent = tf.reduce_sum(y_xent)
#
#    self.y_pred = tf.argmax(self.pre_softmax, 1)
#
#    correct_prediction = tf.equal(self.y_pred, self.y_input)
#
#    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
#    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#  @staticmethod
#  def _weight_variable(shape):
#      initial = tf.truncated_normal(shape, stddev=0.1)
#      return tf.Variable(initial)
#
#  @staticmethod
#  def _bias_variable(shape):
#      initial = tf.constant(0.1, shape = shape)
#      return tf.Variable(initial)
#
#  @staticmethod
#  def _conv2d(x, W):
#      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
#  @staticmethod
#  def _avg_pool_2x2( x):
#      return tf.nn.avg_pool(x,
#                            ksize = [1,2,2,1],
#                            strides=[1,2,2,1],
#                            padding='SAME')
#  @staticmethod
#  def _max_pool_2x2( x):
#      return tf.nn.max_pool(x,
#                            ksize = [1,2,2,1],
#                            strides=[1,2,2,1],
#                            padding='SAME')
