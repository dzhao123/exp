import numpy as np
import tensorflow as tf


class Generator:
    def __init__(self, feature_size, hidden_size, keep_prob=1.0):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        
        with tf.variable_scope('generator'):
            # input placeholders
            self.reward = tf.placeholder(tf.float32, [None], name='reward')
            self.pred_data = tf.placeholder(tf.float32, [None, self.feature_size], name='pred_data')
            self.sample_index = tf.placeholder(tf.int32, [None], name='sample_index')
            
            ########## score of RankNet ##########

            # trainable variables
            self.weight_1 = tf.Variable(tf.truncated_normal([self.feature_size, self.hidden_size], mean=0.0, stddev=0.1), name='weight_1')
            self.bias_1 = tf.Variable(tf.zeros([self.hidden_size]), name='bias_1')
            self.weight_2 = tf.Variable(tf.truncated_normal([self.hidden_size, 1], mean=0.0, stddev=0.1), name='weight_2')
            self.bias_2 = tf.Variable(tf.zeros([1]), name='bias_2')

            # layer 1 (hidden layer)
            self.layer_1 = tf.nn.tanh(tf.nn.xw_plus_b(self.pred_data, self.weight_1, self.bias_1))
            
            # dropout
            self.dropout_1 = tf.nn.dropout(self.layer_1, self.keep_prob)

            # layer 2 (output layer)
            self.layer_2 = tf.nn.xw_plus_b(self.dropout_1, self.weight_2, self.bias_2)
            
            #################################
            
            # probability distribution
            self.opt_prob =tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.layer_2, [1, -1])), [-1]), self.sample_index)
            
            # loss for optimization
            self.opt_loss = -tf.reduce_mean(tf.log(self.opt_prob) * self.reward) # minus signe is needed for maximum.
            
            # score for prediction
            self.pred_score = tf.nn.xw_plus_b(self.layer_1, self.weight_2, self.bias_2)
            self.pred_score = tf.reshape(self.pred_score, [-1])
