import tensorflow as tf
import numpy as np

class Discriminator:
    def __init__(self, feature_size, hidden_size, keep_prob=1.0):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        with tf.variable_scope('discriminator'):
            # input placeholders
            self.pos_data = tf.placeholder(tf.float32, [None, self.feature_size], name='pos_data')
            self.neg_data = tf.placeholder(tf.float32, [None, self.feature_size], name='neg_data')
            self.pred_data = tf.placeholder(tf.float32, [None, self.feature_size], name='pred_data')

            ########## score of RankNet ##########
            
            ## trainable variables
            self.weight_1 = tf.Variable(tf.truncated_normal([self.feature_size, self.hidden_size], mean=0.0, stddev=0.1), name='weight_1')
            self.bias_1 = tf.Variable(tf.zeros([self.hidden_size]), name='bias_1')
            self.weight_2 = tf.Variable(tf.truncated_normal([self.hidden_size, 1], mean=0.0, stddev=0.1), name='weight_2')
            self.bias_2 = tf.Variable(tf.zeros([1]), name='bias_2')
            
            # layer 1 (hidden layer)
            self.pos_layer_1 = tf.nn.tanh(tf.nn.xw_plus_b(self.pos_data, self.weight_1, self.bias_1))
            self.neg_layer_1 = tf.nn.tanh(tf.nn.xw_plus_b(self.neg_data, self.weight_1, self.bias_1))
            
            # dropout
            self.pos_dropout_1 = tf.nn.dropout(self.pos_layer_1, self.keep_prob)
            self.neg_dropout_1 = tf.nn.dropout(self.neg_layer_1, self.keep_prob)
            
            # layer 2 (output layer)
            self.pos_layer_2 = tf.nn.xw_plus_b(self.pos_dropout_1, self.weight_2, self.bias_2)
            self.neg_layer_2 = tf.nn.xw_plus_b(self.neg_dropout_1, self.weight_2, self.bias_2)
            
            #################################
            
            # loss for optimization
            self.opt_loss = -tf.reduce_mean(tf.log(tf.sigmoid(self.pos_layer_2 - self.neg_layer_2))) # minus signe is needed for miximum
            
            # reward for generator
            self.reward = tf.reshape(tf.log(1 + tf.exp(self.neg_layer_2 - self.pos_layer_2)), [-1])
            
            # score for prediction
            self.pred_score = tf.nn.xw_plus_b(tf.nn.tanh(tf.nn.xw_plus_b(self.pred_data, self.weight_1, self.bias_1)), self.weight_2, self.bias_2)
            self.pred_score = tf.reshape(self.pred_score , [-1])
