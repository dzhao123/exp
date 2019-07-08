import tensorflow as tf
import numpy as np

class Optimizer:
    def __init__(self, g, d, learning_rate):
        # get the trainable_variables, split into generator and discriminator parts.
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if var.name.startswith('generator')]
        self.d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

        self.g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g.opt_loss, var_list=self.g_vars)        
        self.d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d.opt_loss, var_list=self.d_vars)
