from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                     'batch_size, num_classes, weight_decay, momentum')


class ResNet(object):
    def __init__(self, hp, images, labels, global_step):
        self._hp = hp  # Hyperparameters
        self._images = images  # Input image
        self._labels = labels
        self._global_step = global_step
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self._flops = 0
        self._weights = 0

    def build_model(self):
        print('Building model')

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(self._images, 7, 64, 2)
            x = self._bn(x)
            x = self._relu(x)

        print('Building unit: pool2')
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        # conv2
        for i in range(3):
            unit_name = "res2%s" % chr(97 + i)
            print('Building unit: %s' % unit_name)
            x = self._residual_block(x, [1, 3, 1], [64, 64, 256], False, unit_name)

        # conv3
        for i in range(4):
            unit_name = "res3%s" % chr(97 + i)
            print('Building unit: %s' % unit_name)
            stride_down = i == 0
            x = self._residual_block(x, [1, 3, 1], [128, 128, 512], stride_down, unit_name)

        # conv4
        for i in range(6):
            unit_name = "res4%s" % chr(97 + i)
            print('Building unit: %s' % unit_name)
            stride_down = i == 0
            x = self._residual_block(x, [1, 3, 1], [256, 256, 1024], stride_down, unit_name)

        # conv5
        for i in range(3):
            unit_name = "res5%s" % chr(97 + i)
            print('Building unit: %s' % unit_name)
            stride_down = i == 0
            x = self._residual_block(x, [1, 3, 1], [512, 512, 2048], stride_down, unit_name)

        # pool5
        print('Building unit: pool5')
        x = tf.nn.avg_pool(x, [1, 7, 7, 1], [1, 1, 1, 1], "VALID")

        # Logit
        with tf.variable_scope('fc1000') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, np.prod(x_shape[1:])])
            x = self._fc(x, self._hp.num_classes)

        self._logits = x

        # Probs & preds & acc
        self.probs = tf.nn.softmax(x, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        with tf.variable_scope("acc"):
            ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
            zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
            correct = tf.select(tf.equal(self.preds, self._labels), ones, zeros)
            self.acc = tf.reduce_mean(correct, name='acc')
        tf.scalar_summary('accuracy', self.acc)

        # Loss & acc
        with tf.variable_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(x, self._labels)
            self.loss = tf.reduce_mean(loss, name='cross_entropy')
        tf.scalar_summary('cross_entropy', self.loss)


    def _residual_block(self, x, filters, channels, stride_down=False, name="unit"):
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            in_channel = x.get_shape().as_list()[-1]
            # Shortcut connection
            if not stride_down and in_channel == channels[-1]:
                shortcut = x
            else:
                shortcut_stride = 2 if stride_down else 1
                with tf.variable_scope("branch1"):
                    shortcut = self._conv(x, 1, channels[-1], shortcut_stride)
                    shortcut = self._bn(shortcut)

            # Residual
            for i, (f, ch) in enumerate(zip(filters, channels)):
                with tf.variable_scope("branch2%s" % (chr(97 + i))):
                    x = self._conv(x, f, ch, 2 if (stride_down and i == 0) else 1)
                    x = self._bn(x)
                    if i < len(filters) - 1:
                        x = self._relu(x)

            # Merge
            x = x + shortcut
            x = self._relu(x)

        return x

    def build_train_op(self):
        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
            # for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
                # tf.histogram_summary(var.op.name, var)
            l2_loss = tf.mul(self._hp.weight_decay, tf.add_n(costs))
        self._total_loss = self.loss + l2_loss

        # Learning rate
        # self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        # self._hp.decay_step, self._hp.lr_decay, staircase=True)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        tf.scalar_summary('learing_rate', self.lr)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        # print '\n'.join([t.name for t in tf.trainable_variables()])
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops+[apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, name)
        self._flops += (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        self._weights += in_channel * out_channel * filter_size * filter_size
        return x

    def _fc(self, x, out_dim, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, name)
        self._flops += (in_dim + 1) * out_dim
        self._weights += (in_dim + 1) * out_dim
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        self._flops += 8 * self._get_data_size(x)
        self._weights += 4 * x.get_shape().as_list()[-1]
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        self._flops += self._get_data_size(x)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])
