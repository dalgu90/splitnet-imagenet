from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                     'num_gpu, batch_size, split, num_classes, weight_decay, '
                     'momentum, no_logit_map')


class ResNet(object):
    def __init__(self, hp, images, labels, global_step):
        self._hp = hp  # Hyperparameters
        self._images = tf.split(0, self._hp.num_gpu, images)  # Input image
        self._labels = tf.split(0, self._hp.num_gpu, labels)
        self._global_step = global_step
        self.lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self._logits = []
        self._probs = []
        self._preds = []
        self._accs = []
        self._losses = []
        self._counted_scope = []
        self._device_name_scopes = []
        self._flops = 0
        self._weights = 0

    def set_clustering(self, clustering):
        # clustering: 4-depth list(list of list of list of list)
        # which is represented as 3-depth tree
        print('Parsing clustering')
        cluster_size = [[[len(sublist3) for sublist3 in sublist2] for sublist2 in sublist1] for sublist1 in clustering]
        self._split3 = [item for sublist1 in cluster_size for sublist2 in sublist1 for item in sublist2]
        self._split2 = [sum(sublist2) for sublist1 in cluster_size for sublist2 in sublist1]
        self._split1 = [sum([sum(sublist2) for sublist2 in sublist1]) for sublist1 in cluster_size]
        self._logit_map = [item for sublist1 in clustering for sublist2 in sublist1 for sublist3 in sublist2 for item in sublist3]
        print('\t1st level: %d splits %s' % (len(self._split1), self._split1))
        print('\t2nd level: %d splits %s' % (len(self._split2), self._split2))
        print('\t3rd level: %d splits %s' % (len(self._split3), self._split3))
        # print self._logit_map

    def _split_channels(self, N, groups):
        group_total = sum(groups)
        float_outputs = [float(N)*t/group_total for t in groups]
        for i in xrange(1, len(float_outputs), 1):
            float_outputs[i] = float_outputs[i-1] + float_outputs[i]
        outputs = map(int, map(round, float_outputs))
        for i in xrange(len(outputs)-1, 0, -1):
            outputs[i] = outputs[i] - outputs[i-1]
        return outputs

    def build_model(self):
        # Build models
        for i in range(self._hp.num_gpu):
            with tf.device('/GPU:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    self._device_name_scopes.append(scope)  # Keep the name scopes(used when building trainig ops)
                    print('Building model for %s' % scope)
                    self._build_model(self._images[i], self._labels[i])
                    tf.get_variable_scope().reuse_variables()

        # Merge losses and accs and preds
        self.loss = tf.identity(tf.add_n(self._losses) / self._hp.num_gpu, name="cross_entropy")
        tf.scalar_summary("cross_entropy", self.loss)
        self.acc = tf.identity(tf.add_n(self._accs) / self._hp.num_gpu, name="acc")
        tf.scalar_summary("accuracy", self.acc)
        self.preds = tf.concat(0, self._preds)

    def _build_model(self, image, label):
        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(image, 7, 64, 2)
            x = self._bn(x)
            x = self._relu(x)

        print('\tBuilding unit: pool2')
        x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], "SAME")

        # conv2
        for i in range(3):
            unit_name = "res2%s" % chr(97 + i)
            # print('Building unit: %s' % unit_name)
            x = self._residual_block(x, [1, 3, 1], [64, 64, 256], False, unit_name)

        # conv3
        for i in range(4):
            unit_name = "res3%s" % chr(97 + i)
            # print('Building unit: %s' % unit_name)
            stride_down = i == 0
            x = self._residual_block(x, [1, 3, 1], [128, 128, 512], stride_down, unit_name)

        # conv4
        for i in range(6):
            unit_name = "res4%s" % chr(97 + i)
            # print('Building unit: %s' % unit_name)
            stride_down = i == 0
            x = self._residual_block(x, [1, 3, 1], [256, 256, 1024], stride_down, unit_name)

        if not self._hp.split:
            # conv5
            for i in range(3):
                unit_name = "res5%s" % chr(97 + i)
                # print('Building unit: %s' % unit_name)
                stride_down = i == 0
                x = self._residual_block(x, [1, 3, 1], [512, 512, 2048], stride_down, unit_name)

            # pool5
            print('\tBuilding unit: pool5')
            x = tf.nn.avg_pool(x, [1, 7, 7, 1], [1, 1, 1, 1], "VALID")

            # Logit
            with tf.variable_scope('fc1000') as scope:
                print('\tBuilding unit: %s' % scope.name)
                x_shape = x.get_shape().as_list()
                x = tf.reshape(x, [-1, np.prod(x_shape[1:])])
                x = self._fc(x, self._hp.num_classes)
        else:  # self._hp.split == True
            # x: [batch_size, 14, 14, 1024]
            x = self._residual_block(x, [1, 3, 1], [512, 512, 2048], True, "res5a")
            # x: [batch_size, 7, 7, 2048]
            in_split = self._split_channels(2048, self._split1)
            out_channels_split = zip(*[self._split_channels(f, self._split1) for f in [512, 512, 2048]])
            x = self._residual_block_split(x, in_split, [1, 3, 1], out_channels_split, False, "res5b")
            # x: [batch_size, 7, 7, 2048]
            in_split = self._split_channels(2048, self._split2)
            out_channels_split = zip(*[self._split_channels(f, self._split2) for f in [512, 512, 2048]])
            x = self._residual_block_split(x, in_split, [1, 3, 1], out_channels_split, False, "res5c")
            # x: [batch_size, 7, 7, 2048]

            # pool5
            print('\tBuilding unit: pool5')
            x = tf.nn.avg_pool(x, [1, 7, 7, 1], [1, 1, 1, 1], "VALID")
            # x: [batch_size, 1, 1, 2048]

            # Logit
            with tf.variable_scope('fc1000') as scope:
                print('\tBuilding unit: %s' % scope.name)
                x_shape = x.get_shape().as_list()
                x = tf.reshape(x, [-1, np.prod(x_shape[1:])])
                in_split = self._split_channels(2048, self._split3)
                x = self._fc_split(x, in_split, self._split3)
                if not self._hp.no_logit_map:
                    x = tf.transpose(tf.gather(tf.transpose(x), self._logit_map))

        # x: [batch_size, 1000]
        logit = x

        # Probs & preds & acc
        prob = tf.nn.softmax(x, name='probs')
        pred = tf.to_int32(tf.argmax(logit, 1, name='preds'))
        with tf.variable_scope("acc"):
            ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
            zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
            correct = tf.select(tf.equal(pred, label), ones, zeros)
            acc = tf.reduce_mean(correct, name='acc')

        # Loss & acc
        with tf.variable_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(x, label)
            loss = tf.reduce_mean(loss, name='cross_entropy')

        self._logits.append(logit)
        self._probs.append(prob)
        self._preds.append(pred)
        self._accs.append(acc)
        self._losses.append(loss)

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

    def _residual_block_split(self, x, in_split, filters, out_channels_split, stride_down=False, name="unit"):
        b, h, w, num_channel = x.get_shape().as_list()
        assert num_channel == sum(in_split)
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s with %d splits' % (scope.name, len(in_split)))
            outs = []
            offset_in = 0
            for i, (n_in, channels) in enumerate(zip(in_split, out_channels_split)):
                sliced = tf.slice(x, [0, 0, 0, offset_in], [b, h, w, n_in])
                sliced_residual = self._residual_block(sliced, filters, channels, stride_down=stride_down, name=('split_%d' % (i+1)))
                outs.append(sliced_residual)
                offset_in += n_in
            concat = tf.concat(3, outs)
        return concat

    def _fc_split(self, x, in_split, out_split, name='unit'):
        b, num_in = x.get_shape().as_list()
        assert num_in == sum(in_split)
        with tf.variable_scope(name) as scope:
            print('\tBuilding fc layer: %s with %d splits' % (scope.name, len(in_split)))
            outs = []
            offset_in = 0
            for i, (n_in, n_out) in enumerate(zip(in_split, out_split)):
                sliced = tf.slice(x, [0, offset_in], [b, n_in])
                sliced_fc = self._fc(sliced, n_out, name="split_%d" % (i+1))
                outs.append(sliced_fc)
                offset_in += n_in
            concat = tf.concat(1, outs)
        return concat

    def build_train_op(self):
        # Learning rate
        # self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        # self._hp.decay_step, self._hp.lr_decay, staircase=True)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        tf.scalar_summary('learing_rate', self.lr)

        # Compute gradients for each GPU
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        tower_grads = []
        for i, loss in enumerate(self._losses):
            with tf.device('/GPU:%d' % i):
                with tf.name_scope(self._device_name_scopes[i]) as scope:
                    print('Compute gradients for %s' % scope)
                    # Add l2 loss
                    with tf.variable_scope('l2_loss'):
                        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                        l2_loss = tf.mul(self._hp.weight_decay, tf.add_n(costs))
                    total_loss = loss + l2_loss

                    # Gradient descent step
                    tr_vars = tf.trainable_variables()
                    grads_and_vars = opt.compute_gradients(total_loss, tr_vars)

                    # If splitted network, Slow down the base layers' learning rate
                    if self._hp.split:
                        num_basenet_var = tr_vars.index('res4f/branch2c/bn/gamma:0')+1
                        # num_basenet_var = tr_vars.index('res5a/branch2c/bn/gamma:0')+1
                        for i in range(num_basenet_var):
                            g, v = grads_and_vars[i]
                            print('\tScale down learning rate of %s' % v.name)
                            g = 0.1 * g
                            grads_and_vars[i] = (g, v)

                    tower_grads.append(grads_and_vars)

        # Average grads from GPUs
        with tf.name_scope('Average_grad'):
            average_grads_and_vars = self._average_gradients(tower_grads)
            apply_grad_op = opt.apply_gradients(average_grads_and_vars)

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
        f = (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, name)
        f = (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        f = 8 * self._get_data_size(x)
        w = 4 * x.get_shape().as_list()[-1]
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        f = self._get_data_size(x)
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
              # Add 0 dimension to the gradients to represent the tower.
              expanded_g = tf.expand_dims(g, 0)

              # Append on a 'tower' dimension which we will average over below.
              grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads
