#!/usr/bin/env python

import cPickle as pickle

import tensorflow as tf
import numpy as np

import resnet

model_pkl_fname = 'baseline/ResNet-50.pkl'
model_ckpt_fname = 'baseline/ResNet50.ckpt'

# Build model to load weights
global_step = tf.Variable(0, trainable=False, name='global_step')
images = tf.placeholder(tf.float32, [100, 224, 224, 3])
labels = tf.placeholder(tf.int32, [100])
hp = resnet.HParams(batch_size=100,
                    num_classes=1000,
                    weight_decay=0.0005,
                    momentum=0.9)
network = resnet.ResNet(hp, images, labels, global_step)
network.build_model()

# Load pkl weight file
print('Load pkl weight file')
with open(model_pkl_fname) as fd:
    weights = pickle.load(fd)

# Build an initialization operation to run below.
init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96),
    log_device_placement=False))
sess.run(init)


def get_tf_var(name):
    return [v for v in tf.all_variables() if v.name == name][0]

def assign_weight(var, value):
    assign_op = var.assign(value)
    sess.run(assign_op)

def assign_res_unit(tf_name, pkl_conv_name, pkl_bn_name):
    # if(tf_name == 'res3a/branch2a'):
        # import pudb; pudb.set_trace()  # XXX BREAKPOINT

    print('\tassign_res_unit(%s, %s, %s)' % (tf_name, pkl_conv_name, pkl_bn_name))
    kernel = weights[pkl_conv_name]
    mu, sigma, gamma, beta = weights[pkl_bn_name]
    kernel_var = get_tf_var('%s/conv/kernel:0' % tf_name)
    mu_var = get_tf_var('%s/bn/mu:0' % tf_name)
    sigma_var = get_tf_var('%s/bn/sigma:0' % tf_name)
    beta_var = get_tf_var('%s/bn/beta:0' % tf_name)
    gamma_var = get_tf_var('%s/bn/gamma:0' % tf_name)
    assign_weight(kernel_var, kernel)
    assign_weight(mu_var, mu)
    assign_weight(sigma_var, sigma)
    assign_weight(beta_var, beta)
    assign_weight(gamma_var, gamma)

# Assign weights
print('Assign weights')
assign_res_unit('conv1', 'conv1', 'bn_conv1')

res_units = [3, 4, 6, 3]
for i, N in enumerate(res_units):
    res_idx = i + 2
    for n in xrange(N):
        if n == 0:
            tf_name = 'res%d%c/branch1' % (res_idx, 97+n)
            pkl_conv_name = 'res%d%c_branch1' % (res_idx, 97+n)
            pkl_bn_name = 'bn%d%c_branch1' % (res_idx, 97+n)
            assign_res_unit(tf_name, pkl_conv_name, pkl_bn_name)
        for m in range(3):
            tf_name = 'res%d%c/branch2%c' % (res_idx, 97+n, 97+m)
            pkl_conv_name = 'res%d%c_branch2%c' % (res_idx, 97+n, 97+m)
            pkl_bn_name = 'bn%d%c_branch2%c' % (res_idx, 97+n, 97+m)
            assign_res_unit(tf_name, pkl_conv_name, pkl_bn_name)

print('\tassign_weight fc1000')
fc_weight_var = get_tf_var('fc1000/fc/weights:0')
fc_bias_var = get_tf_var('fc1000/fc/biases:0')
fc_weight, fc_bias = weights['fc1000']
assign_weight(fc_weight_var, fc_weight)
assign_weight(fc_bias_var, fc_bias)


# Save as checkpoint
print('Save as checkpoint')
saver = tf.train.Saver(tf.all_variables())
saver.save(sess, model_ckpt_fname)

print('Done!')
