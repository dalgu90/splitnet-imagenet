#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, '..')
import resnet

in_ckpt_fname = './resnet_v1_50.ckpt'
out_ckpt_fname = '../baseline/ResNet50.ckpt'

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

# Load input checkpoint file
print('Load input checkpoint file')
reader = tf.train.NewCheckpointReader(in_ckpt_fname)

# Build an initialization operation to run below.
init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96),
    log_device_placement=False))
sess.run(init)


def get_model_var(name):
    return [v for v in tf.all_variables() if v.name == name][0]

def get_ckpt_var(name):
    return reader.get_tensor(name)

def assign_weight(var, value):
    assign_op = var.assign(value)
    sess.run(assign_op)

def assign_res_unit(model_name, ckpt_name):
    # if(model_name == 'res3a/branch2a'):
        # import pudb; pudb.set_trace()  # XXX BREAKPOINT

    print('\tassign_res_unit(%s, %s)' % (model_name, ckpt_name))
    kernel = get_ckpt_var('%s/weights' % ckpt_name)
    beta = get_ckpt_var('%s/BatchNorm/beta' % ckpt_name)
    gamma = get_ckpt_var('%s/BatchNorm/gamma' % ckpt_name)
    mu = get_ckpt_var('%s/BatchNorm/moving_mean' % ckpt_name)
    sigma = get_ckpt_var('%s/BatchNorm/moving_variance' % ckpt_name)
    kernel_var = get_model_var('%s/conv/kernel:0' % model_name)
    mu_var = get_model_var('%s/bn/mu:0' % model_name)
    sigma_var = get_model_var('%s/bn/sigma:0' % model_name)
    beta_var = get_model_var('%s/bn/beta:0' % model_name)
    gamma_var = get_model_var('%s/bn/gamma:0' % model_name)
    assign_weight(kernel_var, kernel)
    assign_weight(mu_var, mu)
    assign_weight(sigma_var, sigma)
    assign_weight(beta_var, beta)
    assign_weight(gamma_var, gamma)

# Assign weights
print('Assign weights')
assign_res_unit('conv1', 'resnet_v1_50/conv1')

res_units = [3, 4, 6, 3]
for i, N in enumerate(res_units):
    res_idx = i + 2
    for n in xrange(N):
        if n == 0:
            model_name = 'res%d%c/branch1' % (res_idx, 97+n)
            ckpt_name = 'resnet_v1_50/block%d/unit_%d/bottleneck_v1/shortcut' % (res_idx-1, n+1)
            assign_res_unit(model_name, ckpt_name)
        for m in range(3):
            model_name = 'res%d%c/branch2%c' % (res_idx, 97+n, 97+m)
            ckpt_name = 'resnet_v1_50/block%d/unit_%d/bottleneck_v1/conv%d' % (res_idx-1, n+1, m+1)
            assign_res_unit(model_name, ckpt_name)

print('\tassign_weight fc1000')
fc_weight_var = get_model_var('fc1000/fc/weights:0')
fc_bias_var = get_model_var('fc1000/fc/biases:0')
fc_weight = get_ckpt_var('resnet_v1_50/logits/weights')
fc_bias = get_ckpt_var('resnet_v1_50/logits/biases')
assign_weight(fc_weight_var, fc_weight.reshape([2048, 1000]))
assign_weight(fc_bias_var, fc_bias)


# Save as checkpoint
print('Save as checkpoint')
saver = tf.train.Saver(tf.all_variables())
saver.save(sess, out_ckpt_fname)

print('Done!')
