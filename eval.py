#!/usr/bin/env python

import sys
import os
from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import imagenet_input as data_input
import resnet
import utils



# Dataset Configuration
tf.app.flags.DEFINE_string('train_dataset', 'scripts/train_shuffle.txt', """Path to the ILSVRC2012 the training dataset list file""")
tf.app.flags.DEFINE_string('train_image_root', '/data1/common_datasets/ILSVRC2012/train/', """Path to the root of ILSVRC2012 training images""")
tf.app.flags.DEFINE_string('test_dataset', 'scripts/val.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('test_image_root', '/data1/common_datasets/ILSVRC2012/val/', """Path to the root of ILSVRC2012 test images""")
tf.app.flags.DEFINE_string('class_list', 'scripts/synset_words.txt', """Path to the ILSVRC2012 class name list file""")
tf.app.flags.DEFINE_integer('num_classes', 1000, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 1281166, """Number of training images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 50000, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('num_gpu', 1, """Number of GPUs""")
tf.app.flags.DEFINE_integer('batch_size', 32, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('split', False, """Whether to use split""")
tf.app.flags.DEFINE_string('cluster_path', './scripts/clustering.pkl', """Path to 3-level clustering of ImageNet.""")
tf.app.flags.DEFINE_boolean('no_logit_map', False, """Whether to re-map logit for classes to be clustered correctly.
                                                      If set to True, the classes will be wrongly clustered.""")

# Testing Configuration
tf.app.flags.DEFINE_string('ckpt_path', '', """Path to the checkpoint or dir.""")
tf.app.flags.DEFINE_bool('train_data', False, """Whether to test over training set.""")
tf.app.flags.DEFINE_integer('test_iter', 100, """Number of iterations during a test""")
tf.app.flags.DEFINE_string('output', '', """Path to the output txt.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.95, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

# Other Configuration(not needed for testing, but required fields in
# build_model())
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tImageNet training root: %s' % FLAGS.train_image_root)
    print('\tImageNet training list: %s' % FLAGS.train_dataset)
    print('\tImageNet test root: %s' % FLAGS.test_image_root)
    print('\tImageNet test list: %s' % FLAGS.test_dataset)
    print('\tImageNet class name list: %s' % FLAGS.class_list)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of test images: %d' % FLAGS.num_test_instance)

    print('[Network Configuration]')
    print('\tNumber of GPUs: %d' % FLAGS.num_gpu)
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tSplitted Network: %s' % FLAGS.split)
    if FLAGS.split:
        print('\tClustering path: %s' % FLAGS.cluster_path)
        print('\tNo logit map: %s' % FLAGS.no_logit_map)

    print('[Testing Configuration]')
    print('\tCheckpoint path: %s' % FLAGS.ckpt_path)
    print('\tDataset: %s' % ('Training' if FLAGS.train_data else 'Test'))
    print('\tNumber of testing iterations: %d' % FLAGS.test_iter)
    print('\tOutput path: %s' % FLAGS.output)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)


    with tf.Graph().as_default():
        # The CIFAR-100 dataset
        with tf.variable_scope('test_image'):
            if FLAGS.train_data:
                test_images, test_labels = data_input.inputs(FLAGS.train_image_root, FLAGS.train_dataset, FLAGS.batch_size*FLAGS.num_gpu, False)
            else:
                test_images, test_labels = data_input.inputs(FLAGS.test_image_root, FLAGS.test_dataset, FLAGS.batch_size*FLAGS.num_gpu, False)

        # The class labels
        with open(FLAGS.class_list) as fd:
            classes = [temp.strip()[:30] for temp in fd.readlines()]

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [FLAGS.batch_size*FLAGS.num_gpu, data_input.IMAGE_HEIGHT, data_input.IMAGE_WIDTH, 3])
        labels = tf.placeholder(tf.int32, [FLAGS.batch_size*FLAGS.num_gpu])

        # Build model
        hp = resnet.HParams(num_gpu=FLAGS.num_gpu,
                            batch_size=FLAGS.batch_size,
                            split=FLAGS.split,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            no_logit_map=FLAGS.no_logit_map)
        network = resnet.ResNet(hp, images, labels, None)
        if FLAGS.split:
            network.set_clustering(clustering)
        network.build_model()
        print('%d flops' % network._flops)
        print('%d params' % network._weights)
        # network.build_train_op()  # NO training op

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
        if os.path.isdir(FLAGS.ckpt_path):
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            # Restores from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
               print('\tRestore from %s' % ckpt.model_checkpoint_path)
               saver.restore(sess, ckpt.model_checkpoint_path)
            else:
               print('No checkpoint file found in the dir [%s]' % FLAGS.ckpt_path)
               sys.exit(1)
        elif os.path.isfile(FLAGS.ckpt_path):
            print('\tRestore from %s' % FLAGS.ckpt_path)
            saver.restore(sess, FLAGS.ckpt_path)
        else:
            print('No checkpoint file found in the path [%s]' % FLAGS.ckpt_path)
            sys.exit(1)

        # Start queue runners
        tf.train.start_queue_runners(sess=sess)

        # Testing!
        result_ll = [[0, 0] for _ in range(FLAGS.num_classes)] # Correct/wrong counts for each class
        test_loss = 0.0, 0.0
        for i in range(FLAGS.test_iter):
            test_images_val, test_labels_val = sess.run([test_images, test_labels])
            preds_val, loss_value, acc_value = sess.run([network.preds, network.loss, network.acc],
                        feed_dict={network.is_train:False, images:test_images_val, labels:test_labels_val})
            test_loss += loss_value
            for j in range(FLAGS.batch_size*FLAGS.num_gpu):
                correct = 0 if test_labels_val[j] == preds_val[j] else 1
                result_ll[test_labels_val[j] % FLAGS.num_classes][correct] += 1
            if i % FLAGS.display == 0:
                format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), i, loss_value, acc_value))
        test_loss /= FLAGS.test_iter

        # Summary display & output
        acc_list = [float(r[0])/float(r[0]+r[1]) for r in result_ll]
        result_total = np.sum(np.array(result_ll), axis=0)
        acc_total = float(result_total[0])/np.sum(result_total)

        print 'Class    \t\t\tT\tF\tAcc.'
        format_str = '%-31s %7d %7d %.5f'
        for i in range(FLAGS.num_classes):
            print format_str % (classes[i], result_ll[i][0], result_ll[i][1], acc_list[i])
        print(format_str % ('(Total)', result_total[0], result_total[1], acc_total))

        # Output to file(if specified)
        if FLAGS.output.strip():
            with open(FLAGS.output, 'w') as fd:
                fd.write('Class    \t\t\tT\tF\tAcc.\n')
                format_str = '%-31s %7d %7d %.5f'
                for i in range(FLAGS.num_classes):
                    t, f = result_ll[i]
                    format_str = '%-31s %7d %7d %.5f\n'
                    fd.write(format_str % (classes[i].replace(' ', '-'), t, f, acc_list[i]))
                fd.write(format_str % ('(Total)', result_total[0], result_total[1], acc_total))


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
