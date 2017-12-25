# Make TFGAN models and TF-Slim models discoverable.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import  matplotlib
matplotlib.use('Agg')
import numpy as np
import time
import functools
import tensorflow as tf
tfgan = tf.contrib.gan
import data_provider
import util
from datasets import download_and_convert_cifar10
import networks
from train_util import  save_image

slim = tf.contrib.slim
layers = tf.contrib.layers
ds = tf.contrib.distributions
batch_size = 32
# noise_dims = 64
cat_dim, cont_dim, noise_dims = 10, 2, 64
CIFAR_DATA_DIR = './cifar10-data'
CIFAR_IMAGE_DIR = './cifar10-image'
def _learning_rate():
  generator_lr = tf.train.exponential_decay(
      learning_rate=0.0001,
      global_step=tf.train.get_or_create_global_step(),
      decay_steps=100000,
      decay_rate=0.9,
      staircase=True)
  discriminator_lr = 0.001
  return generator_lr, discriminator_lr


def _optimizer(gen_lr, dis_lr, use_sync_replicas):
  """Get an optimizer, that's optionally synchronous."""
  generator_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9, momentum=0.1)
  discriminator_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95, momentum=0.1)

if __name__ == '__main__':
    if not tf.gfile.Exists(CIFAR_DATA_DIR):
        tf.gfile.MakeDirs(CIFAR_DATA_DIR)
        download_and_convert_cifar10.run(CIFAR_DATA_DIR)
    if not tf.gfile.Exists(CIFAR_IMAGE_DIR):
        tf.gfile.MakeDirs(CIFAR_IMAGE_DIR)
    images, one_hot_labels, _, _ = data_provider.provide_data(batch_size,CIFAR_DATA_DIR)
    noise = tf.random_normal([batch_size, 64])
    generator_fn = networks.generator
    discriminator_fn = networks.discriminator
    generator_inputs = noise
    gan_model = tfgan.gan_model(
        generator_fn,
        discriminator_fn,
        real_data=images,
        generator_inputs=generator_inputs)
    
    gan_loss = tfgan.gan_loss(gan_model,
                              gradient_penalty_weight=1.0,
                              add_summaries=True)

    # Get the GANTrain ops using the custom optimizers and optional
    # discriminator weight clipping.
    generator_lr = tf.train.exponential_decay(
        learning_rate=0.0001,
        global_step=tf.train.get_or_create_global_step(),
        decay_steps=100000,
        decay_rate=0.9,
        staircase=True)
    discriminator_lr = 0.001
    generator_opt = tf.train.RMSPropOptimizer(generator_lr, decay=.9, momentum=0.1)
    discriminator_opt = tf.train.RMSPropOptimizer(discriminator_lr, decay=.95, momentum=0.1)
    train_ops = tfgan.gan_train_ops(
        gan_model,
        gan_loss,
        generator_optimizer=generator_opt,
        discriminator_optimizer=discriminator_opt,
        summarize_gradients=True,
        colocate_gradients_with_ops=True,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    train_step_fn = tfgan.get_sequential_train_steps()
    global_step = tf.train.get_or_create_global_step()
    loss_values, mnist_score_values = [], []
    generated_data_to_visualize = tfgan.eval.image_reshaper(
        gan_model.generated_data[:20, ...], num_cols=10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with slim.queues.QueueRunners(sess):
            start_time = time.time()
            for i in range(100001):
                cur_loss, _ = train_step_fn(
                    sess, train_ops, global_step, train_step_kwargs={})
                loss_values.append((i, cur_loss))
                if i % 500 == 0:
                    print('Current epoch: %d' % i)
                    print('Current loss: %f' % cur_loss)
                    img_name = "result_" + str(i)
                    save_image(sess.run(generated_data_to_visualize),CIFAR_IMAGE_DIR,img_name)