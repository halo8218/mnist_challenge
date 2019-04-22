"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import model as md
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
#sp_model = md.CompactGSOPwhModel()
#model = md.Model()
sp_model = md.GSOPcModel()
model = md.GSOPwhModel()

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.xent,
                                                   global_step=global_step)
sp_train_step = tf.train.AdamOptimizer(1e-4).minimize(sp_model.xent,
                                                   global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

sp_attack = LinfPGDAttack(sp_model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv sp_train', sp_model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv sp_train', sp_model.xent / batch_size)
tf.summary.image('images nat train', model.nat_test_img)
tf.summary.image('images adv test convention', model.adv_test_img)
tf.summary.image('images adv test spatial', sp_model.adv_test_img)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    sp_x_batch_adv = sp_attack.perturb(x_batch, y_batch, sess)

    convention_adv_test= attack.perturb(x_batch, y_batch, sess, False)
    spatial_adv_test= sp_attack.perturb(x_batch, y_batch, sess, False)

    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    sp_nat_dict = {sp_model.x_input: x_batch,
                sp_model.y_input: y_batch}

    sp_adv_dict = {sp_model.x_input: sp_x_batch_adv,
                sp_model.y_input: y_batch}

    summary_dict = {model.x_input: x_batch_adv,
                    model.y_input: y_batch,
                    model.adv_test_input: convention_adv_test,
                    model.nat_test_input: x_batch,
                    sp_model.x_input: sp_x_batch_adv,
                    sp_model.y_input: y_batch,
                    sp_model.adv_test_input: spatial_adv_test}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      sp_nat_acc = sess.run(sp_model.accuracy, feed_dict=sp_nat_dict)
      sp_adv_acc = sess.run(sp_model.accuracy, feed_dict=sp_adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))

      print('    sp_training nat accuracy {:.4}%'.format(sp_nat_acc * 100))
      print('    sp_training adv accuracy {:.4}%'.format(sp_adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=summary_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    sess.run(sp_train_step, feed_dict=sp_adv_dict)
    end = timer()
    training_time += end - start
