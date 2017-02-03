from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import h5py
import numpy as np
import math

import tensorflow as tf

FLAGS = None

def main(_):
  hiddenWidth1 = 100
  hiddenWidth2 = 64
  outputWidth = 5
  weightInit = -1
  batchSize = 4
  gamma = 0.7

  dataOut = h5py.File('skillWeightsQ.h5', 'w') #_2Layer.h5', 'w')

  # Import data
  print('Loading data...')
  data = h5py.File(FLAGS.file, 'r')
  numSkills = data.get('numberSkills')
  print('Number of skills is ' + str(numSkills[()]))

  dataOut.create_dataset('hiddenWidth', data=hiddenWidth1)
  dataOut.create_dataset('numberSkills', data=numSkills)

  for skill in range(numSkills[()]):
    activations = np.array(data.get('activations_' + str(skill)))
    actions = (np.array(data.get('actions_' + str(skill))) - 1)
    termination = np.array(data.get('termination_' + str(skill)))

    print('Creating model...')
    # Create the model
    step = tf.Variable(0, trainable=False)  # cant attach non trainable variable to gpu
    with tf.device('/gpu:1'):
        x = tf.placeholder(tf.float32, [None, 512, ])


        # Hidden Layer1
        W_hidden1 = tf.Variable(tf.truncated_normal([512, hiddenWidth1], stddev=0.1))
        b_hidden1 = tf.Variable(tf.constant(0.1, shape=[hiddenWidth1]))
        y_hidden1 = tf.add(tf.matmul(x, W_hidden1), b_hidden1)
        act_hidden1 = tf.nn.relu(y_hidden1)

        # Hidden Layer2
        W_hidden2 = tf.Variable(tf.random_uniform([hiddenWidth1, hiddenWidth2], weightInit, 1))
        b_hidden2 = tf.Variable(tf.random_uniform([hiddenWidth2], weightInit, 1))
        y_hidden2 = tf.add(tf.matmul(act_hidden1, W_hidden2), b_hidden2)
        act_hidden2 = tf.nn.relu(y_hidden2)

        # Output Layer
        W_output = tf.Variable(tf.truncated_normal([hiddenWidth1, outputWidth], stddev=0.1))
        #W_output = tf.Variable(tf.truncated_normal([hiddenWidth2, outputWidth], stddev=0.1))
        b_output = tf.Variable(tf.constant(0.1, shape=[outputWidth]))
        y = tf.add(tf.matmul(act_hidden1, W_output), b_output)
        #y = tf.add(tf.matmul(act_hidden2, W_output), b_output)
        predict = tf.argmax(y, 1)
        '''

        # Linear only
        W = tf.Variable(tf.random_uniform([512, outputWidth], weightInit, 0.01))
        b = tf.Variable(tf.random_uniform([outputWidth], weightInit, 0.01))
        y = tf.add(tf.matmul(x, W), b)
        predict = tf.argmax(y, 1)
        '''
        nextQ = tf.placeholder(shape=[None, outputWidth, ], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - y))
        #loss = tf.nn.softmax_cross_entropy_with_logits(y, nextQ)

        rate = tf.train.exponential_decay(0.0005, step, 250, 0.9999)
        trainer = tf.train.AdamOptimizer(rate) #learning_rate=0.000001)  # GradientDescentOptimizer(learning_rate=0.0001)
        updateModel = trainer.minimize(loss, global_step=step)

        # train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
        # train_step = tf.train.RMSPropOptimizer(0.1).minimize(cross_entropy)
        # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    #tf.global_variables_initializer().run()
    tf.initialize_all_variables().run()
    #sess.run(tf.initialize_all_variables())
    # Train
    maxQ = 1
    iteration = 0
    print('Training...')
    for _ in range(20000000):
      if (_ % 1000000 == 0 and _ > 0): # and False):
	testPredictions = sess.run(predict, feed_dict={x: activations[int(math.ceil(activations.shape[0] * 0.8)) + 1:activations.shape[0],:]})
        trainPredictions = sess.run(predict, feed_dict={x: activations[0:int(math.ceil(activations.shape[0] * 0.8)),:]})
        print('Done ' + str(_) + ' iterations. testing error is: ' + str(100 * np.sum(np.sign(np.absolute(testPredictions - actions[int(math.ceil(activations.shape[0] * 0.8)) + 1:activations.shape[0]]))) * 1.0 / (activations.shape[0] - int(math.ceil(activations.shape[0] * 0.8)) + 1)) + '%, training error is: ' + str(100 * np.sum(np.sign(np.absolute(trainPredictions - actions[0:int(math.ceil(activations.shape[0] *   0.8))]))) * 1.0 / (int(math.ceil(activations.shape[0] * 0.8)))) + '%')
        print('Loss: ' + str(loss_val) + ', Skill#: ' + str(skill))

      index = np.random.randint(int(math.ceil(activations.shape[0] * 0.8)), size=batchSize)
      '''
      iteration = iteration + 1
      iteration = iteration % int(math.ceil(activations.shape[0] * 0.8))
      index = np.array([iteration])
      '''

      allQ = sess.run(y,feed_dict={x: activations[index, :]})

      Q1 = sess.run(y,feed_dict={x: activations[index + 1, :]})
      targetQ = np.ones(allQ.shape) * -1
      #targetQ = allQ
      for i in range(index.shape[0]):
        if termination[index[i]] == 1:
	    Q = 0
        else:
	    Q = np.max(Q1[i, :]) * gamma

        # maxQ = max(maxQ, abs(Q))
        targetQ[i, :] = targetQ[i, :] + Q - gamma * gamma
        targetQ[i, int(actions[index[i]])] = targetQ[i, int(actions[index[i]])] + gamma * gamma
      targetQ = targetQ * 1.0 / maxQ

      '''
      targetQ = np.zeros(allQ.shape)
      for i in range(index.shape[0]):
        targetQ[i, int(actions[index[i]])] = 1
      '''

      _, loss_val = sess.run([updateModel, loss], feed_dict={x: activations[index, :], nextQ: targetQ})

    # Test trained model
    print('Testing model on ' + str(len(actions[int(math.ceil(activations.shape[0] * 0.8)) + 1:activations.shape[0]])) + ' samples...')

    prediction = tf.argmax(y,1)
    predictions = prediction.eval(feed_dict={x: activations[int(math.ceil(activations.shape[0] * 0.8)) + 1:activations.shape[0],:]}, session=sess)
    # print(predictions)
    # print(actions[int(math.ceil(np.array(activations).shape[0] * 0.8)) + 1:np.array(activations).shape[0]])
    print('Testing error:')
    print(100 * np.sum(np.sign(np.absolute(predictions - actions[int(math.ceil(activations.shape[0] * 0.8)) + 1:activations.shape[0]]))) * 1.0 / (activations.shape[0] - int(math.ceil(activations.shape[0] * 0.8)) + 1))
    print('Training error:')
    predictions = prediction.eval(feed_dict={x: activations[0:int(math.ceil(activations.shape[0] * 0.8)),:]}, session=sess)
    print(100 * np.sum(np.sign(np.absolute(predictions - actions[0:int(math.ceil(activations.shape[0] * 0.8))]))) * 1.0 / (int(math.ceil(activations.shape[0] * 0.8))))

    dataOut.create_dataset('W_hidden_' + str(skill), data=sess.run(W_hidden1))
    dataOut.create_dataset('b_hidden_' + str(skill), data=sess.run(b_hidden1))
    dataOut.create_dataset('W_output_' + str(skill), data=sess.run(W_output))
    dataOut.create_dataset('b_output_' + str(skill), data=sess.run(b_output))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-file', type=str, required=True,
                      help='Name of Skill extraction file.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main)
