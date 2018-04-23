import tflearn
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import time
from matplotlib.pyplot import plot, savefig
import h5py
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.conv import global_max_pool
import sys
# 1.predict throughput network
#   in -> state;out -> throughput predicted; loss -> mape(throughput pred,ground truth)
#   update gradient
# 2. predict error network
#   in -> state;out -> loss; err_loss -> mean_square_error(mape_loss,loss)
#   update gradient
if len(sys.argv) > 1:
    LOG = sys.argv[1]
else:
    LOG = 'log'
FEATURE_NUM = 128
S_INFO = 3
S_LEN = 8

def load_h5(filename):
    h5f = h5py.File(filename, 'r')
    X = h5f['X']
    Y = h5f['Y']
    return X, Y

def hybrid_header(x, reuse=False):
    dense_net = tflearn.fully_connected(x, FEATURE_NUM, activation='relu')
    #dense_net = tflearn.fully_connected(dense_net, int(FEATURE_NUM / 2), activation='relu')
    return dense_net


def build_error_net(x):
    inputs = tflearn.input_data(placeholder=x)
    #dense_net = tflearn.fully_connected(inputs, FEATURE_NUM, activation='relu')
    dense_net = tflearn.fully_connected(inputs, 1, activation='linear')
    return dense_net


def build_hybrid_net(x=None):
    inputs = tflearn.input_data(placeholder=x)
    header = hybrid_header(inputs)
    out = tflearn.fully_connected(header, 1, activation='linear')
    return out,header


def mape(y_pred, y_true):
    # print 'y_true', y_true.shape
    with tf.name_scope("MAPE"):
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))

def save_plot(pred_value,pred_err,y,j):
    _predict_value = np.reshape(pred_value, (pred_value.shape[0]))
    _predict_err = np.reshape(pred_err, (pred_err.shape[0]))
    plt.figure()
    fig, ax = plt.subplots(figsize=(9, 6),dpi=150)
    ax.grid(True)
    x = np.linspace(0, y.shape[0] - 1, y.shape[0])
    _min_val = np.minimum(_predict_value - _predict_err,_predict_value)
    _max_val = np.maximum(_predict_value + _predict_err,_predict_value)
    ax.fill_between(x, _min_val,
                    _max_val, color='red', lw=1, alpha=0.4)
    ax.plot(x, y)
    ax.plot(x, _predict_value, color='red')
    savefig(LOG + '/save_' + str(j) + '.png')
    
def tensor_network(trainX, trainY, testX, testY):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    x = tf.placeholder(shape=(None, S_LEN, S_INFO), dtype=tf.float32)
    y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    #merge = tf.placeholder(shape=(None, FEATURE_NUM * 2), dtype=tf.float32)

    hybrid_net, hybrid_header = build_hybrid_net(x)
    hybrid_net_loss = tf.abs(hybrid_net -  y_) / y_
    hybrid_train_op = tf.train.AdamOptimizer(
        learning_rate=0.000625).minimize(hybrid_net_loss)
    hybrid_accuracy = tf.reduce_mean(hybrid_net_loss)
    hybrid_err = tf.abs(hybrid_net - y_)

    err_net = build_error_net(hybrid_header)
    err_net_loss = tflearn.objectives.mean_square(err_net, hybrid_err)
    err_train_op = tf.train.AdamOptimizer(
        learning_rate=0.000625).minimize(err_net_loss)
    err_accuracy = tf.reduce_mean(err_net_loss)

    sess.run(tf.global_variables_initializer())

    epoch = 100
    batch = 4000
    train_len = trainX.shape[0]
    _writer = open(LOG + '.csv','w')
    for j in range(epoch):
        i = 0
        while i < train_len - batch:
            batch_xs, batch_ys = trainX[i:i+batch], trainY[i:i+batch]
            sess.run(hybrid_train_op, feed_dict={x: batch_xs, y_: batch_ys})
            sess.run(err_train_op, feed_dict={x: batch_xs, y_: batch_ys})
            i += batch
            #if i % (batch * 20) == 0:
            #    _test_accuracy = sess.run(hybrid_accuracy, feed_dict={x: testX, y_: testY})
            #    _err_accuracy = sess.run(err_accuracy, feed_dict={x: testX, y_: testY})
            #    print 'epoch', j, 'step', i, 'test accuracy', _test_accuracy, 'error accuracy', _err_accuracy

        _test_accuracy = sess.run(hybrid_accuracy, feed_dict={x: testX, y_: testY})
        _err_accuracy = sess.run(err_accuracy, feed_dict={x: testX, y_: testY})
        _predict_value = sess.run(hybrid_net, feed_dict={x: testX})
        _predict_err = sess.run(err_net, feed_dict={x:testX})
        _test_len = testY.shape[0]
        _count = 0
        for (_y,_val,_err) in zip(testY,_predict_value,_predict_err):
            _min_val = min(_val - _err, _val)
            _max_val = max(_val + _err, _val)
            if _y >= _min_val and _y <= _max_val:
                _count += 1
        _writer.write(str(j) + ',' + str(_test_accuracy) + ',' + str(_err_accuracy) + ',' + str(_count * 100.0 / _test_len))
        _writer.write('\n')
        print 'epoch', j, 'value', _test_accuracy, 'error', _err_accuracy,'total',_count * 100.0 / _test_len,'%'
        save_plot(_predict_value,_predict_err,testY,j)
    _writer.close()
    #return _predict_value, _predict_err

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    #trainX, trainY = load_data('dataset.db')
    #testX, testY = load_data('3')
    trainX, trainY = load_h5('train.h5')
    testX, testY = load_h5('test.h5')
    print 'starting predict network'
    os.system('mkdir ' + LOG)
    tensor_network(trainX, trainY, testX, testY)
    print 'done'


if __name__ == '__main__':
    main()
