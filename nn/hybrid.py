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


def load_data(filename):
    _x, _y = [], []
    f = open(filename, 'r')
    for line in f:
        _jsondata = json.loads(line)
        if _jsondata is not None:
            state = np.zeros((S_INFO, S_LEN))
            state[0] = np.array(_jsondata['b'])
            state[1] = np.array(_jsondata['d'])
            state[2, -1] = _jsondata['x']
            _x.append(state.T)
            _y.append(_jsondata['y'])
    _x = np.array(_x)
    _y = np.array(_y)
    _y = np.reshape(_y, [_y.shape[0], 1])
    return _x, _y


def load_h5(filename):
    h5f = h5py.File(filename, 'r')
    X = h5f['X']
    Y = h5f['Y']
    return X, Y


def build_error_net(x):
    inputs = tflearn.input_data(placeholder=x)
    dense_net = tflearn.fully_connected(inputs, FEATURE_NUM*2, activation='relu', regularizer="L2")
    dense_net = tflearn.fully_connected(dense_net, FEATURE_NUM, activation='relu', regularizer="L2")
    dense_net = tflearn.fully_connected(dense_net, 1, activation='sigmoid', regularizer="L2")
    return dense_net


def build_hybrid_net(x):
    inputs = tflearn.input_data(placeholder=x)
    with tf.name_scope('1d-cnn'):
        network_array = []
        for p in xrange(S_INFO):
            branch = tflearn.conv_1d(inputs[:, :, p:p+1], FEATURE_NUM, 3, activation='relu',regularizer="L2")
            branch = tflearn.flatten(branch)
            network_array.append(branch)
        out_cnn = tflearn.merge(network_array, 'concat')
    with tf.name_scope('gru'):
        net = tflearn.gru(inputs, FEATURE_NUM, return_seq=True)
        out_gru = tflearn.gru(net, FEATURE_NUM)

    header = tflearn.merge([out_cnn, out_gru], 'concat')
    dense_net = tflearn.fully_connected(header, FEATURE_NUM*2, activation='relu', regularizer="L2")
    dense_net = tflearn.fully_connected(dense_net, FEATURE_NUM*1, activation='relu', regularizer="L2")
    out = tflearn.fully_connected(dense_net, 1, activation='sigmoid', regularizer="L2")
    return out, header


def mape(y_pred, y_true):
    # print 'y_true', y_true.shape
    with tf.name_scope("MAPE"):
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))


def save_plot(pred_value, pred_err, y, j):
    _predict_value = np.reshape(pred_value, (pred_value.shape[0]))
    _predict_err = np.reshape(pred_err, (pred_err.shape[0]))
    plt.figure()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.grid(True)
    x = np.linspace(0, y.shape[0] - 1, y.shape[0])
    ax.set_ylim(0,1)
    #_predict_range = _predict_value + _predict_err
    _min_val = np.minimum( _predict_value - _predict_err, _predict_value)
    _max_val = np.maximum( _predict_value + _predict_err, _predict_value)
    ax.fill_between(x, _min_val,
                    _max_val, color='red', lw=1, alpha=0.3)
    ax.plot(x, y)
    ax.plot(x, _predict_value, color='red')
    savefig(LOG + '/save_' + str(j) + '.png')

def tensor_network(trainX, trainY, testX, testY):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        x = tf.placeholder(shape=(None, S_LEN, S_INFO), dtype=tf.float32)
        y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)

        hybrid_net, hybrid_header = build_hybrid_net(x)
        hybrid_net_loss =  tflearn.objectives.mean_square(hybrid_net, y_)
        #tf.div(tf.abs(tf.subtract(hybrid_net,y_)),y_)
        #tflearn.objectives.mean_square(hybrid_net, y_)
        #tf.abs(hybrid_net - y_) / y_
        #tflearn.objectives.mean_square(hybrid_net, y_)
                        
        hybrid_train_op = tf.train.AdamOptimizer(
            learning_rate=0.0000625).minimize(hybrid_net_loss)
        hybrid_accuracy = tf.reduce_mean(hybrid_net_loss)
        hybrid_err = tf.abs(tf.subtract(y_,hybrid_net))

        err_net = build_error_net(hybrid_header)
        err_net_loss = tflearn.objectives.mean_square(err_net, hybrid_err)
        err_train_op = tf.train.AdamOptimizer(
            learning_rate=0.00001).minimize(err_net_loss)
        err_accuracy = tf.reduce_mean(err_net_loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        epoch = 300
        batch = 3000
        train_len = trainX.shape[0]
        _writer = open(LOG + '.csv', 'w')
        for j in range(epoch):
            i = 0
            while i < train_len - batch:
                batch_xs, batch_ys = trainX[i:i+batch], trainY[i:i+batch]
                sess.run(hybrid_train_op, feed_dict={
                         x: batch_xs, y_: batch_ys})
                sess.run(err_train_op, feed_dict={x: batch_xs, y_: batch_ys})
                i += batch

            _test_accuracy = sess.run(
                hybrid_accuracy, feed_dict={x: testX, y_: testY})
            _err_accuracy = sess.run(err_accuracy, feed_dict={
                                     x: testX, y_: testY})
            _predict_value = sess.run(hybrid_net, feed_dict={x: testX})
            _predict_err = sess.run(err_net, feed_dict={x: testX})
            _test_len = testY.shape[0]
            _count = 0
            for (_y, _val, _err) in zip(testY, _predict_value, _predict_err):
                #_range_val = _val + _err
                _min_val = min(_val - _err, _val)
                _max_val = max(_val + _err, _val)
                #_min_val = min(_range_val,_val)
                #_max_val = max(_range_val,_val)
                if _y >= _min_val and _y <= _max_val:
                    _count += 1
            _writer.write(str(j) + ',' + str(_test_accuracy) + ',' +
                          str(_err_accuracy) + ',' + str(_count * 100.0 / _test_len))
            _writer.write('\n')
            print 'epoch', j, 'value', _test_accuracy, 'error', _err_accuracy, 'total', _count * \
                100.0 / _test_len, '%'
            if j % 10 == 0:
                saver.save(sess, LOG + "_model/nn_model_ep_" + str(j) + ".ckpt")
            save_plot(_predict_value, _predict_err, testY, j)
        _writer.close()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    #trainX, trainY = load_data('dataset.db')
    #testX, testY = load_data('3')
    trainX, trainY = load_h5('train.h5')
    testX, testY = load_h5('test.h5')
    print 'starting predict network'
    os.system('mkdir ' + LOG)
    os.system('mkdir ' + LOG + '_model')
    tensor_network(trainX, trainY, testX, testY)
    print 'done'


if __name__ == '__main__':
    main()
