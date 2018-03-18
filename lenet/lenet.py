from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# start pre-load data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels


X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


X_train, y_train = shuffle(X_train, y_train)
# end


EPOCHES = 10
BATCH_SIZE = 128


"""
conv-1: 32*32*1 -> 28*28*6
relu
pool-1: 28*28*6 -> 14*14*6

conv-2: 14*14*6 -> 10*10*16
relu
pool-2: 10*10*16 -> 5*5*16

flatten: 5*5*16 -> 400

fully-connected-1: 400 -> 120
relu

fully-connected-2: 120 -> 84
relu

fully-connected-3: 84 -> 10
"""
def leNet(x):
    mu = 0
    sigma = 0.1

    # 32*32*1 -> 28*28*6
    conv1_W = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    # 28*28*6 -> 14*14*6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 14*14*6 -> 10*10*16
    conv2_W = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 10*10*16 -> 5*5*16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 5*5*16 -> 400
    fc0 = flatten(conv2)

    # 400 -> 120
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    fc1 = tf.nn.relu(fc1)

    # 120 -> 84
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    fc2 = tf.nn.relu(fc2)

    # 84 -> 10
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    return fc3


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
y_one_hot = tf.one_hot(y, 10)


rate = 0.001
logits = leNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(x_data, y_data):
    total_acc = 0.0
    count = len(x_data)
    sess = tf.get_default_session()

    for offset in range(0, count, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        temp = sess.run(acc, feed_dict={x: batch_x, y: batch_y})
        temp = temp * len(batch_x)
        total_acc += temp

    return total_acc / len(x_data)


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHES):
            for offset in range(0, len(X_train), BATCH_SIZE):
                next_x, next_y = X_train[offset:offset+BATCH_SIZE], y_train[offset:offset+BATCH_SIZE]
                sess.run(optimizer, feed_dict={x:next_x, y:next_y})

            acc = evaluate(X_validation, y_validation)
            print(acc)
        saver.save(sess, "./data/lenet")


if __name__ == '__main__':
    train()