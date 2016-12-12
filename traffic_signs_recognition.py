# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt

training_file = "C:\\Users\\AW51R2\\code\\carnd\\traffic-signs-data\\train.p"
testing_file = "C:\\Users\\AW51R2\\code\\carnd\\traffic-signs-data\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

print("shape", X_train.shape)
print("unique labels", np.unique(np.concatenate([train['labels'], test['labels']])).size)

v_max = np.amax(train['features'])
v_min = np.amin(train['features'])
print("v_max", v_max, "v_min", v_min)
# it's RGB channel

n_classes = 43

import tensorflow as tf
# image is 32x32x3
x = tf.placeholder(tf.float32, [None,32,32,3])
# unique labels: 43
y = tf.placeholder(tf.float32, [None,n_classes])

layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'fully_connected': 120
}

weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 3, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [5*5*16, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes]))
}
biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')


def LeNet(x):
    #reshape from 2D to 4D
    x = tf.reshape(x, (-1,32,32,3))

    #define architecture
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'], strides=1) #28x28x6
    conv1 = maxpool2d(conv1, k=2) #14x14x6

    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'], strides=1) #10x10x16
    conv2 = maxpool2d(conv2, k=2) #5x5x16

    fc1 = tf.reshape(
        conv2,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fully_connected']), biases['fully_connected'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


fc2 = LeNet(x)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y,1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def one_hot(a):
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size),a] = 1
    return b

#take [0:50] as train, [51:100] as loss
x_validate, y_validate = X_train[100:150], one_hot(y_train[100:150])

EPOCHS = 10
BATCH_SIZE = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_x, batch_y = X_train[:50], one_hot(y_train[:50])

    sess.run(train_op, feed_dict={x:batch_x, y:batch_y})

    loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x:x_validate, y:y_validate})

    print("loss", loss, "acc", acc)




