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


class Dataset(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = one_hot(y_train)
        assert(x_train.shape[0] == y_train.shape[0])
        self._num_examples = x_train.shape[0]

        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # finish epoch
            self._epochs_completed += 1
            # shuffle
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            print("shuffle!")
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]
            print("after shuffle shape", self.x_train.shape, self.y_train.shape)
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self.x_train[start:end].reshape(BATCH_SIZE, 32*32*3), self.y_train[start:end]


print("shape", X_train.shape)
print("unique labels", np.unique(np.concatenate([train['labels'], test['labels']])).size)

v_max = np.amax(train['features'])
v_min = np.amin(train['features'])
print("v_max", v_max, "v_min", v_min)
# it's RGB channel

n_classes = 43

import tensorflow as tf
# image is 32x32x3
x = tf.placeholder(tf.float32, (None, 32*32*3))
# unique labels: 43
y = tf.placeholder(tf.float32, (None, n_classes))

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
validation_size = 5000
train_dataset = Dataset(X_train[validation_size:], y_train[validation_size:])
validation_dataset = Dataset(X_train[:validation_size], y_train[:validation_size])

EPOCHS = 20
BATCH_SIZE = 50

def eval_data(dataset):
    steps_per_epoch = dataset._num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss / num_examples, total_acc / num_examples



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    steps_per_epoch = train_dataset._num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE

    for i in range(EPOCHS):
        for step in range(steps_per_epoch):
            batch_x, batch_y = train_dataset.next_batch(BATCH_SIZE)
            loss = sess.run(train_op, feed_dict={x:batch_x, y:batch_y})

        val_loss, val_acc = eval_data(validation_dataset)
        print("EPOCH {}...".format(i+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.3f}".format(val_acc))
        print()



