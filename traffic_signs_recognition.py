# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

training_file = "C:\\Users\\AW51R2\\code\\carnd\\traffic-signs-data\\train.p"
testing_file = "C:\\Users\\AW51R2\\code\\carnd\\traffic-signs-data\\test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

n_classes = 43
count = train['features'].shape[0]

'''
Question 1: preprocessing
- split data into test and validation set, also apply one hot encoding to y
- [TODO] convert features into YUV
'''
from sklearn.model_selection import train_test_split

def one_hot(a):
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size),a] = 1
    return b

train_features = train['features'].astype(float)
train_features = train_features / 255. - 0.5 #normalize

test_features = test['features'].astype(float)
test_features = test_features / 255. - 0.5
test_labels = one_hot(test['labels'])


print(train_features[0][0][0])

# YUV doesn't work...
'''
for i in range(train_features.shape[0]):
    yuv = cv2.cvtColor(train_features[i], cv2.COLOR_RGB2YUV)
    train_features[i] = yuv
'''


# generate translated image:
TRANSLATE_DELTA = 4
JIGGLE_COPY = 8

bins = np.bincount(train['labels'])
average_label_count = np.average(bins)

# for each image, generate N copies that are randomly jiggled on X,Y with [0.9,1.1] of delta 4

train_and_jiggle = [train_features]
train_and_jiggle_labels = [train['labels']]

for i in range(JIGGLE_COPY):
    train_and_jiggle.append(np.copy(train_features))
    train_and_jiggle_labels.append(np.copy(train['labels']))

for i in range(count):
    for j in range(JIGGLE_COPY):
        m = np.float32([[1,0,random.uniform(0.9,1.1)*TRANSLATE_DELTA], [0,1,random.uniform(0.9,1.1)*TRANSLATE_DELTA]])
        train_and_jiggle[j+1][i] = cv2.warpAffine(train_features[i], m, (32,32))


train_data_all = np.concatenate(train_and_jiggle)

train_label_all = np.concatenate(train_and_jiggle_labels)

print("fake data generated", train_data_all.shape)

def shuffle():
    x_train, x_validate, y_train, y_validate = train_test_split(
        train_data_all,
        train_label_all,
        test_size=0.2, random_state=42, stratify=train_label_all
    )

    y_train = one_hot(y_train)
    y_validate = one_hot(y_validate)

    return x_train, x_validate, y_train, y_validate





'''
Question 3: model architecture
2 layer ConvNet
'''

import tensorflow as tf

p_stdev = 0.1

layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'fully_connected': 120
}

weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 3, layer_width['layer_1']], stddev=p_stdev)),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']], stddev=p_stdev)),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [5*5*16, layer_width['fully_connected']], stddev=p_stdev)),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes], stddev=p_stdev))
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

    # todo: connect conv1 to fc1

    fc1 = tf.reshape(
        conv2,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fully_connected']), biases['fully_connected'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, 0.5)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


'''
Question : how do you train model?
'''

# image is 32x32x3
x = tf.placeholder(tf.float32, [None, 32,32,3])
# unique labels: 43
y = tf.placeholder(tf.float32, (None, n_classes))

fc2 = LeNet(x)
# loss function
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
# optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss_op)
# accuracy
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y,1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# hyper parameters
EPOCHS = 20
BATCH_SIZE = 50

def eval_data(x_data,y_data):
    steps_per_epoch = x_data.shape[0] // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        start = step * BATCH_SIZE
        end = min(start + BATCH_SIZE, x_data.shape[0])
        batch_x = x_data[start:end]
        batch_y = y_data[start:end]
        #g1, g2, g3, g4 = sess.run([fc2,tf.argmax(fc2,1), tf.argmax(y,1), tf.equal(tf.argmax(fc2,1), tf.argmax(y,1))], feed_dict={x:batch_x, y:batch_y})
        #print("g1:", g1, "g2:", g2, "g3:", g3, "g4:", g4)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])

    return total_loss / num_examples, total_acc / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # shuffle data:
    x_train, x_validate, y_train, y_validate = shuffle()
    steps_per_epoch = x_train.shape[0] // BATCH_SIZE

    for i in range(EPOCHS):
        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end = min(start + BATCH_SIZE, x_train.shape[0])
            batch_x = x_train[start:end]
            batch_y = y_train[start:end]
            loss = sess.run(train_op, feed_dict={x:batch_x, y:batch_y})

        val_loss, val_acc = eval_data(x_validate, y_validate)
        print("EPOCH {}...".format(i+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.8f}".format(val_acc))
        print()

        # shuffle data:
        x_train, x_validate, y_train, y_validate = shuffle()

    print("running on test")
    test_loss, test_acc = eval_data(test_features, test_labels)
    print("test loss", test_loss, "test accuracy", test_acc)