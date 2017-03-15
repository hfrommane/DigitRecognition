import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    cross_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape, name):
    with tf.name_scope(name + '_weight'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape, name):
    with tf.name_scope(name + '_bias'):
        return tf.Variable(tf.constant(0.1, shape=shape))


def con2d(x, w, name):
    # 我们的卷积使用1步长（stride size），0边距（padding size）的模板
    # 保证输出和输入是同一个大小
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    with tf.name_scope(name):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, name):
    # 我们的池化用简单传统的2x2大小的模板做max pooling
    # 缩小为原来的1/4
    # stride [1, x_movement, y_movement, 1]
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

'''
第一层卷积
现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
卷积在每个5x5的patch中算出32个特征。
卷积的权重张量形状是[5, 5, 1, 32]，
前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
而对于每一个输出通道都有一个对应的偏置量。
'''
w_conv1 = weight_variable([5, 5, 1, 32], 'c1')
b_conv1 = bias_variable([32], 'c1')
# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
# 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(xs, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(con2d(x_image, w_conv1, 'conv1') + b_conv1)
h_pool1 = max_pool_2x2(h_conv1, 'pool1')

'''
第二层卷积
第二层中，每个5x5的patch会得到64个特征。
'''
w_conv2 = weight_variable([5, 5, 32, 64], 'c2')
b_conv2 = bias_variable([64], 'c2')
h_conv2 = tf.nn.relu(con2d(h_pool1, w_conv2, 'conv2') + b_conv2)
h_pool2 = max_pool_2x2(h_conv2, 'pool2')

'''
fc1 layer
现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
'''
w_fc1 = weight_variable([7 * 7 * 64, 1024], 'f1')
b_fc2 = bias_variable([1024], 'f1')
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
with tf.name_scope('fc1'):
    fc1 = tf.matmul(h_pool2_flat, w_fc1)
h_fc1 = tf.nn.relu(fc1 + b_fc2)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
fc2 layer
输出层，我们添加一个softmax层
'''
w_fc2 = weight_variable([1024, 10], 'f2')
b_fc2 = bias_variable([10], 'f2')
with tf.name_scope('fc2'):
    fc2 = tf.matmul(h_fc1_drop, w_fc2)
h_fc2 = fc2 + b_fc2
prediction = tf.nn.softmax(h_fc2)

with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.scalar_summary('loss', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('logs/', sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(1500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        writer.add_summary(result, i)
        test_batch_xs, test_batch_ys = mnist.test.next_batch(200)
        print(compute_accuracy(test_batch_xs, test_batch_ys))
