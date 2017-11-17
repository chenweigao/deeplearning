import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0))

x = tf.placeholder(tf.float32, shape=(None, 2), name="input")
y_ = tf.placeholder(tf.float32, shape=None, name='y_input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# sess = tf.Session()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)

# print(sess.run(y))
# print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(
    learning_rate=0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1) for (x1, x2) in X]]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training steps,cross entropy is %g"%(i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
