import tensorflow as tf 
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32,shape = (None,2),name = 'x-input')
y_ = tf.placeholder(tf.float32,shape = (None,1),name = 'y-input')

w1 = tf.Variable(tf.random_normal([2,1],stddev=1.0,seed = 1.0))
y = tf.matmul(x,w1)

loss_less = 10
loss_more = 1
# loss = tf.reduce_mean(tf.where(tf.greater(y,y_),(y-y_)*loss_more,(y_-y)*loss_less))
loss = tf.reduce_mean(tf.square(y_-y))
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.001,global_step, 100 , 0.96 ,staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    Steps = 5000
    for i in range(Steps):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end] , y_:Y[start:end]})
        print(sess.run(w1))
