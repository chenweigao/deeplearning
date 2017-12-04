## What I did this week are as follows:
- Paper Research
- Pantent Edit
- Code about deep learning
- Submit an expense account / apply for reimbursement

### Abstract
At the start of this weekday(Day01 - Day02),I read a paper named "You Only Look Once"(YOLO) about Object Detection algorism,on Day03,there were some courses,I did the code work in the evening,Day04 is the meeting,today is Day05-Day07.

### Introduction
Human glance at an image and instantly know what objects are in the image,where they are,and how they interact.Fast,accurate and excellent performance algorithms would allow computer to do the same thing as human.What I researched is the starting of this:Object Detection.There is an idea that we can realize it with the deeplearning tools such as CNN,RNN and so on.
Current research in our laboratory is OPENCV,the principal person is Han Yishen and Wangzhuo,they use the classical ways which emphasis on strategy rather than machine learning.There is great potential in this research.
Second,one of the most important work is to release the classical NETWORK such as VGG-Net,GooLeNet,so I did this work.
Third,I learned something about proc file system.

### Object Detection
I will implement this model in three weeks on Python if everything goes well.At the start of this work,I need to know some details about OpenCV.

There is a good algorithm named YOLO, using this at an image to predict what objects are present and where they are.It divides the image into an even grid and simultaneously predicts bounding boxes, confidence in those boxes, and class probabilities. These predictions are encoded as an S * S * (B *5 + C) tensor.

[YOLO](C:\Users\Administrator\Pictures\Saved Pictures\YOLO.png)![YOLO](C:\Users\Administrator\Pictures\Saved Pictures\YOLO.png)

### Capsule
The capsule is Hinton`s newly paper which about his idea, I read this paper and make some code to implemented it by Python.

```python
import tensorflow as tf
import numpy as np

M = np.array([
    [1, -1, 0],
    [-1, 2, 1],
    [0, 2, -2]
])

filter_weight = tf.get_variable('weights', shape=[2, 2, 1, 1], initializer=tf.constant_initializer(
    [[1, -1], [0, 2]]
))
biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1))

M = np.array(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)

x = tf.placeholder(tf.float32, [1, None, None, 1])

conv = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding='SAME')
bias = tf.nn.bias_add(conv, biases)

pool = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convluted_M = sess.run(bias, feed_dict={x: M})
    pooled_M = sess.run(pool, feed_dict={x: M})

    print('convluted_M:\n', convluted_M)
    print('pooled_M:\n', pooled_M)
```

It is a simple example about capsule, an in-depth study is essential in future.  

 ###  Argparse

In python, argparse is an useful function addition to call the  command parameter. In the project of deeplearning, we can use it to adjust parameter easily. To use it:

```python
import argparse
import os
import sys

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of images to process in a batch')

parser.add_argument('--data_dir', type=str, default='./mnist_data',
                    help='Path to the MNIST data directory.')

parser.add_argument('--model_dir', type=str, default='./mnist_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=40,
                    help='Number of epochs to train.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

```

First, we use the import to initialize, then we call the *argparse.ArgumentParser()* to generate a *parser*, the is the command parameter, it is really a good trick for coding.

###  Records

In deep learning, we usually celebrate a lot of data, which always to be format firstly. TFRecords is a tools to help us format data. It is an example: 

```python
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(dataset, name, directory):
  """Converts a dataset to TFRecords."""
  images = dataset.images
  labels = dataset.labels
  num_examples = dataset.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  # Get the data.
  datasets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(datasets.train, 'train', FLAGS.directory)
  convert_to(datasets.validation, 'validation', FLAGS.directory)
  convert_to(datasets.test, 'test', FLAGS.directory)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

### Train Models

There is a good news, I realized that official model is import, so I took 6 * 3 hours(totally 18h) to train a more accurate model, until  now, I get a model which accuracy is up to 99.8% in MNIST, the net contains  2 convolution layers and 2 pool layers, in addition, I added a dense and a dropout layers which returns more 41% performance in the end. As long as logits and softmax, I selected logits layers to output the final results, which also returns more 12% accuracy than softmax. It turns out that in 3-10 labels classify problem, softmax is not necessary.

I list my models for your reference:

```python
# Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=inputs,
      filters=32,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,
                                  data_format=data_format)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding='same',
      activation=tf.nn.relu,
      data_format=data_format)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,
                                  data_format=data_format)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                          activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
```

The difficult of this code lies in the channel`s value, and a trick in pool layer 2, a reshape, is used to prepare for next dense.

I think it`s really really a great code skills!!

### TF function

- **tf.argmax()**

  To use it, look the API first:

  ```
  tf.argmax(input, axis=None, name=None, dimension=None)

  Returns the index with the largest value across axes of a tensor.
  Args:
  input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half.
  axis: A Tensor. Must be one of the following types: int32, int64. int32, 0 <= axis < rank(input). Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0.
  name: A name for the operation (optional).
  Returns:
  A Tensor of type int64.
  ```

  **axis** is an important parameter:

  ```python
   predictions = {
          'classes': tf.argmax(input=logits, axis=1),
          #axis = 0 returns max at col
          #axis = 1 returns max at row
          'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
      }
  ```

  In this example, last return is [batch_size, 10], so it`s the max value in a row, axis=1.

- **reshape()**

  The reshape function is easy to understand, but there is a special example:

  `reshape(tensor, [-1, x]) and reshape(tensor, [x, -1])`

  If `shape` is 1-D or higher, then the operation returns a tensor with shape`shape` filled with the values of `tensor`. In this case, the number of elements implied by `shape` must be the same as the number of elements in `tensor`.

  In a word, -1 can be any value if you give a parameter, it automatically returns the unknown columns or rows.   

- **assert()**

   This statement helps coders find bugs more quickly and with less pain. There is an example:

  ```python
      assert (tf.gfile.Exists(train_file) and tf.gfile.Exists(test_file)), (
          'Run convert_to_records.py first to convert the MNIST data to TFRecord '
          'file format.')
  ```

  If it goes wrong, return Run convert_to_records.py first to convert the MNIST data to TFRecord file format.

### Conclusion

Fast YOLO is the fastest general-purpose object detector in the literature and pushes the state-of-the-art in real-time object detection, in future research, It will be a main work for me to implement it by python.

In the areas of deep learning, to release the classic models and improve their accuracy.

In the pages, I will read the document about Jekyll. For more information at GitHub:[GitHub](https://github.com/chenweigao/deeplearning)



