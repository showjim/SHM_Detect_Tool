import tensorflow as tf
a = tf.add(1,2).numpy()
print(a)
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy())