import tensorflow as tf

hello_constant = tf.constant('hello world')

sess = tf.Session()
output = sess.run(hello_constant)
print(output)

a = tf.constant(1234)
b = tf.constant([123, 456, 789])
c = tf.constant([[123, 456, 789], [222, 333, 444]])

output = sess.run(a)
print(output)

output = sess.run(b)
print(output)

output = sess.run(c)
print(output)

x = tf.placeholder(tf.string)

output = sess.run(x, feed_dict={x: "hello x"})
print(output)

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

output = sess.run(x, feed_dict={x: 'test', y: 123, z: 345})
print(output)

x = tf.add(5,2)
output = sess.run(x)
print(output)

x = tf.subtract(10,4)
y = tf.multiply(2,5)

output = sess.run(x)
print(output)

output = sess.run(y)
print(output)

x = tf.subtract(tf.constant(2.0),tf.cast(tf.constant(1),tf.float32))
output = sess.run(x)
print(output)