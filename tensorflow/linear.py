import tensorflow as tf

x = tf.Variable(5)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    output = sess.run(init)
    print(output)
    output = sess.run(x)
    print(output)

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(weights)
    print(output)
