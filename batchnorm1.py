import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

Sizedata = 10000
mean = np.array([2, -2])
cov = np.array([[2, 0], [0, 2]])
data_train = np.random.multivariate_normal(mean, cov, Sizedata)
data_test = np.random.multivariate_normal(mean, cov, Sizedata)


tf.reset_default_graph()

is_training = tf.placeholder_with_default(False, (), 'is_training')
x = tf.placeholder(tf.float32, [None, 2], 'x')
y = tf.layers.batch_normalization(x, training=is_training)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	z = tf.identity(y)

with tf.variable_scope("", reuse=True):
	meanr=tf.get_variable('batch_normalization/moving_mean')
	varr= tf.get_variable('batch_normalization/moving_variance')

init = tf.global_variables_initializer()

errorVar=[]
errormean=[]
with tf.Session() as sess:
	sess.run(init)
	for i in range(2000):
		idx = np.random.randint(Sizedata, size=16) # our batch size is 16
		zn, meanc, varc = sess.run([z, meanr, varr], feed_dict={x: data_train[idx,:], is_training: True})
		errorVar += [np.linalg.norm(mean-meanc)]
		errormean += [np.linalg.norm(varc-[cov[0][0],cov[1][1]])]


	# finished training
	normalized_test = np.empty((0, 2))
	for i in range(data_test.shape[0]):
		zn = sess.run(z, feed_dict={x: np.expand_dims(data_train[i,:], axis=0), is_training: False})
		normalized_test  = np.append(normalized_test , zn, axis=0)





ax1=plt.subplot(1, 3, 1)
ax1.scatter(data_train[:,0], data_train[:,1], c='red', s=30, label='train',  alpha=0.3, edgecolors='none')
ax1.scatter(data_test[:,0], data_test[:,1], c='blue', s=30, label='test',  alpha=0.3, edgecolors='none')
ax1.axis('equal')
ax1.legend()
ax1.grid(True)
ax1.set_title("input data")
ax1.axhline(0, color='black')
ax1.axvline(0, color='black')



ax2=plt.subplot(1, 3, 2)

ax2.scatter(normalized_test[:, 0], normalized_test[:, 1], c='blue', s=30, label='normalized', alpha=0.3, edgecolors='none')
ax2.grid(True)
ax2.set_title("normalized test data")
ax2.axhline(0, color='black')
ax2.axvline(0, color='black')


ax3=plt.subplot(1, 3, 3)
ax3.plot(errorVar,label='L2 norm of variance errors')
ax3.plot(errormean,label='L2 norm of mean errors')
ax3.set_title("statistics estimate versus iteration")
ax3.legend()
plt.show()
