###待修改
import tensorflow as tf


num_classes = 10  # 输出大小
input_size = 784  # 输入大小
hidden_units_size = 30  # 隐藏层节点数量
batch_size = 100
training_iterations = 10000

X = tf.placeholder(tf.float32, shape = [None, input_size])
Y = tf.placeholder(tf.float32, shape = [None, num_classes])

W1 = tf.Variable(tf.random_normal ([input_size, hidden_units_size], stddev = 0.1))
B1 = tf.Variable(tf.constant (0.1), [hidden_units_size])
W2 = tf.Variable(tf.random_normal ([hidden_units_size, num_classes], stddev = 0.1))
B2 = tf.Variable(tf.constant (0.1), [num_classes])

hidden_opt = tf.matmul(X, W1) + B1  # 输入层到隐藏层正向传播
hidden_opt = tf.nn.relu(hidden_opt)  # 激活函数，用于计算节点输出值
final_opt = tf.matmul(hidden_opt, W2) + B2  # 隐藏层到输出层正向传播


# 对输出层计算交叉熵损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=final_opt))
# 梯度下降算法，这里使用了反向传播算法用于修改权重，减小损失
opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()
# 计算准确率
correct_prediction =tf.equal (tf.argmax (Y, 1), tf.argmax(final_opt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session ()
sess.run (init)
for i in range (training_iterations) :
    batch = mnist.train.next_batch (batch_size)
    batch_input = batch[0]
    batch_labels = batch[1]
    # 训练
    training_loss = sess.run ([opt, loss], feed_dict = {X: batch_input, Y: batch_labels})
    if i % 1000 == 0 :
        train_accuracy = accuracy.eval (session = sess, feed_dict = {X: batch_input,Y: batch_labels})
        print ("step : %d, training accuracy = %g " % (i, train_accuracy))