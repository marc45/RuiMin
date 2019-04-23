import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

PI = 3.1415926535898
min_, max_ = -5, 5


class rbf_bp:
    # 对输入值进行径向基向量计算
    def kernel_(self, x_):
        # 函数：两点之间的欧式距离
        self.distant_ = lambda x1, x2: np.sqrt(np.sum(np.square(x1 - x2)))
        # 函数：高斯核
        # self.Gaussian = lambda x:np.exp(-np.power(x/self.gamma,2))  
        self.Gaussian = lambda x: x ** self.gamma
        mount_ = x_.shape[0]
        x_dis = np.zeros((mount_, self.num_))  # 中间矩阵:存储两点之间的距离
        matrix_ = np.zeros((mount_, self.num_))  # 距离，进行高斯核变换
        for i in range(mount_):
            for j in range(self.num_):
                x_dis[i, j] = self.distant_(x_[i], self.x_nodes[j])
                matrix_[i, j] = self.Gaussian(x_dis[i, j])
        return matrix_

    def __init__(self, x_nodes, y_nodes, gamma):
        # 节点的x坐标值
        self.x_nodes = x_nodes
        # 高斯系数
        self.gamma = gamma
        self.num_ = len(y_nodes)  # 节点数
        matrix_ = self.kernel_(x_nodes)
        # 计算初始化权重weights_
        weights_ = np.dot(np.linalg.pinv(matrix_), y_nodes.copy())
        # 定义一个两层的网络，第1层为高斯核函数节点的输出，第2层为回归的值
        self.x_ = tf.placeholder(tf.float32, shape=(None, x_nodes.shape[0]), name="x_")
        self.y_ = tf.placeholder(tf.float32, shape=(None), name="y_")
        weights_ = weights_[:, np.newaxis]
        self.weights = tf.Variable(weights_, name="weights", dtype=tf.float32)
        self.biaes = tf.Variable(0.0, name="biaes", dtype=tf.float32)
        self.predict_ = tf.matmul(self.x_, self.weights) + self.biaes
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.predict_))
        self.err_rate = tf.reduce_mean(tf.abs((self.y_ - self.predict_) / self.y_))

    def train(self, x_train, y_train, x_test, y_test, batch_size, learn_rate, circles_):
        print(x_train.shape)
        x_train = self.kernel_(x_train)
        print(x_train.shape)
        x_test = self.kernel_(x_test)
        self.train_ = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        saver = tf.train.Saver()
        size_ = x_train.shape[0]  # 训练集的数量
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for step_ in range(circles_):  # 训练次数
                start = int((step_ * batch_size) % (size_ - 1))
                end = start + batch_size
                if end < (size_ - 1):
                    in_x = x_train[start:end, :]
                    in_y = y_train[start:end]
                else:
                    end_ = end % (size_ - 1)
                    in_x = np.concatenate((x_train[start:size_ - 1, :], x_train[0:end_, :]))
                    in_y = np.concatenate((y_train[start:size_ - 1], y_train[0:end_]))
                if step_ % 50 == 0:
                    print("第", step_, "次迭代")
                    print("test_错误率：", sess.run(self.err_rate, feed_dict={self.x_: x_test, self.y_: y_test}))
                    print("train_错误率：", sess.run(self.err_rate, feed_dict={self.x_: in_x, self.y_: in_y}))
                    # print "predict:",sess.run(self.predict_, feed_dict={self.x_:in_x,self.y_:in_y})
                sess.run(self.train_, feed_dict={self.x_: in_x, self.y_: in_y})
            saver.save(sess, "Model/model.ckpt")

    def predict(self, x_data, y_data):
        x_data = self.kernel_(x_data)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "Model/model.ckpt")
            prediction = sess.run(self.predict_, feed_dict={self.x_: x_data, self.y_: y_data})
        return prediction


def gen_y(x):
        y = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            x_ = x[i]
            y[i] = np.sin(x_)
        return y


def main():
        filename = '../Datasets/maweiweather.csv'
        data = pd.read_csv(filename).drop(['日期'], axis=1)

        X = np.array(data.drop(['成品率'], axis=1))
        y = np.array(data['成品率'])
        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)

        min_max_scaler = preprocessing.MinMaxScaler()
        X = min_max_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # 生成100个等差分布的节点
        x_nodes = np.linspace(min_, max_, 50).reshape((50, 1))
        y_nodes = gen_y(x_nodes)
        rbf_ = rbf_bp(x_nodes, y_nodes, 1.0)
        rbf_.train(X_train, y_train, X_test, y_test, 500, 0.0001, 1000)
        pp = rbf_.predict(X_test[0:20], y_test[0:20])
        print(pp)
        print(y_test)


if __name__ == "__main__":
    main()
