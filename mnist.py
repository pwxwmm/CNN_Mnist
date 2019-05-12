def conv_model():
    """
    自定义的卷积网络结构
    :return: x, y_true, y_predict
    """
    # 1、准备数据占位符
    # x [None, 784]  y_true [None, 10]
    with tf.variable_scope("data"):

        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2、卷积层一 32个filter, 大小5*5,strides=1, padding=“SAME”

    with tf.variable_scope("conv1"):
        # 随机初始化这一层卷积权重 [5, 5, 1, 32], 偏置[32]
        w_conv1 = weight_variables([5, 5, 1, 32])

        b_conv1 = bias_variables([32])

        # 首先进行卷积计算
        # x [None, 784]--->[None, 28, 28, 1]  x_conv1 -->[None, 28, 28, 32]
        x_conv1_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # input-->4D
        x_conv1 = tf.nn.conv2d(x_conv1_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1

        # 进行激活函数计算
        #  x_relu1 -->[None, 28, 28, 32]
        x_relu1 = tf.nn.relu(x_conv1)

        # 进行池化层计算
        # 2*2, strides 2
        #  [None, 28, 28, 32]------>[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3、卷积层二 64个filter, 大小5*5,strides=1,padding=“SAME”
    # 输入：[None, 14, 14, 32]
    with tf.variable_scope("conv2"):
        # 每个filter带32张5*5的观察权重，一共有64个filter去观察
        # 随机初始化这一层卷积权重 [5, 5, 32, 64], 偏置[64]
        w_conv2 = weight_variables([5, 5, 32, 64])

        b_conv2 = bias_variables([64])

        # 首先进行卷积计算
        # x [None, 14, 14, 32]  x_conv2 -->[None, 14, 14, 64]
        # input-->4D
        x_conv2 = tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2

        # 进行激活函数计算
        #  x_relu1 -->[None, 28, 28, 32]
        x_relu2 = tf.nn.relu(x_conv2)

        # 进行池化层计算
        # 2*2, strides 2
        #  [None, 14, 14, 64]------>x_pool2[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4、全连接层输出
    # 每个样本输出类别的个数10个结果
    # 输入：x_poll2 = [None, 7, 7, 64]
    # 矩阵运算： [None, 7 * 7 * 64] * [7 * 7 * 64, 10] +[10] = [None, 10]
    with tf.variable_scope("fc"):
        # 确定全连接层权重和偏置
        w_fc = weight_variables([7 * 7 * 64, 10])

        b_fc = bias_variables([10])

        # 对上一层的输出结果的形状进行处理成2维形状
        x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行全连接层运算
        y_predict = tf.matmul(x_fc, w_fc) + b_fc

    return x, y_true, y_predict
    # 1、准备数据API
    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)

    # 2、定义模型,两个卷积层、一个全连接层
    x, y_true, y_predict = conv_model()

    # 3、softmax计算和损失计算
    with tf.variable_scope("softmax_loss"):

        # labels:真实值 [None, 10]  one_hot
        # logits:全脸层的输出[None,10]
        # 返回每个样本的损失组成的列表
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                      logits=y_predict))
    # 4、梯度下降损失优化
    with tf.variable_scope("optimizer"):
        # 学习率
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

    # 5、准确率计算
    with tf.variable_scope("accuracy"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量op
    init_op = tf.global_variables_initializer()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

tf.app.flags.DEFINE_integer("is_train", 1, "指定是否是训练模型，还是拿数据去预测")
FLAGS = tf.app.flags.FLAGS

# 定义两个专门初始化权重和偏置的函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1))
    return w


def bias_variables(shape):
    b = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=0.1))
    return b


def cnn_model():
    """
    自定义CNN 卷积模型
    第一层
    卷积：32个filter、大小5 * 5、strides = 1、padding = "SAME"
    激活：Relu
    池化：大小2x2、strides2
    第一层
    卷积：64个filter、大小5 * 5、strides = 1、padding = "SAME"
    激活：Relu
    池化：大小2x2、strides2
    全连接层: [7*7*64, 10] [10]
    :return:
    """
    # 1、准备数据的占位符，便于后面卷积计算
    # x [None, 784], y_true = [None, 10]
    with tf.variable_scope("x_data"):

        x = tf.placeholder(tf.float32, [None, 784], name="x")

        y_true = tf.placeholder(tf.int32, [None, 10], name="y_true")

    # 2、第一层
    # 卷积：32个filter、大小5 * 5、strides = 1、padding = "SAME"
    # 激活：Relu
    # 池化：大小2x2、strides2
    with tf.variable_scope("conv_1"):

        # 准备权重和偏置参数
        # 权重数量：[5, 5, 1 ,32]
        # 偏置数量：[32]
        w_conv1 = weight_variables([5, 5, 1, 32])
        b_conv1 = bias_variables([32])

        # 特征形状变成4维，用于卷积运算
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # 进行卷积,激活函数运算
        # [None, 28, 28, 1]--->[None, 28, 28, 32]
        # [None, 28, 28, 32]--->[None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape,
                                          w_conv1,
                                          strides=[1, 1, 1, 1],
                                          padding="SAME",
                                          name="conv1") + b_conv1,
                             name="relu1")
        # 进行池化层
        # [None, 28, 28, 32]--->[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool1")

    # 3、第二层
    # 卷积：64 个filter、大小5 * 5、strides = 1、padding = "SAME"
    # 激活：Relu
    # 池化：大小2x2、strides2
    with tf.variable_scope("conv_2"):

        # 确定权重、偏置形状
        # 权重数量：[5, 5, 32, 64]
        # 偏置数量：[64]
        w_conv2 = weight_variables([5, 5, 32, 64])
        b_conv2 = bias_variables([64])

        # 进行卷积、激活运算
        # 卷积：[None, 14, 14, 32]--- >[None, 14, 14, 64]
        # 激活：[None, 14, 14, 64]--- >[None, 14, 14, 64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1,
                                          w_conv2,
                                          strides=[1, 1, 1, 1],
                                          padding="SAME",
                                          name="conv2") + b_conv2,
                             name='relu2')

        # 进行池化运算
        # 池化：[None, 14, 14, 64]--- >[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding="SAME",
                                 name="pool2")

    # 4、全连接层
    # 全连接层: [7 * 7 * 64, 10][10]
    with tf.variable_scope("fc"):

        # 初始化权重，偏置
        w_fc = weight_variables([7 * 7 * 64, 10])
        b_fc = bias_variables([10])

        # 矩阵运算转换二维
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 全连接层矩阵运算
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def train():
    """
    卷积网络识别训练
    :return:
    """
    # 1、准备数据输入
    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)

    # 2、建立卷积网络模型
    # y_true :[None, 10]
    # y_predict :[None, 10]
    x, y_true, y_predict = cnn_model()

    # 3、根据输出结果与真是结果建立softmax、交叉熵损失计算
    with tf.variable_scope("softmax_cross"):
        # 先进性网络输出的值的概率计算softmax,在进行交叉熵损失计算
        all_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                           logits=y_predict,
                                                           name="compute_loss")
        # 求出平均损失
        loss = tf.reduce_mean(all_loss)

    # 4、定义梯度下降优化器进行优化
    with tf.variable_scope("GD"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5、求出每次训练的准确率为
    with tf.variable_scope("accuracy"):
        # 求出每个样本是否相等的一个列表
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        # 计算相等的样本的比例
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # (2)、收集要显示的变量
    # 先收集损失和准确率
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 初始化变量op
    init_op = tf.global_variables_initializer()

    # （3）、合并所有变量op
    merged = tf.summary.merge_all()

    # 创建模型保存和加载
    saver = tf.train.Saver()

    # 开启会话进行训练
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # （1）创建一个events文件实例
        file_writer = tf.summary.FileWriter("./tmp/summary/", graph=sess.graph)

        # 加载模型
        if os.path.exists("./tmp/modelckpt/checkpoint"):
            saver.restore(sess, "./tmp/modelckpt/fc_nn_model")

        if FLAGS.is_train == 1:

            # 循环步数去训练
            for i in range(1000):

                # 每批次给50个样本
                mnist_x, mnist_y = mnist.train.next_batch(50)

                _, loss_run, acc_run = sess.run([train_op, loss, accuracy],
                                                feed_dict={x: mnist_x, y_true: mnist_y})

                # 打印每部训练的效果
                print("第 %d 步的50个样本损失为：%f , 准确率为：%f" % (i, loss_run, acc_run))

                # 运行合变量op，写入事件文件当中
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})

                file_writer.add_summary(summary, i)

                if i % 100 == 0:
                    saver.save(sess, "./tmp/modelckpt/fc_nn_model")
        else:

            # 如果不是训练，我们就去进行预测测试集数据从样本
            for i in range(100):
                # 每次拿一个样本进行测试
                mnist_x, mnist_y = mnist.test.next_batch(1)

                print("第%d个样本的真实值为：%d, 模型预测结果为：%d" % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: mnist_x, y_true: mnist_y}), 1).eval()
                )
                      )

    return None


if __name__ == '__main__':
    train()
