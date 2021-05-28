import tensorflow as tf
import forward
import os
import ImgHandle as IMG
import random

#定义每轮20个数据
BATCH_SIZE = 20
#正则化系数0.001
REGULARIZER = 0.001
#训练10000轮
STEPS = 10000
#滑动平均系数0.01
MOVING_AVERAGE_DECAY = 0.01
#保存训练好的模型的路径
MODEL_SAVE_PATH="./model/"
MODEL_NAME="train_model"
FILE_NAME="Classification.xlsx"

def backward(data, label):
    tf.compat.v1.disable_eager_execution()
    #x是样本
    x = tf.compat.v1.placeholder(tf.float32, shape = (None, forward.INPUT_NODE))

    #y_是标签，也就是正确结果
    y_ = tf.compat.v1.placeholder(tf.float32, shape = (None, forward.OUTPUT_NODE))

    #y是结果前向传播的输出值
    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    #损失函数的定义
    loss = cem + tf.add_n(tf.compat.v1.get_collection('losses'))

    #训练过程，目的是使损失函数最小
    train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(loss, global_step=global_step)

    ema = tf.compat.v1.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.compat.v1.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.compat.v1.train.Saver()

    #建立会话，初始化参数，运行训练过程
    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)

    #保存模型训练过程的记录点，如果训练结束，下次训练也能从记录点继续开始
        ckpt = tf.compat.v1.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

    #循环STEPS轮，将x,y_输入神经网络，每100轮打印loss值
        for i in range(STEPS):
            start = (i*BATCH_SIZE)%len(data)
            end = start+BATCH_SIZE
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: data[start:end], y_: label[start:end]})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    data, label = IMG.img_handle()
    for i in range(len(data)):
        x, y = random.randint(0, len(data)-1), random.randint(0, len(data)-1)
        temp_data = data[x]
        data[x] = data[y]
        data[y] = temp_data
        temp_label = label[x]
        label[x] = label[y]
        label[y] = temp_label
    print(len(data), len(label))
    backward(data, label)
