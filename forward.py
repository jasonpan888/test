import tensorflow as tf

INPUT_NODE = 256    # 输入层
OUTPUT_NODE = 10    # 输出层
LAYER1_NODE = 100   # 隐藏层
LAYER2_NODE = 100   # 隐藏层2

#获取节点权重
def get_weight(shape, regularizer):
    w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape,stddev=0.1))
    if regularizer is not None:
        tf.compat.v1.add_to_collection('losses',  tf.keras.regularizers.l2(regularizer)(w))
    return w

#获取偏置值
def get_bias(shape):
    b = tf.compat.v1.Variable(tf.zeros(shape))
    return b

#前向传播
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, LAYER2_NODE], regularizer)
    b2 = get_bias([LAYER2_NODE])
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)

    w3 = get_weight([LAYER2_NODE, OUTPUT_NODE], regularizer)
    b3 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y2, w3) + b3
    return y
