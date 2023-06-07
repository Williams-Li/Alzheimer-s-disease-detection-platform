import tensorflow as tf
from data_process import get_data, get_data_test

path_train = 'data/train/*/*.png'    # 训练集样本的路径
path_test = 'data/test/*/*.png'      # 测试集样本的路径

X_train, y_train = get_data(path_train)         # 训练集样本数据
X_test = get_data_test(path_test)               # 测试集样本自变量


class LRN(tf.keras.layers.Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius = 5
        self.bias = 2
        self.alpha = 1e-4
        self.beta = 0.75

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius,
                         bias=self.bias, alpha=self.alpha,
                         beta=self.beta)


# 搭建AlexNet
model = tf.keras.models.Sequential()

# 卷积C1
model.add(tf.keras.layers.Conv2D(     # 卷积操作
    filters=96,
    kernel_size=(11, 11),
    strides=(4, 4),
    activation='relu',
    input_shape=(227, 227, 1)
))
model.add(tf.keras.layers.MaxPool2D((3, 3), strides=2))    # 池化操作
model.add(LRN())     # 局部响应归一化

# 卷积C2
model.add(tf.keras.layers.Conv2D(     # 卷积操作
    filters=256,
    kernel_size=(5, 5),
    activation='relu',
    padding='same'
))
model.add(tf.keras.layers.MaxPool2D((3, 3), strides=2))    # 池化操作
model.add(LRN())     # 局部响应归一化

# 卷积C3
model.add(tf.keras.layers.Conv2D(     # 卷积操作
    filters=384,
    kernel_size=(3, 3),
    activation='relu',
    padding='same'
))

# 卷积C4
model.add(tf.keras.layers.Conv2D(     # 卷积操作
    filters=384,
    kernel_size=(3, 3),
    activation='relu'
))

# 卷积C5
model.add(tf.keras.layers.Conv2D(     # 卷积操作
    filters=256,
    kernel_size=(3, 3),
    activation='relu'
))

model.add(tf.keras.layers.MaxPool2D((3, 3), strides=2))    # 池化操作
model.add(tf.keras.layers.Flatten())     # 将上层网络的输出拉伸成一维结构

model.add(tf.keras.layers.Dense(4096, activation='relu'))   # 全连接层
model.add(tf.keras.layers.Dropout(0.5))                     # 抑制过拟合
model.add(tf.keras.layers.Dense(4096, activation='relu'))   # 全连接层
model.add(tf.keras.layers.Dropout(0.5))                     # 抑制过拟合
model.add(tf.keras.layers.Dense(2, activation='softmax'))   # 输出层

# 模型编译
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=40)      # 模型训练
# loss: 6.1412e-04 - accuracy: 0.9998 - val_loss: 0.0930 - val_accuracy: 0.9756

model.save('model_AlexNet.tf')    # 保存模型

# predicted = model.predict(X_test).argmax(axis=1)    # 模型预测
