#模型定义:LeNet5
#yang, 2020/2/6

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt


#------------------------------【MNIST数据集】---------------------------------

#内置数据集查看
#print(dir(tf.keras.datasets))

#1.加载网络数据
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1.加载本地数据
f = np.load("mnist.npz")
x_train, y_train = f['x_train'],f['y_train']
x_test, y_test = f['x_test'],f['y_test']
f.close()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2.MNIST数据集可视化
#image_index = 123
#print(y_train[image_index])     #查看随机一张图片的label
#plt.imshow(x_train[image_index], cmap='Greys')  #图片显示
#plt.show()

#3.数据集格式转换
def DataFormat(x):
    x = np.pad(x, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)  #将图片从28*28扩展成32*32
    x = x.astype('float32') #数据类型转换
    x /= 255  # 数据正则化
    x = x.reshape(x.shape[0], 32, 32, 1)  # 数据维度转换
    return x
# x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)  #将图片从28*28扩展成32*32
# x_train = x_train.astype('float32')  #数据类型转换
# x_train /= 255  #数据正则化
# x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)  #数据维度转换
x_train = DataFormat(x_train)
print(x_train.shape)


#------------------------------【LeNet模型】---------------------------------

#方法一：Model类方式
class LeNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #模型
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5, 5),
            padding='valid',
            activation=tf.nn.relu)

        self.pool_layer_1 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            padding='same')

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding='valid',
            activation=tf.nn.relu)

        self.pool_layer_2 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            padding='same')

        self.flatten = tf.keras.layers.Flatten()

        self.fc_layer_1 = tf.keras.layers.Dense(
            units=120,
            activation=tf.nn.relu)

        self.fc_layer_2 = tf.keras.layers.Dense(
            units=84,
            activation=tf.nn.relu)

        self.output_layer = tf.keras.layers.Dense(
            units=10,
            activation=tf.nn.relu)

    def call(self, inputs):
        x = self.conv_layer_1(inputs)
        x = self.pool_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.pool_layer_2(x)
        x = self.flatten(x)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        output = self.output_layer(x)

        return output

#model = LeNet()

#方法二：Sequential类方式
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu, input_shape=(32, 32, 1)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
])

#model.summary()


#------------------------------【训练】---------------------------------

#超参数设置
num_epochs = 1
batch_size = 64
learning_rate = 0.01

#优化器
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

#编译
model.compile(optimizer=adam_optimizer,
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

#训练
start_time = datetime.datetime.now()

model.fit(x=x_train,
          y=y_train,
          batch_size=batch_size,
          epochs=num_epochs)

endtime = datetime.datetime.now()

time_cost = endtime - start_time
print('time_cost = ', time_cost)

#保存/加载模型
model.save('lenet.h5')
#model = tf.keras.models.load_model('lenet.h5')


#------------------------------【评估】---------------------------------

x_test = DataFormat(x_test)
print(x_test.shape)

print(model.evaluate(x_test, y_test))

#------------------------------【预测】---------------------------------

image_index = 2333
print(x_test[image_index].shape)
plt.imshow(x_test[image_index].reshape(32, 32), cmap='Greys')
plt.show()

pred = model.predict(x_test[image_index].reshape(1, 32, 32, 1))
print(pred.argmax())