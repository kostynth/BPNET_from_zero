import numpy as np
import matplotlib.pyplot as plt

def generate_data(n: int) -> np.ndarray: # 生成数据：
    x = np.linspace(0, 1, n)  # 生成从0到1的等间距的n个点
    x = x.reshape(len(x), 1)  # 将x的形状重塑为(n, 1)
    y = np.sin(2 * np.pi * x)  # 计算x的sin值
    return x, y  # 返回生成的数据

class NN:
    def __init__(self, n_input: int, n_hidden: int, n_output: int, num_layers: int) -> None:
        """
        n_input: 输入的维度
        n_hidden: 隐藏层神经元的数量
        n_output: 输出的维度
        num_layers: 隐藏层的数量
        """
        self.n_input = n_input  # 输入层的维度
        self.n_hidden = n_hidden  # 隐藏层的神经元数量
        self.n_output = n_output  # 输出层的维度
        self.num_layers = num_layers  # 隐藏层的数量
        self.w = []  # 权重列表
        self.b = []  # 偏置列表
        self.gradw = []  # 权重梯度列表
        self.gradb = []  # 偏置梯度列表
        self.deltas = []  # 各层的误差
        # 初始化权重
        self.w.append(np.random.randn(n_input, n_hidden))  # 输入层到第一个隐藏层的权重
        for i in range(num_layers - 1):
            self.w.append(np.random.randn(n_hidden, n_hidden))  # 隐藏层到隐藏层的权重
        self.w.append(np.random.randn(n_hidden, n_output))  # 最后一层隐藏层到输出层的权重
        # 初始化偏置
        self.b.append(np.zeros((1, n_hidden)))  # 第一个隐藏层的偏置
        for i in range(num_layers - 1):
            self.b.append(np.zeros((1, n_hidden)))  # 隐藏层的偏置
        self.b.append(np.zeros((1, n_output)))  # 输出层的偏置

    def activation(self, x: np.ndarray) -> np.ndarray: # 激活函数：
        return np.tanh(x)  # 使用tanh作为激活函数

    def activation_deriv(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.power(self.activation(x), 2)  # 激活函数的导数

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = []  # 存储每层的加权和
        self.a = []  # 存储每层的激活值
        self.z.append(x)  # 输入层的加权和
        self.a.append(x)  # 输入层的激活值

        for i in range(self.num_layers):
            self.z.append(np.dot(self.a[i], self.w[i]) + self.b[i])  # 计算每层的加权和
            self.a.append(self.activation(self.z[i + 1]))  # 计算每层的激活值
        return self.a[-1]  # 返回输出层的激活值

    def mse(self, x: np.ndarray, y: np.ndarray) -> float: # MSE 损失函数
        return np.mean(np.power(self.predict(x) - y, 2))  # 计算均方误差

    def backward(self, x: np.ndarray, y: np.ndarray) -> None:

        self.gradw = []  # 权重梯度列表
        self.gradb = []  # 偏置梯度列表
        self.deltas = []  # 各层的误差

        self.deltas.append(self.a[-1] - y)  # 计算输出层的误差

        self.gradw.append(np.dot(self.a[-2].T, self.deltas[-1]))  # 计算输出层的权重梯度
        self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))  # 计算输出层的偏置梯度

        for i in range(self.num_layers - 1, 0, -1):
            self.deltas.append(np.dot(self.deltas[-1], self.w[i].T) * self.activation_deriv(self.z[i]))  # 计算隐含层的误差
            self.gradw.append(np.dot(self.a[i - 1].T, self.deltas[-1]))  # 计算隐含层的权重梯度
            self.gradb.append(np.sum(self.deltas[-1], axis=0, keepdims=True))  # 计算隐含层的偏置梯度

        self.gradw.reverse()  # 反转梯度列表
        self.gradb.reverse()  # 反转梯度列表
        self.deltas.reverse()  # 反转误差列表

    def update(self, lr: float) -> None:
        for i in range(self.num_layers):
            self.w[i] -= lr * self.gradw[i]  # 更新权重
            self.b[i] -= lr * self.gradb[i]  # 更新偏置

    def train(self, x: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> None:
        for i in range(epochs):
            self.forward(x)  # 前向传播
            self.backward(x, y)  # 反向传播
            self.update(lr)  # 更新权重和偏置

            self.w.append(self.b)

            if i % 100 == 0:
                print(f'Epoch {i}: {self.mse(x, y)}')  # 打印每100次迭代的损失值


    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)  # 返回前向传播的结果


# 创建神经网络
x, y = generate_data(100)  # 生成100个数据点
nn = NN(1, 3, 1, 3)  # 创建一个具有1个输入层、3个隐藏层和1个输出层的神经网络

nn.train(x, y, 0.01, 10000)  # 训练神经网络，学习率为0.01，迭代10000次

y_pred = [np.mean(a) for a in nn.predict(x)]  # 预测结果
plt.scatter(x, y)  # 绘制真实数据点
plt.plot(x, y_pred)  # 绘制预测结果
plt.show()  # 显示图像