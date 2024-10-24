import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
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





    def train(self,x, y, swarm_count=10, max_speed=10, inertion_coef=0.9,c1=2, c2=2,  epochs=10000) -> None:
        swarm = []
        swarm.append(np.array([deepcopy(self.w), deepcopy(self.b)], dtype="object"))

        # Инициируем частицы точно также, как начальные веса
        for i in range(1, swarm_count):
            w = []
            b = []

            w.append(np.random.randn(self.n_input, self.n_hidden))
            for i in range(self.num_layers - 1):
                w.append(np.random.randn(self.n_hidden, self.n_hidden))
            w.append(np.random.randn(self.n_hidden, self.n_output))

            b.append(np.zeros((1, self.n_hidden)))
            for i in range(self.num_layers - 1):
                b.append(np.zeros((1, self.n_hidden)))
            b.append(np.zeros((1, self.n_output)))
            swarm.append(np.array([w, b],dtype="object"))
        gBest = self.mse(x, y) #значение loss-функции в ГЛОБАЛЬНО лучшей точке (сейчас - текущие веса сети)
        gBest_coords = np.array([deepcopy(self.w), deepcopy(self.b)], dtype="object") # значение (координаты) ГЛОБАЛЬНО лучшей точки
        pBest_coords = [] # архив ЛОКАЛЬНО лучших координат для КАЖДОЙ точки
        pBest_coords.append(np.array([deepcopy(self.w), deepcopy(self.b)], dtype="object")) # добавляем в архив первой точки её саму

        # В начале для каждой точки лучшие ЛОКАЛЬНЫЕ координаты - сами начальные координаты
        for i in range(1, swarm_count):
            pBest_coords.append(swarm[i])


        pBest = np.zeros(swarm_count)

        # Добавляем  исходные значения loss - функции для каждой точки
        for i in range(swarm_count):

            self.w = swarm[i][0]
            self.b = swarm[i][1]
            pBest[i] = self.mse(x, y)
            if pBest[i] < gBest:
                gBest = pBest[i]
                gBest_ind = i
                gBest_coords  = swarm[i]

        v_i = []

        # Инициируем  начальные вектора скорости для  каждой точки (случайные)
        for i in range( swarm_count): # Array with speed
            w = []
            b = []

            w.append(np.random.randn(self.n_input, self.n_hidden))
            for i in range(self.num_layers - 1):
                w.append(np.random.randn(self.n_hidden, self.n_hidden))
            w.append(np.random.randn(self.n_hidden, self.n_output))

            b.append(np.zeros((1, self.n_hidden)))
            for i in range(self.num_layers - 1):
                b.append(np.zeros((1, self.n_hidden)))
            b.append(np.zeros((1, self.n_output)))
            v_i.append(np.array([w, b],dtype="object"))


        # Алгоритм роя частиц
        for j in range(epochs):
            for i in range(swarm_count):
                # Для каждой частицы пересчитываем для неё скорость
                v_now = (inertion_coef if i < 500 else 0.6) *v_i[i] + c1* np.random.rand() * (pBest_coords[i] - swarm[i]) + c2 * np.random.rand() * (gBest_coords - swarm[i])
                # Сдвигаем частицу по полученному вектору
                swarm[i] = swarm[i] + v_now

                self.w = swarm[i][0]
                self.b = swarm[i][1]
                #  Считаем ошибку в новой точке
                err = self.mse(x, y)

                # Если новая точка оптимальнее локально/глобально - модифицируем соответствующие значения
                if err < pBest[i]:
                    pBest[i] = err
                    pBest_coords[i][0]  = swarm[i][0]
                    pBest_coords[i][1] = swarm[i][1]
                if err < gBest:
                    gBest = err
                    gBest_coords[0] = swarm[i][0]
                    gBest_coords[1] = swarm[i][1]
            if j % 100 == 0:
                #print(f'Epoch{j} : Loss {gBest}')
                print(f'{gBest},')
        # Присваиваем сети глобально лучшие координаты (веса)
        self.w = gBest_coords[0]
        self.b = gBest_coords[1]










    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)  # 返回前向传播的结果


# 创建神经网络
x, y = generate_data(100)  # 生成100个数据点
nn = NN(1, 3, 1, 3)  # 创建一个具有1个输入层、3个隐藏层和1个输出层的神经网络
nn.train(x, y, 100, 10)  # тренируем с размером роя в 50 частиц

y_pred = [np.mean(a) for a in nn.predict(x)]  # 预测结果
plt.scatter(x, y)  # 绘制真实数据点
plt.plot(x, y_pred)  # 绘制预测结果
plt.show()  # 显示图像