import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
def generate_data(n: int) -> np.ndarray: # Generate data:
    x = np.linspace(0, 1, n)  # Generate n equally spaced points from 0 to 1
    x = x.reshape(len(x), 1)  # Reshape x to (n, 1)
    y = np.sin(2 * np.pi * x)  # Compute sin of x
    return x, y  # Return generated data

class NN:
    def __init__(self, n_input: int, n_hidden: int, n_output: int, num_layers: int) -> None:
        """
        n_input: input dimension
        n_hidden: number of hidden layer neurons
        n_output: output dimension
        num_layers: number of hidden layers
        """
        self.n_input = n_input  # input layer dimension
        self.n_hidden = n_hidden  # number of hidden layer neurons
        self.n_output = n_output  # output layer dimension
        self.num_layers = num_layers  # number of hidden layers
        self.w = []  # weight list
        self.b = []  # bias list
        # Initialize weights
        self.w.append(np.random.randn(n_input, n_hidden))  # weights from input layer to first hidden layer
        for i in range(num_layers - 1):
            self.w.append(np.random.randn(n_hidden, n_hidden))  # weights from hidden layer to hidden layer
        self.w.append(np.random.randn(n_hidden, n_output))  # weights from last hidden layer to output layer
        # Initialize biases
        self.b.append(np.zeros((1, n_hidden)))  # bias of first hidden layer
        for i in range(num_layers - 1):
            self.b.append(np.zeros((1, n_hidden)))  # hidden layer bias
        self.b.append(np.zeros((1, n_output)))  # output layer bias

    def activation(self, x: np.ndarray) -> np.ndarray: # Activation function:
        return np.tanh(x)  # Use tanh as activation function



    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = []  # Store weighted sum for each layer
        self.a = []  # Store activation value for each layer
        self.z.append(x)  # weighted sum of input layer
        self.a.append(x)  # activation value of input layer

        for i in range(self.num_layers):
            self.z.append(np.dot(self.a[i], self.w[i]) + self.b[i])  # Compute weighted sum for each layer
            self.a.append(self.activation(self.z[i + 1]))  # Compute activation value for each layer
        return self.a[-1]  # Return activation value of output layer

    def mse(self, x: np.ndarray, y: np.ndarray) -> float: # MSE loss function
        return np.mean(np.power(self.predict(x) - y, 2))  # Compute mean squared error





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
                print(f'Epoch{j} : Loss {gBest}')

        # Присваиваем сети глобально лучшие координаты (веса)
        self.w = gBest_coords[0]
        self.b = gBest_coords[1]










    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)  # Return result of forward propagation


# Create neural network
x, y = generate_data(100)  # Generate 100 data points
nn = NN(1, 3, 1, 3)  # Create a neural network with 1 input layer, 3 hidden layers, and 1 output layer
nn.train(x, y, 100, 10)  # тренируем с размером роя в 50 частиц

y_pred = [np.mean(a) for a in nn.predict(x)]  # prediction results
plt.scatter(x, y)  # Plot true data points
plt.plot(x, y_pred)  # Plot prediction results
plt.show()  # Show image
plt.plot(x, y_pred)  # 绘制预测结果
plt.show()  # 显示图像
