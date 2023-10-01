import math
import random


def get_start_weight():  # генерируем веса для 15 входных нейронов
    input_neurons_weight = []
    for j in range(15):
        weight = []
        for i in range(5):
            weight.append(random.uniform(0, 0.5))
        input_neurons_weight.append(weight)
    print('Начльные веса входных нейронов')
    for i in input_neurons_weight:
        print(i)
    print()
    return input_neurons_weight


def get_first_hidden_weight():  # генерируем веса для 1 скрытого слоя, 5 нейронов
    hidden_neurons_weight = []
    for j in range(5):
        weight = []
        for i in range(5):
            weight.append(random.uniform(0, 0.5))
        hidden_neurons_weight.append(weight)
    print('Начльные веса на первом слое скрытых нейронов')
    for i in hidden_neurons_weight:
        print(i)
    print()
    return hidden_neurons_weight


def get_second_hidden_weight():  # генерируем веса для 2 скрытого слоя, 5 нейронов
    hidden_neurons_weight = []
    for i in range(5):
        hidden_neurons_weight.append(random.uniform(0, 0.5))
    print('Начльные веса на втром слое скрытых нейронов')
    for i in hidden_neurons_weight:
        print(i)
    print()
    return hidden_neurons_weight


def get_data_and_y():
    data = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]]

    y = [[0], [1], [0], [1], [0], [1]]
    return data, y


def singmoid(input_value):
    return 1 / (1 + math.e ** (-input_value))


def grad_singmoid(input_value):
    return input_value * (1 - input_value)


def leaky_relu(x):
    if x < 0:
        return 0.1
    elif x >= 0:
        return x


def grad_leaky_relu(x):
    if x < 0:
        return 0.01
    elif x >= 0:
        return 1


class Neural:
    def __init__(self, data, y, input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight,
                 learning_step, iteration):
        self.data = data
        self.y = y
        self.input_neurons_weight = input_neurons_weight
        self.first_hidden_neurons_weight = first_hidden_neurons_weight
        self.second_hidden_neurons_weight = second_hidden_neurons_weight
        self.learning_step = learning_step
        self.iteration = iteration

    def train(self):
        epoch = 0
        while epoch != self.iteration:
            self.direct_distribution()
            if epoch % 10 == 0:
                print()
                print(f'Эпоха - {epoch}')
                print(f'Входной слой веса {self.input_neurons_weight}')
                print(f'Первый скрытый слой веса {self.first_hidden_neurons_weight}')
                print(f'Второй скрытый слой веса {self.first_hidden_neurons_weight}')
                print()
            epoch += 1

    def direct_distribution(self):
        for d, dd in enumerate(self.data):  # проход по данным
            values_on_input_second_layer = [0] * len(self.second_hidden_neurons_weight)
            values_on_out_layer = 0
            for i, ii in enumerate(self.first_hidden_neurons_weight):
                for j, jj in enumerate(self.first_hidden_neurons_weight):
                    values_on_input_second_layer[j] += dd[j] * self.first_hidden_neurons_weight[j][i]
            values_on_input_second_layer = [singmoid(i) for i in values_on_input_second_layer]
            # print(f'Значения после второго слоя - {values_on_input_second_layer}')

            for j, jj in enumerate(self.second_hidden_neurons_weight):
                values_on_out_layer += values_on_input_second_layer[j] * self.second_hidden_neurons_weight[j]
            values_on_out_layer = singmoid(values_on_out_layer)
            # print(f'Значения на выходном слое слое - {values_on_out_layer}')
            # print()
            # обратное распрастранение ошибки
            # расчёт ошибки на всех уровнях
            answer_error = self.y[d][0] - values_on_out_layer
            # print('Ошибки сети', answer_error)

            second_layer_error = [0] * len(values_on_input_second_layer)
            # first_layer_error = [0] * len(values_on_input_first_layer)

            for j, jj in enumerate(self.second_hidden_neurons_weight):
                second_layer_error[j] += answer_error * self.second_hidden_neurons_weight[j]
            # print('Ошибки на втором скрытом слое', second_layer_error)

            # обновляем веса

            for i, ii in enumerate(self.input_neurons_weight[0]):
                for j, jj in enumerate(self.input_neurons_weight):
                    self.input_neurons_weight[j][i] += \
                        second_layer_error[i] * grad_singmoid(values_on_input_second_layer[i]) * \
                        data[d][i] * self.learning_step
            # print(f'Новые веса для первого слоя')
            # for i in input_neurons_weight:
            #     print(i)
            # print()

            for j, jj in enumerate(self.second_hidden_neurons_weight):
                self.second_hidden_neurons_weight[j] += \
                    answer_error * grad_singmoid(values_on_out_layer) * \
                    values_on_input_second_layer[j] * self.learning_step
            # print(f'Новые веса для выходного слоя')
            # for i in second_hidden_neurons_weight:
            #     print(i)
            # print()

    def predict(self, new_data):

        values_on_input_second_layer = [0] * len(self.second_hidden_neurons_weight)
        values_on_out_layer = 0
        for i, ii in enumerate(self.first_hidden_neurons_weight):
            for j, jj in enumerate(self.first_hidden_neurons_weight):
                values_on_input_second_layer[j] += new_data[j] * self.first_hidden_neurons_weight[j][i]
        values_on_input_second_layer = [singmoid(i) for i in values_on_input_second_layer]

        for j, jj in enumerate(self.second_hidden_neurons_weight):
            values_on_out_layer += values_on_input_second_layer[j] * self.second_hidden_neurons_weight[j]
        values_on_out_layer = singmoid(values_on_out_layer)
        print(f'На вход подалось {new_data}')
        print(f'Ответ нейросети: {values_on_out_layer}')
        print()


if __name__ == '__main__':
    input_neurons_weight = get_start_weight()
    first_hidden_neurons_weight = get_first_hidden_weight()
    second_hidden_neurons_weight = get_second_hidden_weight()
    data, y = get_data_and_y()

    multi_layered_thing = Neural(data, y, input_neurons_weight, first_hidden_neurons_weight,
                                 second_hidden_neurons_weight,
                                 learning_step=0.1, iteration=10000)

    multi_layered_thing.train()

    new_data = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    print('0')
    multi_layered_thing.predict(new_data)

    print('1')
    new_data = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    multi_layered_thing.predict(new_data)
    print('1')
    new_data = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    multi_layered_thing.predict(new_data)
    print('1')
    new_data = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    multi_layered_thing.predict(new_data)
    print('0')
    new_data = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    multi_layered_thing.predict(new_data)
