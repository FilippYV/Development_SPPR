import math
import random
import cv2

def find_answers(answers):
    maximum = -1
    index = -1
    for i, ii in enumerate(answers):
        if ii > maximum:
            maximum = ii
            index = i
    return index

def convert(i):
    data = []
    for i, ii in enumerate(cv2.imread(f"../static/photo/{i}.png", 0)):
        for j, jj in enumerate(ii):
            if jj == 255:
                data.append(0)
            else:
                data.append(1)
    return data

def convert_data():
    massive = []
    data = []
    for k in range(1,16):
        for i, ii in enumerate(cv2.imread(f"../static/photo/{k}.png", 0)):
            for j, jj in enumerate(ii):
                if jj == 255:
                    data.append(0.01)
                else:
                    data.append(1)
        massive.append(data)
    y = []
    for i in range(3):
        for j in range(5):
            if i == 0:
                y.append([1, 0, 0])
            elif i == 1:
                y.append([0, 1, 0])
            elif i == 2:
                y.append([0, 0, 1])

    return massive, y

# def get_x_y():
#     # x = [[0.25, 0.02]]
#     # y = [[0, 1]]
#     x = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
#          [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
#          # [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
#          # [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
#          [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
#          # [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
#          # [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#          [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
#          # [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#          [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]]
#     y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#          # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#          # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#          # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#          # [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
#     return x, y


def get_weight(number_of_neurons_per_layer, number_of_neurons_on_next_per_layer, random_from, random_to):
    # генерируем веса для нейронов
    hidden_neurons_weight = []
    for j in range(number_of_neurons_per_layer):
        weight = []
        for i in range(number_of_neurons_on_next_per_layer):
            weight.append(random.uniform(random_from, random_to))
        hidden_neurons_weight.append(weight)
    print('Начльные веса на втром слое скрытых нейронов')
    for i in hidden_neurons_weight:
        print(i)
    print()
    return hidden_neurons_weight


def sigmoid(input_value):
    return 1 / (1 + math.e ** (-input_value))


def grad_sigmoid(input_value):
    return input_value * (1 - input_value)


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
        while epoch != self.iteration + 1:
            self.direct_distribution()
            if epoch % 1000 == 0:
                print()
                print(f'Эпоха - {epoch}')
                print(f'Входной слой веса {self.input_neurons_weight}')
                print(f'Второй скрытый слой веса {self.second_hidden_neurons_weight}')
                print()
            epoch += 1

    def calk_value_on_layer(self, weight_neurons, value):
        values_on_layer = [0] * len(weight_neurons[0])
        for i, ii in enumerate(weight_neurons[0]):
            for j, jj in enumerate(weight_neurons):
                values_on_layer[i] += \
                    value[j] * weight_neurons[j][i]
        values_on_layer = [sigmoid(i) for i in values_on_layer]
        # print(f'Значения после слоя - {values_on_layer}')
        return values_on_layer

    def direct_distribution(self):
        for d, dd in enumerate(self.data):  # проход по данным
            values_on_input_first_layer = [0] * len(self.input_neurons_weight[0])
            values_on_input_second_layer = [0] * len(self.first_hidden_neurons_weight[0])
            values_on_out_layer = [0] * len(self.second_hidden_neurons_weight[0])
            for i, ii in enumerate(self.input_neurons_weight[0]):
                for j, jj in enumerate(self.input_neurons_weight):
                    values_on_input_first_layer[i] += \
                        dd[j] * self.input_neurons_weight[j][i]
            values_on_input_first_layer = [sigmoid(i) for i in values_on_input_first_layer]
            # print(f'Значения после второго слоя - {values_on_input_first_layer}')

            for i, ii in enumerate(self.first_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.first_hidden_neurons_weight):
                    values_on_input_second_layer[i] += \
                        values_on_input_first_layer[j] * self.first_hidden_neurons_weight[j][i]
            values_on_input_second_layer = [sigmoid(i) for i in values_on_input_second_layer]
            # print(f'Значения после второго слоя - {values_on_input_first_layer}')

            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    values_on_out_layer[i] += values_on_input_second_layer[j] * \
                                              self.second_hidden_neurons_weight[j][i]
            values_on_out_layer = [sigmoid(i) for i in values_on_out_layer]

            # обратное распрастранение ошибки
            # расчёт ошибки на всех уровнях
            # self.backpropagation()


            answer_error = [0] * len(values_on_out_layer)
            for i, ii in enumerate(answer_error):
                answer_error[i] += self.y[d][i] - values_on_out_layer[i]
            # print('Ошибки сети', answer_error)

            second_layer_error = [0] * len(values_on_input_second_layer)
            input_layer_error = [0] * len(values_on_input_first_layer)

            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    second_layer_error[j] += answer_error[j] * self.second_hidden_neurons_weight[j][i] *\
                                             grad_sigmoid(values_on_input_second_layer[i])

            for i, ii in enumerate(self.first_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.first_hidden_neurons_weight):
                    input_layer_error[i] += second_layer_error[i] * self.first_hidden_neurons_weight[j][i] *\
                    grad_sigmoid(values_on_input_first_layer[i])


            # обновляем веса
            for i, ii in enumerate(self.input_neurons_weight[0]):
                for j, jj in enumerate(self.input_neurons_weight):
                    self.input_neurons_weight[j][i] += \
                        input_layer_error[i] * self.learning_step
            # print(f'Новые веса для входного слоя {self.input_neurons_weight}')

            for i, ii in enumerate(self.first_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.first_hidden_neurons_weight):
                    self.first_hidden_neurons_weight[j][i] += \
                        second_layer_error[i]  * self.learning_step
            # print(f'Новые веса для входного слоя {self.input_neurons_weight}')

            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    self.second_hidden_neurons_weight[j][i] += \
                        answer_error[i]  * self.learning_step
            # print(f'Новые веса для второго слоя {self.input_neurons_weight}')

    def predict(self, new_data):
        values_on_input_first_layer = self.calk_value_on_layer(input_neurons_weight, new_data)

        values_on_input_second_layer = self.calk_value_on_layer(first_hidden_neurons_weight,
                                                                values_on_input_first_layer)

        values_on_out_layer = self.calk_value_on_layer(second_hidden_neurons_weight,
                                                       values_on_input_second_layer)

        answer_neural = find_answers(values_on_out_layer)
        print(f'На вход подалось {new_data}')
        print(f'Ответ нейросети: {answer_neural}')
        print(f'{values_on_out_layer}')
        # print()


if __name__ == '__main__':
    input_neurons_weight = get_weight(100, 30, 0., 0.5)
    first_hidden_neurons_weight = get_weight(30, 10, 0., 0.5)
    second_hidden_neurons_weight = get_weight(10, 3, 0., 0.5)
    data, y = convert_data()

    multi_layered_thing = Neural(data, y, input_neurons_weight, first_hidden_neurons_weight,
                                 second_hidden_neurons_weight,
                                 learning_step=0.1, iteration=1000)

    multi_layered_thing.train()

    multi_layered_thing.predict(convert(1))

    multi_layered_thing.predict(convert(3))

    multi_layered_thing.predict(convert(6))

    multi_layered_thing.predict(convert(7))

