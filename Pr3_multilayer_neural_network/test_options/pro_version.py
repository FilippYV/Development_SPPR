import math
import random


def data_and_answer():
    data = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]]
    y = [[1, 0],
         [0, 1]]
    return data, y


def sigmoid(input_value):
    return 1 / (1 + math.e ** (-input_value))


def grad_sigmoid(input_value):
    return input_value * (1 - input_value)


def generate_weight_to_layer(count):
    weight = []
    for c in range(len(count) - 1):
        weight_layer = []
        for i in range(count[c]):
            weight_to_layer = []
            for j in range(count[c + 1]):
                weight_to_layer.append(random.uniform(0, 0.5))
            weight_layer.append(weight_to_layer)
        weight.append(weight_layer)
    for i in weight:
        print(i)
    return weight


class Neural:
    def __init__(self, data, y, weight, learning_step, iteration):
        self.data = data
        self.y = y
        self.weight = weight
        self.learning_step = learning_step
        self.iteration = iteration

    def train(self):
        epoch = 0
        while epoch != self.iteration + 1:
            self.direct_distribution()
            if epoch % 100 == 0:
                print(f'Эпоха - {epoch + 1}')
            epoch += 1

    def get_value_on_layer(self, weight_, data_, value_):
        for i, ii in enumerate(weight_[0]):
            for j, jj in enumerate(weight_):
                value_[i] += \
                    data_[j] * weight_[j][i]
        value_ = [sigmoid(i) for i in value_]
        return value_

    def backpropagation(self, dd):
        total_quadratic_error = 0
        for i, ii in enumerate(self.values_on_layer[-1]):
            total_quadratic_error += (ii - dd[i]) ** 2
        total_quadratic_error *= 0.5
        print(total_quadratic_error)

        delta_error = [0] * len(self.values_on_layer[-1])
        for i, ii in enumerate(delta_error):
            delta_error[i] = self.values_on_layer[-1][i] * (1 - self.values_on_layer[-1][i]) * (
                        dd[i] - self.values_on_layer[-1][i])
        print(delta_error)

        summ = 0
        for i, ii in enumerate(self.weight[-1][0]):
            for j, jj in enumerate(self.weight[-1]):
                summ += self.weight[j][i] * delta_error[i]
        print(summ)

        delta_error_1 = [0] * len(self.values_on_layer[-1])



        # for i, ii in enumerate(self.weight[-1][0]):
        #     for j, jj in enumerate(self.weight[-1]):
        #         self.weight[j][i]

        delta_error_1 = [0] * len(self.values_on_layer[-1])

        for i, ii in enumerate(self.weight[-1][0]):
            for j, jj in enumerate(self.weight[-1]):
                self.weight[-1][j] += delta_error[j] * self.weight[-1][j][i]  * self.learning_step * self.values_on_layer[-2][i]
        print(delta_error_1)

        exit(123)

    def direct_distribution(self):
        for d, dd in enumerate(self.data):  # проход по данным
            self.values_on_layer = []
            for i in range(len(self.weight)):
                self.values_on_layer.append([0] * len(self.weight[i][0]))
            print(self.values_on_layer)

            self.values_on_layer[0] = self.get_value_on_layer(self.weight[0], dd, self.values_on_layer[0])

            self.values_on_layer[1] = self.get_value_on_layer(self.weight[1], self.values_on_layer[0],
                                                              self.values_on_layer[1])

            self.values_on_layer[2] = self.get_value_on_layer(self.weight[2], self.values_on_layer[1],
                                                              self.values_on_layer[2])
            print(self.values_on_layer)

            self.backpropagation(dd)
            exit(123)

            # # обратное распрастранение ошибки
            # # расчёт ошибки на всех уровнях
            # answer_error = [0] * len(values_on_out_layer)
            # for i, ii in enumerate(answer_error):
            #     answer_error[i] = self.y[d][i] - values_on_out_layer[i]
            # # print('Ошибки сети', answer_error)
            #
            # second_layer_error = [0] * len(values_on_input_second_layer)
            # input_layer_error = [0] * len(input_neurons_weight)
            #
            # for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
            #     for j, jj in enumerate(self.second_hidden_neurons_weight):
            #         second_layer_error[i] += answer_error[i] * self.second_hidden_neurons_weight[j][i]
            # # print('Ошибки на втором скрытом слое', second_layer_error)
            #
            # for i, ii in enumerate(self.first_hidden_neurons_weight[0]):
            #     for j, jj in enumerate(self.first_hidden_neurons_weight):
            #         input_layer_error[i] += second_layer_error[i] * self.first_hidden_neurons_weight[j][i]
            # # print('Ошибки на втором скрытом слое', second_layer_error)
            #
            # # обновляем веса
            # for i, ii in enumerate(self.input_neurons_weight[0]):
            #     for j, jj in enumerate(self.input_neurons_weight):
            #         self.input_neurons_weight[j][i] += \
            #             input_layer_error[i] * grad_sigmoid(values_on_input_first_layer[i]) * \
            #             data[d][i] * self.learning_step
            # # print(f'Новые веса для входного слоя {self.input_neurons_weight}')
            #
            # for i, ii in enumerate(self.first_hidden_neurons_weight[0]):
            #     for j, jj in enumerate(self.first_hidden_neurons_weight):
            #         self.first_hidden_neurons_weight[j][i] += \
            #             second_layer_error[i] * grad_sigmoid(values_on_input_second_layer[i]) * \
            #             values_on_input_first_layer[j] * self.learning_step
            # # print(f'Новые веса для входного слоя {self.input_neurons_weight}')
            #
            # for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
            #     for j, jj in enumerate(self.second_hidden_neurons_weight):
            #         self.second_hidden_neurons_weight[j][i] += \
            #             answer_error[i] * grad_sigmoid(values_on_out_layer[i]) * \
            #             values_on_input_second_layer[j] * self.learning_step
            # # print(f'Новые веса для второго слоя {self.input_neurons_weight}')


if __name__ == '__main__':
    count_layer = [15, 5, 5, 2]
    weight = generate_weight_to_layer(count_layer)
    data, y = data_and_answer()
    multi_layered_thing = Neural(data, y, weight, learning_step=0.1, iteration=100)
    multi_layered_thing.train()
