import math
import random
import cv2


def get_x_y():
    # x = [[0.25, 0.02]]
    # y = [[0, 1]]
    x = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
         [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
         [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
         [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
         [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]]
    y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    return x, y


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
    data = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]]
    y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    return data, y


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


def get_data_and_y(start_photo, stop):
    X_train = []
    y_train = []
    for k in range(start_photo, stop + 1):
        data = []
        for i, ii in enumerate(cv2.imread(f"../static/photo/{k}.png", 0)):
            for j, jj in enumerate(ii):
                if jj >= 128:
                    data.append(0)
                else:
                    data.append(1)
        X_train.append(data)
        answer = [0] * ((stop + 1) - start_photo)
        answer[k - 1] = 1
        y_train.append(answer)
    print(y_train)
    data = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]]
    y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    return data, y


def sigmoid(input_value):
    return 1 / (1 + math.e ** (-input_value))


def grad_sigmoid(input_value):
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


def find_answers(answers):
    maximum = -1
    index = -1
    for i, ii in enumerate(answers):
        if ii > maximum:
            maximum = ii
            index = i
    return index


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

    def direct_distribution(self):
        for d, dd in enumerate(self.data):  # проход по данным
            values_on_input_first_layer = [0] * len(self.first_hidden_neurons_weight[0])
            values_on_input_second_layer = [0] * len(self.second_hidden_neurons_weight[0])
            values_on_out_layer = [0] * len(self.second_hidden_neurons_weight[0])

            for i, ii in enumerate(self.input_neurons_weight[0]):
                for j, jj in enumerate(self.input_neurons_weight):
                    values_on_input_first_layer[i] += dd[j] * self.input_neurons_weight[j][i]
            values_on_input_first_layer = [sigmoid(i) for i in values_on_input_first_layer]

            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    values_on_input_second_layer[i] += values_on_input_first_layer[j] * self.second_hidden_neurons_weight[j][i]
            values_on_input_second_layer = [sigmoid(i) for i in values_on_input_second_layer]


            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    values_on_out_layer[i] += values_on_input_second_layer[j] * \
                                              self.second_hidden_neurons_weight[j][i]
            values_on_out_layer = [sigmoid(i) for i in values_on_out_layer]


            # обратное распрастранение ошибки
            # расчёт ошибки на всех уровнях
            answer_error = [0] * len(values_on_out_layer)
            for i, ii in enumerate(answer_error):
                answer_error[i] = self.y[d][i] - values_on_out_layer[i]
            # print('Ошибки сети', answer_error)

            second_layer_error = [0] * len(values_on_input_second_layer)
            first_layer_error = [0] * len(values_on_input_second_layer)

            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    second_layer_error[j] += answer_error[i] * self.second_hidden_neurons_weight[j][i]

            for i, ii in enumerate(self.first_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.first_hidden_neurons_weight):
                    first_layer_error[j] += second_layer_error[i] * self.first_hidden_neurons_weight[j][i]


            # обновляем веса
            for i, ii in enumerate(self.input_neurons_weight[0]):
                for j, jj in enumerate(self.input_neurons_weight):
                    self.input_neurons_weight[j][i] += \
                        first_layer_error[i] * grad_sigmoid(values_on_input_first_layer[i]) * \
                        data[d][i] * self.learning_step

            for i, ii in enumerate(self.input_neurons_weight[0]):
                for j, jj in enumerate(self.input_neurons_weight):
                    self.input_neurons_weight[j][i] += \
                        second_layer_error[i] * grad_sigmoid(values_on_input_second_layer[i]) * \
                        data[d][i] * self.learning_step
            # print(f'Новые веса для входного слоя {self.input_neurons_weight}')

            for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
                for j, jj in enumerate(self.second_hidden_neurons_weight):
                    self.second_hidden_neurons_weight[j][i] += \
                        answer_error[i] * grad_sigmoid(values_on_out_layer[i]) * \
                        values_on_input_second_layer[j] * self.learning_step
            # print(f'Новые веса для второго слоя {self.input_neurons_weight}')

    def predict(self, new_data):
        values_on_input_second_layer = [0] * len(self.second_hidden_neurons_weight)
        values_on_out_layer = [0] * len(self.second_hidden_neurons_weight[0])
        for i, ii in enumerate(self.input_neurons_weight[0]):
            for j, jj in enumerate(self.input_neurons_weight):
                values_on_input_second_layer[i] += new_data[j] * self.input_neurons_weight[j][i]
        values_on_input_second_layer = [sigmoid(i) for i in values_on_input_second_layer]
        # print(f'Значения после второго слоя - {values_on_input_second_layer}')

        for i, ii in enumerate(self.second_hidden_neurons_weight[0]):
            for j, jj in enumerate(self.second_hidden_neurons_weight):
                values_on_out_layer[i] += values_on_input_second_layer[j] * \
                                          self.second_hidden_neurons_weight[j][i]
        values_on_out_layer = [sigmoid(i) for i in values_on_out_layer]
        answer_neural = find_answers(values_on_out_layer)
        print(f'На вход подалось {new_data}')
        print(f'Ответ нейросети: {answer_neural}')
        print(f'{values_on_out_layer}')
        print()


if __name__ == '__main__':
    input_neurons_weight = get_weight(15, 4, 0., 0.5)
    first_hidden_neurons_weight = get_weight(4, 4, 0., 0.5)
    second_hidden_neurons_weight = get_weight(4, 10, 0., 0.5)
    data, y = convert_data()
    print(y)

    multi_layered_thing = Neural(data, y, input_neurons_weight, first_hidden_neurons_weight,
                                 second_hidden_neurons_weight,
                                 learning_step=0.1, iteration=100000)

    multi_layered_thing.train()

    multi_layered_thing.predict([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1])

    multi_layered_thing.predict([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
