import math
import random


def get_start_weight():  # генерируем веса для 15 входных нейронов
    input_neurons_weight = []
    for j in range(15):
        weight = []
        for i in range(5):
            weight.append(random.uniform(-0.5, 0.5))
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
            weight.append(random.uniform(-0.5, 0.5))
        hidden_neurons_weight.append(weight)
    print('Начльные веса на первом слое скрытых нейронов')
    for i in hidden_neurons_weight:
        print(i)
    print()
    return hidden_neurons_weight


def get_second_hidden_weight():  # генерируем веса для 2 скрытого слоя, 5 нейронов
    hidden_neurons_weight = []
    for i in range(5):
        hidden_neurons_weight.append(random.uniform(-0.5, 0.5))
    print('Начльные веса на втром слое скрытых нейронов')
    for i in hidden_neurons_weight:
        print(i)
    print()
    return hidden_neurons_weight


def get_data_and_y():
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
    # y = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    y = [[0],
         [1],
         [2],
         [3],
         [4],
         [5],
         [6],
         [7],
         [8],
         [9]]
    return data, y


def singmoid(input_value):
    return round(1 / (1 + math.e ** (-input_value)), 3)


def grad_singmoid(input_value):
    return round(input_value * (1 - input_value))


def leaky_relu(x):
    if x < 0:
        return 0.01
    elif x >= 0:
        return x


def grad_leaky_relu(x):
    if x < 0:
        return 0.01
    elif x >= 0:
        return 1


def count_mse(pred, answer):
    return (answer - pred) ** 2


def direct_distribution(data, y, input_neurons_weight, first_hidden_neurons_weight,
                        second_hidden_neurons_weight, neurons_offset):
    learning_step = 0.1
    values_on_input_first_layer = [0] * len(first_hidden_neurons_weight)
    values_on_input_second_layer = [0] * len(second_hidden_neurons_weight)
    values_on_out_layer = 0
    for d, dd in enumerate(data):  # проход по данным
        for i, ii in enumerate(input_neurons_weight[0]):
            for j, jj in enumerate(input_neurons_weight):
                values_on_input_first_layer[i] += dd[j] * input_neurons_weight[j][i]
        values_on_input_first_layer = [singmoid(i) for i in values_on_input_first_layer]
        # print(f'Значения после первого слоя - {values_on_input_first_layer}')

        for i, ii in enumerate(first_hidden_neurons_weight[0]):
            for j, jj in enumerate(first_hidden_neurons_weight):
                values_on_input_second_layer[j] += values_on_input_first_layer[j] * first_hidden_neurons_weight[j][i]
        values_on_input_second_layer = [singmoid(i) for i in values_on_input_second_layer]
        # print(f'Значения после второго слоя - {values_on_input_second_layer}')

        for j, jj in enumerate(second_hidden_neurons_weight):
            values_on_out_layer += values_on_input_second_layer[j] * second_hidden_neurons_weight[j]
        values_on_out_layer = leaky_relu(values_on_out_layer)
        # print(f'Значения на выходном слое слое - {values_on_out_layer}')
        # print()

        # обратное распрастранение ошибки
        # расчёт ошибки на всех уровнях
        # print('Обратное распрастранение ошибки\n'
        #       'Расчёт ошибки на всех уровнях')
        answer_error = y[d][0] - values_on_out_layer
        # print('Ошибки сети', answer_error)

        second_layer_error = [0] * len(values_on_input_second_layer)
        first_layer_error = [0] * len(values_on_input_first_layer)

        for j, jj in enumerate(second_hidden_neurons_weight):
            second_layer_error[j] += answer_error * second_hidden_neurons_weight[j]
        # print('Ошибки на втором скрытом слое', second_layer_error)

        for i, ii in enumerate(first_hidden_neurons_weight[0]):
            for j, jj in enumerate(first_hidden_neurons_weight):
                first_layer_error[j] += second_layer_error[j] * first_hidden_neurons_weight[j][i]
        # print('Ошибки на первом скрытом слое', first_layer_error)
        # print()

        # обновляем веса

        for i, ii in enumerate(input_neurons_weight[0]):
            for j, jj in enumerate(input_neurons_weight):
                input_neurons_weight[j][i] += \
                    first_layer_error[i] * grad_singmoid(values_on_input_first_layer[i]) * data[d][i] * learning_step
        # print(f'Новые веса для первого слоя')
        # for i in input_neurons_weight:
        #     print(i)
        # print()

        for i, ii in enumerate(first_hidden_neurons_weight[0]):
            for j, jj in enumerate(first_hidden_neurons_weight):
                first_hidden_neurons_weight[j][i] += \
                    second_layer_error[i] * grad_singmoid(values_on_input_second_layer[i]) * \
                    values_on_input_first_layer[i] * learning_step
        # print(f'Новые веса для второго слоя')
        # for i in first_hidden_neurons_weight:
        #     print(i)
        # print()

        for j, jj in enumerate(second_hidden_neurons_weight):
            second_hidden_neurons_weight[j] += \
                answer_error * grad_leaky_relu(values_on_out_layer) * \
                values_on_input_second_layer[j] * learning_step
        # print(f'Новые веса для выходного слоя')
        # for i in second_hidden_neurons_weight:
        #     print(i)
        # print()
    return input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight


def train(data, y, iteration, input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight,
          neurons_offset):
    count = 0
    while count != iteration:
        input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight = \
            direct_distribution(data, y, input_neurons_weight, first_hidden_neurons_weight,
                                second_hidden_neurons_weight, neurons_offset)
        print(f'Итерация {count}')
        print(f'Входной слой веса {input_neurons_weight}')
        print(f'Первый скрытый слой веса {first_hidden_neurons_weight}')
        print(f'Второй скрытый слой веса {first_hidden_neurons_weight}')
        print()
        count += 1
    return input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight


def predict(input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight, new_data):
    values_on_input_first_layer = [0] * len(first_hidden_neurons_weight)
    values_on_input_second_layer = [0] * len(second_hidden_neurons_weight)
    values_on_out_layer = 0
    for i, ii in enumerate(input_neurons_weight[0]):
        for j, jj in enumerate(input_neurons_weight):
            values_on_input_first_layer[i] += new_data[j] * input_neurons_weight[j][i]
    values_on_input_first_layer = [singmoid(i) for i in values_on_input_first_layer]
    # print(f'Значения после первого слоя - {values_on_input_first_layer}')

    for i, ii in enumerate(first_hidden_neurons_weight[0]):
        for j, jj in enumerate(first_hidden_neurons_weight):
            values_on_input_second_layer[j] += values_on_input_first_layer[j] * first_hidden_neurons_weight[j][i]
    values_on_input_second_layer = [singmoid(i) for i in values_on_input_second_layer]
    # print(f'Значения после второго слоя - {values_on_input_second_layer}')

    for j, jj in enumerate(second_hidden_neurons_weight):
        values_on_out_layer += values_on_input_second_layer[j] * second_hidden_neurons_weight[j]
    values_on_out_layer = leaky_relu(values_on_out_layer)
    print(f'На вход подалось {new_data}')
    print(f'Ответ нейросети: {values_on_out_layer}')
    print()


if __name__ == '__main__':
    input_neurons_weight = get_start_weight()
    first_hidden_neurons_weight = get_first_hidden_weight()
    second_hidden_neurons_weight = get_second_hidden_weight()
    neurons_offset = [0.5, 0.5]
    data, y = get_data_and_y()
    iteration = 10
    input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight = \
        train(data, y, iteration, input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight,
          neurons_offset)

    new_data = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    predict(input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight, new_data)

    new_data = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    predict(input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight, new_data)

    new_data = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    predict(input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight, new_data)

    new_data = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    predict(input_neurons_weight, first_hidden_neurons_weight, second_hidden_neurons_weight, new_data)
