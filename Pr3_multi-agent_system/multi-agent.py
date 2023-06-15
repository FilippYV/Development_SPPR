import math
import random
import decimal


def get_start_weight():  # генерируем веса для 15 входных нейронов
    input_neurons_weight = []
    for j in range(15):
        weight = []
        for i in range(5):
            weight.append(round(random.uniform(0, 0.01), 4))
        input_neurons_weight.append(weight)
    print('Начльные веса входных нейронов')
    for i in input_neurons_weight:
        print(i)
    print()
    return input_neurons_weight


def get_weight_weight():  # генерируем веса для 15 входных нейронов
    hidden_neurons_weight = []
    for j in range(5):
        weight = []
        for i in range(10):
            weight.append(round(random.uniform(0, 0.01), 4))
        hidden_neurons_weight.append(weight)
    print('Начльные веса скрытых нейронов')
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
    # y = [[0],
    #      [1],
    #      [2],
    #      [3],
    #      [4],
    #      [5],
    #      [6],
    #      [7],
    #      [8],
    #      [9]]
    return data, y


def singmoid(imput_value):
    return round(1 / (1 + math.e ** (-imput_value)), 3)


def relu(x):
    if x < 0:
        return 0
    elif x >= 0:
        return x


def combiner(x):
    if x >= 0.5:
        return 1
    else:
        return 0


def count_mse(pred, answer):
    return (answer - pred) ** 2


# def reverse_propagation(values_on_output_layer, values_on_hidden_layer, data, y, input_neurons_weight,
#                         hidden_neurons_weight):
#     # производная от mse и активации
#     # dL_by_relu = 2 * (answer - pred)
#     # print(dL_by_relu, 'dL_by_relu')
#     pass
#     # производная от релу
#     exit(123)


# def reverse_propagation_sigm(pred, answer):
#     # производная от mse и активации
#     dL_by_sigm = 2 * (answer - pred)
#     print(dL_by_sigm, 'dL_by_sigm')
#
#     # производная от релу
#
#     exit(123)


def direct_distribution(data, y, input_neurons_weight, hidden_neurons_weight, neurons_offset):
    values_on_hidden_layer = [0] * len(hidden_neurons_weight)
    values_on_output_layer = [0] * len(y[0])
    for d, dd in enumerate(data):  # проход по данным
        for i, ii in enumerate(input_neurons_weight[0]):
            for j, jj in enumerate(input_neurons_weight):
                values_on_hidden_layer[i] += dd[j] * input_neurons_weight[j][i]
        values_on_hidden_layer = [singmoid(i) for i in values_on_hidden_layer]
        print(f'Значения на скрытом слое - {values_on_hidden_layer}')

        for i, ii in enumerate(y[0]):
            for j, jj in enumerate(hidden_neurons_weight):
                values_on_output_layer[i] += dd[j] * hidden_neurons_weight[j][i]
        values_on_output_layer = [singmoid(i) for i in values_on_output_layer]
        print(f'Значения на выходном слое - {values_on_output_layer}')

        # обратное распрастранение ошибки
        weighted_amount = [0] * len(hidden_neurons_weight) # вычисляем взвешенную сумму
        for i, ii in enumerate(weighted_amount):
            for j, jj in enumerate(hidden_neurons_weight[0]):
                weighted_amount[i] += hidden_neurons_weight[i][j] * values_on_output_layer[j]
        print(f'взвешенная сумма {weighted_amount}' )

        for i, ii in enumerate(weighted_amount):
            weighted_amount[i] *= (values_on_hidden_layer[i] * (1 - values_on_hidden_layer[i]))
        print(f'Локальный градиент для нейронов выходного слоя {weighted_amount}')
        d_for_hidden_layer = [0] * len(hidden_neurons_weight)
        # for i, ii in enumerate(hidden_neurons_weight):
        #     d_for_hidden_layer[i] +=
        # mse = count_mse(values_on_output_layer[m], y[d][m])  # квадрат ошибки
        # print('mse -', mse)
        # dL_by_sigm = 2 * (values_on_output_layer[m] - y[d][m])  # производная квадрата ошибки


        d_for_hidden_layer = [0] * len(hidden_neurons_weight)
        for i, ii in enumerate(d_for_hidden_layer):
            d_for_hidden_layer[i] += weighted_amount[i] * values_on_hidden_layer[i]




        # print('dL_by_sigm -', dL_by_sigm)
        # d_for_hidden_layer = [0] * len(hidden_neurons_weight)
        d_for_input_layer = [0] * len(input_neurons_weight)
        # dL_by_sigm * (values_on_output_layer[i] * (1 - values_on_output_layer[i]))
        for i, ii in enumerate(d_for_hidden_layer):  # считаем производные для скрытого словая
            d_for_hidden_layer[i] = dL_by_sigm * (values_on_output_layer[i] * (1 - values_on_output_layer[i]))
            # print(d_for_hidden_layer[i])
            for j, jj in enumerate(input_neurons_weight):  # считаем производные для скрытого словая
                d_for_input_layer[j] = d_for_hidden_layer[i] * (data[m][j])
                # print(d_for_hidden_layer[i])
        for i, ii in enumerate(input_neurons_weight):
            input_neurons_weight[i] += d_for_input_layer[i]
        for i, ii in enumerate(hidden_neurons_weight):
            input_neurons_weight[i] += d_for_hidden_layer[i]

        # reverse_propagation(values_on_output_layer, values_on_hidden_layer, data, y, input_neurons_weight, hidden_neurons_weight)

        # for m, mm in enumerate(values_on_output_layer):
        #     mse = count_mse(values_on_output_layer[m], y[d][m])  # квадрат ошибки
        #     # print('mse -', mse)
        #     dL_by_sigm = 2 * (values_on_output_layer[m] - y[d][m])  # производная квадрата ошибки
        #     # print('dL_by_sigm -', dL_by_sigm)
        #     d_for_hidden_layer = [0] * len(hidden_neurons_weight)
        #     d_for_input_layer = [0] * len(input_neurons_weight)
        #     # dL_by_sigm * (values_on_output_layer[i] * (1 - values_on_output_layer[i]))
        #     for i, ii in enumerate(d_for_hidden_layer):  # считаем производные для скрытого словая
        #         d_for_hidden_layer[i] = dL_by_sigm * (values_on_output_layer[i] * (1 - values_on_output_layer[i]))
        #         # print(d_for_hidden_layer[i])
        #         for j, jj in enumerate(input_neurons_weight):  # считаем производные для скрытого словая
        #             d_for_input_layer[j] = d_for_hidden_layer[i] * (data[m][j])
        #             # print(d_for_hidden_layer[i])
        #     for i, ii in enumerate(input_neurons_weight):
        #         input_neurons_weight[i] += d_for_input_layer[i]
        #     for i, ii in enumerate(hidden_neurons_weight):
        #         input_neurons_weight[i] += d_for_hidden_layer[i]

        # mse = count_mse(values_on_output_layer[0], y[d][0])
        # print(mse)
        #
        # # обратное распрастранение ошибки
        # exit(132)


def train(data, y, iteration, input_neurons_weight, hidden_neurons_weight, neurons_offset):
    count = 0
    while count != iteration:
        direct_distribution(data, y, input_neurons_weight, hidden_neurons_weight, neurons_offset)
        print(f'Итерация {count}')
        print(f'Входные нейроны {input_neurons_weight}')
        print(f'Скрыте скрытые {input_neurons_weight}')
        print()
        count += 1
    pass


if __name__ == '__main__':
    input_neurons_weight = get_start_weight()
    hidden_neurons_weight = get_weight_weight()
    neurons_offset = [0.5, 0.5]
    data, y = get_data_and_y()
    iteration = 10
    train(data, y, iteration, input_neurons_weight, hidden_neurons_weight, neurons_offset)
