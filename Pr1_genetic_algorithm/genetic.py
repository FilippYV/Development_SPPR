import itertools
import random


def sort_by_long(mass):
    new_mass = []
    while len(mass) != 0:
        minimum = [0, 10000]
        for i in range(len(mass)):
            if mass[i][1] < minimum[1]:
                minimum = [i, mass[i][1]]
        new_mass.append(mass[minimum[0]])
        mass.pop(minimum[0])
    print('Отсортированная по лучшим особям популяция')
    for i in range(0, 2):
        print(new_mass)
    print()
    return new_mass


def count_len(mass, routes):
    for i in range(len(mass)):
        if len(mass[i]) != 2 and mass[i][0] is not list:
            way = mass[i]
            long_way = 0
            for point in range(len(way) - 1):
                for long in routes:
                    if way[point] == long[0] and way[point + 1] == long[1] or \
                            way[point] == long[1] and way[point + 1] == long[0]:
                        long_way += long[-1]
                        break
            mass[i] = [way, long_way]
    return sort_by_long(mass)


def generate_start_routes(routes, cities):
    mass_routes = []
    start_gen_routers = []
    for i in range(cities):
        start_gen_routers.append(i)
    print(start_gen_routers)
    new_data = itertools.permutations(start_gen_routers, cities)
    for mass in new_data:
        mass_routes.append(list(mass))
    start_routers = []
    for i in range(6):
        index = random.randint(0, len(mass_routes) - 1)
        if mass_routes[index] not in start_routers:
            mass_routes[index].append(mass_routes[index][0])
            start_routers.append(mass_routes[index])
    print('Начальная популяция!')
    print(start_routers)
    print()
    # start_routers =
    return count_len(start_routers, routes)


def generating_all_paths(count):
    mass_all_routes = []
    for i in range(count - 1):  # генераця всех путей
        for j in range(i, count):
            mass_all_routes.append([i, j, random.randint(20, 50)])
    mass_all_routes = [[0, 1, 52], [0, 2, 73], [0, 3, 87], [0, 4, 66], [0, 5, 89], [1, 2, 60], [1, 3, 59], [1, 4, 54],
                       [1, 5, 90], [2, 3, 100], [2, 4, 79], [2, 5, 79], [3, 4, 58], [3, 5, 93], [4, 5, 69]]
    print('Таблица всех путей')
    for i in mass_all_routes:
        print(i)
    print()
    return mass_all_routes


def mutation(mass):
    while True:
        index_1 = random.randint(0, len(mass) - 1)
        index_2 = random.randint(0, len(mass) - 1)
        if index_1 != index_2:
            mass[index_1], mass[index_2] = mass[index_2], mass[index_1]
            return mass


def crossing(mass, count_s, all_routes):
    if len(mass) // 2 == 0:
        number_of_descendants = len(mass) // 2
    else:
        number_of_descendants = len(mass) // 2 - 1
    print('количество потомков -', number_of_descendants)

    massive = []
    element_to_split = []
    while len(element_to_split) != number_of_descendants // 2:
        pod = []
        while len(pod) != 2:
            parent_for_crossing = random.randint(0, len(mass) - 1)
            if parent_for_crossing not in pod:
                pod.append(parent_for_crossing)
        element_to_split.append(pod)
    for i in range(len(element_to_split)):
        massive.append(random.randint(1, len(mass[0][0]) - 2))
    massive_2 = []
    for i in range(len(element_to_split)):
        massive_2.append(random.randint(1, len(mass[0][0]) - 2))
    for ii, i in enumerate(element_to_split):
        element = []
        # print(mass[i[0]][0])
        for j in range(0, massive[ii]):
            if mass[i[0]][0][j] not in element:
                element.append(mass[i[0]][0][j])

        for j in range(massive[ii], len(mass[0][0])):
            if mass[i[1]][0][j] not in element:
                element.append(mass[i[1]][0][j])

        for j in range(0, len(mass[0][0])):
            if mass[i[0]][0][j] not in element:
                element.append(mass[i[0]][0][j])

        for j in range(0, len(mass[0][0])):
            if mass[i[1]][0][j] not in element:
                element.append(mass[i[1]][0][j])

        if random.randint(0, 100) < 10:
            element = mutation(element)
        element.append(element[0])
        mass.append(element)
        element = []
        # print(mass[i[0]][0])
        for j in range(0, massive_2[ii]):
            if mass[i[0]][0][j] not in element:
                element.append(mass[i[0]][0][j])

        for j in range(massive_2[ii], len(mass[0][0])):
            if mass[i[1]][0][j] not in element:
                element.append(mass[i[1]][0][j])

        for j in range(0, len(mass[0][0])):
            if mass[i[0]][0][j] not in element:
                element.append(mass[i[0]][0][j])

        for j in range(0, len(mass[0][0])):
            if mass[i[1]][0][j] not in element:
                element.append(mass[i[1]][0][j])

        if random.randint(0, 100) < 10:
            element = mutation(element)
        element.append(element[0])
        mass.append(element)
    # mass_gen = sort_by_long(mass)
    return count_len(mass, all_routes)


if __name__ == '__main__':
    count_cities = 6
    all_routes = generating_all_paths(count_cities)
    new_generation = generate_start_routes(all_routes, count_cities)
    iteration = 0
    minimum = [[], 1000000]
    while iteration < 10:
        new_generation = crossing(new_generation, count_cities, all_routes)
        print('Итерация -', iteration)
        for i in range(len(new_generation)):
            print(f"Елемент №{i} - {new_generation[i]}")
        print()
        for i in range(len(new_generation)):
            if new_generation[i][1] < minimum[1]:
                minimum = new_generation[i]
        iteration += 1
    print(minimum, 'minimum')
