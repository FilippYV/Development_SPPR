import random
import itertools


def generating_all_paths(count):
    mass_all_routes = []
    # for i in range(count - 1):  # генераця всех путей
    #     for j in range(i, count):
    #         mass_all_routes.append([i, j, random.randint(20, 50)])
    print('Таблица всех путей')
    for i in mass_all_routes:
        print(i)
    print()
    mass_all_routes = [[0, 0, 40], [0, 1, 47], [0, 2, 48], [0, 3, 22], [0, 4, 31], [1, 1, 31], [1, 2, 24],
                       [1, 3, 48], [1, 4, 47], [2, 2, 41], [2, 3, 24], [2, 4, 30], [3, 3, 25], [3, 4, 48]]
    return mass_all_routes


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
    # start_routers = [[0, 1, 3, 4, 2, 0], [3, 2, 1, 4, 0, 3], [3, 4, 1, 0, 2, 3], [2, 0, 4, 1, 3, 2], [1, 0, 3, 4, 2, 1],
    #                  [4, 3, 0, 2, 1, 4]]
    return count_len(start_routers, routes)


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


def sort_by_long(mass):
    new_mass = []
    while len(mass) != 0:
        minimum = [0, 10000]
        for i in range(len(mass)):
            if mass[i][1] < minimum[1]:
                minimum = [i, mass[i][1]]
        new_mass.append(mass[minimum[0]])
        mass.pop(minimum[0])
    # print('Отсортированная по лучшим особям популяция')
    # for i in new_mass:
    #     print(i)
    # print()
    return new_mass


def crossing(mass, count_s, all_routes):
    number_of_descendants = len(mass) // 3
    # print('количество потомков -', number_of_descendants)
    massive = []
    element_to_split = []
    while len(element_to_split) != number_of_descendants:
        parent_for_crossing = random.randint(0, len(mass) - 1)
        if parent_for_crossing not in element_to_split:
            element_to_split.append(parent_for_crossing)
    for i in range(len(element_to_split)):
        massive.append(random.randint(1, len(mass[0][0]) - 2))
    # print(element_to_split, massive)
    mass_gen = []
    # count = len(element_to_split) // 2
    # print(len(element_to_split) // 2)
    for index in range(0, len(element_to_split) // 2):
        inx_start = index * 2
        for i in range(inx_start, inx_start + 2):
            new_elem = []
            for j in range(0, i):  # часть 1 родителя
                if mass[element_to_split[inx_start]][0][j] not in new_elem:
                    new_elem.append(mass[element_to_split[inx_start]][0][j])
            for j in range(i, len(mass[0][0])):  # часть 2 родителя
                if mass[element_to_split[inx_start + 1]][0][j] not in new_elem:
                    new_elem.append(mass[element_to_split[inx_start + 1]][0][j])

            # print(element_to_split[inx_start])
            for j in range(0, len(mass[0][0])):
                if mass[element_to_split[inx_start]][0][j] not in new_elem:
                    new_elem.append(mass[element_to_split[inx_start]][0][j])

            for j in range(0, len(mass[0][0])):
                if mass[element_to_split[inx_start + 1]][0][j] not in new_elem:
                    new_elem.append(mass[element_to_split[inx_start + 1]][0][j])
            if random.randint(0, 100) < 10:
                new_elem = mutation(new_elem)
            new_elem.append(new_elem[0])
            mass.append(new_elem)
            # print(new_elem)

    # print()
    for i in range(len(mass)):
        if i not in element_to_split:
            mass_gen.append(mass[i])
    # for i in mass_gen:
    #     print(i)
    mass_gen = sort_by_long(mass_gen)
    return count_len(mass_gen, all_routes)


def mutation(mass):
    while True:
        index_1 = random.randint(0, len(mass) - 1)
        index_2 = random.randint(0, len(mass) - 1)
        if index_1 != index_2:
            mass[index_1], mass[index_2] = mass[index_2], mass[index_1]
            return mass


if __name__ == '__main__':
    count_cities = 5
    all_routes = generating_all_paths(count_cities)
    new_generation = generate_start_routes(all_routes, count_cities)
    iteration = 0
    minimum = [[], 1000000]
    while iteration < 15:
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