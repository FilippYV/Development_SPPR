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


def mutation(mass, count_cities):
    while True:
        index_1 = random.randint(0, len(mass) - 1)
        index_2 = random.randint(0, len(mass) - 1)
        if index_1 != index_2:
            mass[index_1], mass[index_2] = mass[index_2], mass[index_1]
            return mass


def mutation_element(element, count_cities):
    switch = True
    # print(element)
    elements = element[0]
    elements.pop(-1)
    direction = random.randint(0, 1)  # если 1 то; + если 0 то -
    first_element_to_mutation = random.randint(0, count_cities - 1)
    second_element_to_mutation = first_element_to_mutation
    count = 0
    if direction:
        second_element_to_mutation += 1
    else:
        second_element_to_mutation -= 1
    while switch:
        on_off = random.randint(0, 1)
        if second_element_to_mutation == count_cities:
            second_element_to_mutation = 0
        if second_element_to_mutation == -7:
            second_element_to_mutation = -1
        if on_off == 0:
            if direction:
                second_element_to_mutation += 1
            else:
                second_element_to_mutation -= 1
        else:
            switch = False
    elements[first_element_to_mutation], elements[second_element_to_mutation] = \
        elements[second_element_to_mutation], elements[first_element_to_mutation]
    elements.append(elements[0])
    return elements


def leave_the_giant_enough(massive):
    new_massive = []
    for i in range(6):
        new_massive.append(massive[i])
    return new_massive


if __name__ == '__main__':
    count_cities = 6
    all_routes = generating_all_paths(count_cities)
    new_generation = generate_start_routes(all_routes, count_cities)
    iteration = 0
    minimum = [[], 1000000]
    while iteration < 10:
        mutation_generation = []
        for element_id, element in enumerate(new_generation):
            mutation_generation.append(mutation_element(element, count_cities))
        for i in mutation_generation:
            new_generation.append(i)
        new_generation = count_len(new_generation, all_routes)
        new_generation = leave_the_giant_enough(new_generation)
        print('Итерация -', iteration)
        for i in range(len(new_generation)):
            print(f"Елемент №{i + 1} - {new_generation[i]}")
        print()
        for i in range(len(new_generation)):
            if new_generation[i][1] < minimum[1]:
                minimum = new_generation[i]
        iteration += 1
    print(minimum, 'minimum')
