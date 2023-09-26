import math
import random as random
import bisect
import time

from matplotlib import pyplot as plt


def genetic_algorithm(population, fn_fitness, gene_pool, mutation, recombine, selection, fn_thres=None, ngen=1000,
                      pmut=0.1):
    # for each generation
    for i in range(ngen):
        #print(i)

        # create a new population
        new_population = []

        # repeat to create len(population) individuals
        for i in range(len(population)):
            # select the parents
            p1, p2 = selection(2, population, fn_fitness)

            # recombine the parents, thus producing the child
            child_x, child_y = recombine(p1, p2)

            child = child_x if fn_fitness(child_x) > fn_fitness(child_y) else child_y

            # mutate the child
            child = mutation(child, gene_pool, pmut)

            # add the child to the new population
            new_population.append(child)

        # move to the new population
        population = new_population

        # check if one of the individuals achieved a fitness of fn_thres; if so, return it
        fittest_individual = fitness_threshold(fn_fitness, fn_thres, population)
        if fittest_individual:
            return fittest_individual

    # return the individual with highest fitness
    return max(population, key=fn_fitness)


# get the best individual of the received population and return it if its
# fitness is higher than the specified threshold fn_thres
def fitness_threshold(fn_fitness, fn_thres, population):
    if not fn_thres:
        return None

    fittest_individual = max(population, key=fn_fitness)
    if fn_fitness(fittest_individual) >= fn_thres:
        return fittest_individual

    return None

# return a single sample from seq; the probability of a sample being returned
# is proportional to its weight
def weighted_sampler(seq, weights):
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


# genetic operator for recombination (crossover) of individuals;
# this function implements single-point crossover, where the resulting individual
# carries a portion [0,c] from parent x and a portion [c,n] from parent y, with
# c selected at random
def recombine(x, y):
    n = len(x)
    c = random.randrange(0, n)
    return x[:c] + y[c:]


# genetic operator for mutation;
# this function implements uniform mutation, where a single element of the
# individual is selected at random and its value is changed by a randomly chosen
# value (out of the possible values in gene_pool)
def mutate(x, gene_pool, pmut):
    # if random >= pmut, then no mutation is performed
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    g = len(gene_pool)
    c = random.randrange(0, n)  # gene to be mutated
    r = random.randrange(0, g)  # new value of the selected gene

    new_gene = gene_pool[r]
    return x[:c] + [new_gene] + x[c + 1:]


def init_population(pop_number, gene_pool, state_length):
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = random.sample(list(gene_pool), len(gene_pool))
        population.append(new_individual)

    return population


# evaluation class;
# since that a solution needs to be evaluated with respect to the problem instance
# in consideration, we created this class to store the problem instance and to
# allow the evaluation to be performed without having the problem instance at hand
class EvaluateTSM:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance

    def __call__(self, solution):
        distance = 0
        origin = None
        visited = []
        for city in solution:
            if city in visited:
                distance += 0
            if origin is None:
                origin = city
            else:
                distance += euclidean_distance(self.problem_instance[origin], self.problem_instance[city])
            origin = city
            visited.append(city)

        # voltar a cidade de início
        distance += euclidean_distance(self.problem_instance[origin], self.problem_instance[solution[0]])
        return 1 / distance * 10000


def euclidean_distance(origin, destiny):
    return math.sqrt((destiny[1] - origin[1]) ** 2 + (destiny[2] - origin[2]) ** 2)


def insertion_mutation(x, gene_pool, pmut):
    # if random >= pmut, then no mutation is performed
    if random.random() >= pmut:
        return x

    n = len(x)
    c = random.randrange(0, n)  # gene to be inserted

    value = x.pop(c)
    p = random.randrange(0, n - 1)  # insertion position
    x.insert(p, value)

    return x


def simple_inversion_mutation(x, gene_pool, pmut):
    if random.random() >= pmut:
        return x

    cutoff_lower = int(random.randrange(1, len(x)))
    cutoff_upper = int(random.randrange(cutoff_lower, len(x)))

    sublist = x[cutoff_lower: cutoff_upper]
    sublist.reverse()

    return x[0:cutoff_lower] + sublist + x[cutoff_upper:]


def position_based_crossover(x, y):
    # select amount of positions to change
    n = random.randrange(0, len(x))

    # create the random set of positions
    positions = []
    for i in range(0, n):
        position = random.randrange(0, len(x))
        if position not in positions:
            positions.append(position)

    new_y = y.copy()
    for position in positions:
        value = x[position]
        if value in new_y:
            new_y.remove(value)
            if position >= len(new_y):
                new_y.append(x[position])
            else:
                new_y.insert(position, x[position])

    new_x = x.copy()
    for position in positions:
        if y[position] in new_x:
            new_x.remove(y[position])
            new_x.insert(position, y[position])

    return new_x, new_y


def partially_mapped_crossover(x, y):
    cutoff_lower = int(random.randrange(1, len(x)))
    cutoff_upper = int(random.randrange(cutoff_lower, len(x)))

    new_x, new_y = [-1] * len(x), [-1] * len(x)
    for i in range(cutoff_lower, cutoff_upper):
        new_x[i] = y[i]
        new_y[i] = x[i]

    for i in range(0, len(x)):
        if i < cutoff_lower or i >= cutoff_upper:
            if x[i] not in new_x:
                new_x[i] = x[i]
            else:
                value = x[i]
                while new_x[i] == -1:
                    index = new_x.index(value)
                    if x[index] not in new_x:
                        new_x[i] = x[index]
                    else:
                        value = x[index]

            if y[i] not in new_y:
                new_y[i] = y[i]
            else:
                value = y[i]
                while new_y[i] == -1:
                    index = new_y.index(value)
                    if y[index] not in new_y:
                        new_y[i] = y[index]
                    else:
                        value = y[index]

    return new_x, new_y


def ranking_selection(r, population, fn_fitness):
    fitnesses = list(map(fn_fitness, population))
    players = {}
    for i in range(0, len(population)):
        players[i] = fitnesses[i]

    ranking = list(sorted(players, key=players.get))
    n = int(((1 + len(ranking)) * len(ranking)) / 2)

    sampler = weighted_sampler(ranking, ranking)
    return [population[sampler()] for i in range(r)]


def tournament_selection(r, population, fn_fitness):
    fitnesses = list(map(fn_fitness, population))

    n = len(population)
    k = 8

    players = []
    while len(players) < k:
        player = random.randrange(0, n)
        if player not in players:
            players.append(player)

    winners = []
    while len(winners) < r:
        winner = tournament_round(players, fitnesses)
        winners.append(population[winner])
    return winners


def tournament_round(players, fitnesses):
    if len(players) <= 1:
        return players[0]
    if len(players) == 2:
        if fitnesses[players[0]] > fitnesses[players[1]]:
            return players[0]
        else:
            return players[1]
    if len(players) > 1:
        return tournament_round([tournament_round(players[0: int(len(players) / 2)], fitnesses),
                                 tournament_round(players[int(len(players) / 2): int(len(players))], fitnesses)],
                                fitnesses)


def plot(instance, solution):
    plt.rcParams["figure.autolayout"] = True
    plt.grid()
    plt.title("Traveling salesman")
    for city in instance:
        plt.plot(city[1], city[2], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="red")
    origin = None
    for city in solution:
        if origin is not None:
            plt.plot([instance[origin][1], instance[city][1]],
                     [instance[origin][2], instance[city][2]],
                     color="black")
        origin = city
    plt.plot([instance[origin][1], instance[solution[0]][1]],
             [instance[origin][2], instance[solution[0]][2]],
             color="blue")

    plt.show()


eli51 = [[1, 37.0, 52.0],
         [2, 49.0, 49.0],
         [3, 52.0, 64.0],
         [4, 20.0, 26.0],
         [5, 40.0, 30.0],
         [6, 21.0, 47.0],
         [7, 17.0, 63.0],
         [8, 31.0, 62.0],
         [9, 52.0, 33.0],
         [10, 51.0, 21.0],
         [11, 42.0, 41.0],
         [12, 31.0, 32.0],
         [13, 5.0, 25.0],
         [14, 12.0, 42.0],
         [15, 36.0, 16.0],
         [16, 52.0, 41.0],
         [17, 27.0, 23.0],
         [18, 17.0, 33.0],
         [19, 13.0, 13.0],
         [20, 57.0, 58.0],
         [21, 62.0, 42.0],
         [22, 42.0, 57.0],
         [23, 16.0, 57.0],
         [24, 8.0, 52.0],
         [25, 7.0, 38.0],
         [26, 27.0, 68.0],
         [27, 30.0, 48.0],
         [28, 43.0, 67.0],
         [29, 58.0, 48.0],
         [30, 58.0, 27.0],
         [31, 37.0, 69.0],
         [32, 38.0, 46.0],
         [33, 46.0, 10.0],
         [34, 61.0, 33.0],
         [35, 62.0, 63.0],
         [36, 63.0, 69.0],
         [37, 32.0, 22.0],
         [38, 45.0, 35.0],
         [39, 59.0, 15.0],
         [40, 5.0, 6.0],
         [41, 10.0, 17.0],
         [42, 21.0, 10.0],
         [43, 5.0, 64.0],
         [44, 30.0, 15.0],
         [45, 39.0, 10.0],
         [46, 32.0, 39.0],
         [47, 25.0, 32.0],
         [48, 25.0, 55.0],
         [49, 48.0, 28.0],
         [50, 56.0, 37.0],
         [51, 30.0, 40.0]]

berlin52 = [[1, 565.0, 575.0],
            [2, 25.0, 185.0],
            [3, 345.0, 750.0],
            [4, 945.0, 685.0],
            [5, 845.0, 655.0],
            [6, 880.0, 660.0],
            [7, 25.0, 230.0],
            [8, 525.0, 1000.0],
            [9, 580.0, 1175.0],
            [10, 650.0, 1130.0],
            [11, 1605.0, 620.0],
            [12, 1220.0, 580.0],
            [13, 1465.0, 200.0],
            [14, 1530.0, 5.0],
            [15, 845.0, 680.0],
            [16, 725.0, 370.0],
            [17, 145.0, 665.0],
            [18, 415.0, 635.0],
            [19, 510.0, 875.0],
            [20, 560.0, 365.0],
            [21, 300.0, 465.0],
            [22, 520.0, 585.0],
            [23, 480.0, 415.0],
            [24, 835.0, 625.0],
            [25, 975.0, 580.0],
            [26, 1215.0, 245.0],
            [27, 1320.0, 315.0],
            [28, 1250.0, 400.0],
            [29, 660.0, 180.0],
            [30, 410.0, 250.0],
            [31, 420.0, 555.0],
            [32, 575.0, 665.0],
            [33, 1150.0, 1160.0],
            [34, 700.0, 580.0],
            [35, 685.0, 595.0],
            [36, 685.0, 610.0],
            [37, 770.0, 610.0],
            [38, 795.0, 645.0],
            [39, 720.0, 635.0],
            [40, 760.0, 650.0],
            [41, 475.0, 960.0],
            [42, 95.0, 260.0],
            [43, 875.0, 920.0],
            [44, 700.0, 500.0],
            [45, 555.0, 815.0],
            [46, 830.0, 485.0],
            [47, 1170.0, 65.0],
            [48, 830.0, 610.0],
            [49, 605.0, 625.0],
            [50, 595.0, 360.0],
            [51, 1340.0, 725.0],
            [52, 1740.0, 245.0]]

rat99 = [[1, 6.0, 4.0],
         [2, 15.0, 15.0],
         [3, 24.0, 18.0],
         [4, 33.0, 12.0],
         [5, 48.0, 12.0],
         [6, 57.0, 14.0],
         [7, 67.0, 10.0],
         [8, 77.0, 10.0],
         [9, 86.0, 15.0],
         [10, 6.0, 21.0],
         [11, 17.0, 26.0],
         [12, 23.0, 25.0],
         [13, 32.0, 35.0],
         [14, 43.0, 23.0],
         [15, 55.0, 35.0],
         [16, 65.0, 36.0],
         [17, 78.0, 39.0],
         [18, 87.0, 35.0],
         [19, 3.0, 53.0],
         [20, 12.0, 44.0],
         [21, 28.0, 53.0],
         [22, 33.0, 49.0],
         [23, 47.0, 46.0],
         [24, 55.0, 52.0],
         [25, 64.0, 50.0],
         [26, 71.0, 57.0],
         [27, 87.0, 57.0],
         [28, 4.0, 72.0],
         [29, 15.0, 78.0],
         [30, 22.0, 70.0],
         [31, 34.0, 71.0],
         [32, 42.0, 79.0],
         [33, 54.0, 77.0],
         [34, 66.0, 79.0],
         [35, 78.0, 67.0],
         [36, 87.0, 73.0],
         [37, 7.0, 81.0],
         [38, 17.0, 95.0],
         [39, 26.0, 98.0],
         [40, 32.0, 97.0],
         [41, 43.0, 88.0],
         [42, 57.0, 89.0],
         [43, 64.0, 85.0],
         [44, 78.0, 83.0],
         [45, 83.0, 98.0],
         [46, 5.0, 109.0],
         [47, 13.0, 111.0],
         [48, 25.0, 102.0],
         [49, 38.0, 119.0],
         [50, 46.0, 107.0],
         [51, 58.0, 110.0],
         [52, 67.0, 110.0],
         [53, 74.0, 113.0],
         [54, 88.0, 110.0],
         [55, 2.0, 124.0],
         [56, 17.0, 134.0],
         [57, 23.0, 129.0],
         [58, 36.0, 131.0],
         [59, 42.0, 137.0],
         [60, 53.0, 123.0],
         [61, 63.0, 135.0],
         [62, 72.0, 134.0],
         [63, 87.0, 129.0],
         [64, 2.0, 146.0],
         [65, 16.0, 147.0],
         [66, 25.0, 153.0],
         [67, 38.0, 155.0],
         [68, 42.0, 158.0],
         [69, 57.0, 154.0],
         [70, 66.0, 151.0],
         [71, 73.0, 151.0],
         [72, 86.0, 149.0],
         [73, 5.0, 177.0],
         [74, 13.0, 162.0],
         [75, 25.0, 169.0],
         [76, 35.0, 177.0],
         [77, 46.0, 172.0],
         [78, 54.0, 166.0],
         [79, 65.0, 174.0],
         [80, 73.0, 161.0],
         [81, 86.0, 162.0],
         [82, 2.0, 195.0],
         [83, 14.0, 196.0],
         [84, 28.0, 189.0],
         [85, 38.0, 187.0],
         [86, 46.0, 195.0],
         [87, 57.0, 194.0],
         [88, 63.0, 188.0],
         [89, 77.0, 193.0],
         [90, 85.0, 194.0],
         [91, 8.0, 211.0],
         [92, 12.0, 217.0],
         [93, 22.0, 210.0],
         [94, 34.0, 216.0],
         [95, 47.0, 203.0],
         [96, 58.0, 213.0],
         [97, 66.0, 206.0],
         [98, 78.0, 210.0],
         [99, 85.0, 204.0]]

pr152 = [[1, 2100.0, 1850.0],
         [2, 2100.0, 3000.0],
         [3, 2100.0, 4400.0],
         [4, 2100.0, 5550.0],
         [5, 2100.0, 6950.0],
         [6, 2100.0, 8100.0],
         [7, 2100.0, 9500.0],
         [8, 2100.0, 10650.0],
         [9, 2348.0, 11205.0],
         [10, 2350.0, 10050.0],
         [11, 2348.0, 8655.0],
         [12, 2350.0, 7500.0],
         [13, 2348.0, 6105.0],
         [14, 2350.0, 4950.0],
         [15, 2348.0, 3555.0],
         [16, 2350.0, 2400.0],
         [17, 2625.0, 11175.0],
         [18, 2775.0, 10995.0],
         [19, 2625.0, 10025.0],
         [20, 2634.0, 9748.0],
         [21, 2607.0, 9831.0],
         [22, 2625.0, 8625.0],
         [23, 2775.0, 8445.0],
         [24, 2625.0, 7475.0],
         [25, 2607.0, 7281.0],
         [26, 2634.0, 7198.0],
         [27, 2625.0, 6075.0],
         [28, 2775.0, 5895.0],
         [29, 2625.0, 4925.0],
         [30, 2607.0, 4731.0],
         [31, 2634.0, 4648.0],
         [32, 2625.0, 3525.0],
         [33, 2775.0, 3345.0],
         [34, 2625.0, 2375.0],
         [35, 2634.0, 2098.0],
         [36, 2607.0, 2181.0],
         [37, 2825.0, 3025.0],
         [38, 2825.0, 5575.0],
         [39, 2825.0, 8125.0],
         [40, 2825.0, 10675.0],
         [41, 8349.0, 10106.0],
         [42, 8353.0, 9397.0],
         [43, 8349.0, 7556.0],
         [44, 8353.0, 6847.0],
         [45, 8349.0, 5006.0],
         [46, 8353.0, 4297.0],
         [47, 8349.0, 2456.0],
         [48, 8353.0, 1747.0],
         [49, 8474.0, 1777.0],
         [50, 8576.0, 1803.0],
         [51, 8575.0, 2325.0],
         [52, 8474.0, 4327.0],
         [53, 8576.0, 4353.0],
         [54, 8575.0, 4875.0],
         [55, 8474.0, 6877.0],
         [56, 8576.0, 6903.0],
         [57, 8575.0, 7425.0],
         [58, 8474.0, 9427.0],
         [59, 8576.0, 9453.0],
         [60, 8575.0, 9975.0],
         [61, 8625.0, 9875.0],
         [62, 8675.0, 9675.0],
         [63, 8675.0, 9525.0],
         [64, 8669.0, 9450.0],
         [65, 8625.0, 7325.0],
         [66, 8675.0, 7125.0],
         [67, 8675.0, 6975.0],
         [68, 8669.0, 6900.0],
         [69, 8625.0, 4775.0],
         [70, 8675.0, 4575.0],
         [71, 8675.0, 4425.0],
         [72, 8669.0, 4350.0],
         [73, 8625.0, 2225.0],
         [74, 8675.0, 1875.0],
         [75, 8675.0, 2025.0],
         [76, 8669.0, 1800.0],
         [77, 9250.0, 1850.0],
         [78, 9250.0, 3000.0],
         [79, 9250.0, 4400.0],
         [80, 9250.0, 5550.0],
         [81, 9250.0, 6950.0],
         [82, 9250.0, 8100.0],
         [83, 9250.0, 9500.0],
         [84, 9250.0, 10650.0],
         [85, 9498.0, 11205.0],
         [86, 9500.0, 10050.0],
         [87, 9498.0, 8655.0],
         [88, 9500.0, 7500.0],
         [89, 9498.0, 6105.0],
         [90, 9500.0, 4950.0],
         [91, 9498.0, 3555.0],
         [92, 9500.0, 2400.0],
         [93, 9784.0, 2098.0],
         [94, 9757.0, 2181.0],
         [95, 9775.0, 2375.0],
         [96, 9775.0, 3525.0],
         [97, 9784.0, 4648.0],
         [98, 9757.0, 4731.0],
         [99, 9775.0, 4925.0],
         [100, 9775.0, 6075.0],
         [101, 9784.0, 7198.0],
         [102, 9757.0, 7281.0],
         [103, 9775.0, 7475.0],
         [104, 9775.0, 8625.0],
         [105, 9784.0, 9748.0],
         [106, 9757.0, 9831.0],
         [107, 9775.0, 10025.0],
         [108, 9775.0, 11175.0],
         [109, 9925.0, 10995.0],
         [110, 9975.0, 10675.0],
         [111, 9925.0, 8445.0],
         [112, 9975.0, 8125.0],
         [113, 9925.0, 5895.0],
         [114, 9975.0, 5575.0],
         [115, 9925.0, 3345.0],
         [116, 9975.0, 3025.0],
         [117, 15499.0, 10106.0],
         [118, 15503.0, 9397.0],
         [119, 15499.0, 7556.0],
         [120, 15503.0, 6847.0],
         [121, 15499.0, 5006.0],
         [122, 15503.0, 4297.0],
         [123, 15499.0, 2456.0],
         [124, 15503.0, 1747.0],
         [125, 15624.0, 1777.0],
         [126, 15726.0, 1803.0],
         [127, 15775.0, 2225.0],
         [128, 15725.0, 2325.0],
         [129, 15624.0, 4327.0],
         [130, 15726.0, 4353.0],
         [131, 15775.0, 4775.0],
         [132, 15725.0, 4875.0],
         [133, 15624.0, 6877.0],
         [134, 15726.0, 6903.0],
         [135, 15775.0, 7325.0],
         [136, 15725.0, 7425.0],
         [137, 15624.0, 9427.0],
         [138, 15726.0, 9453.0],
         [139, 15775.0, 9875.0],
         [140, 15725.0, 9975.0],
         [141, 15825.0, 9675.0],
         [142, 15825.0, 9525.0],
         [143, 15819.0, 9450.0],
         [144, 15825.0, 7125.0],
         [145, 15825.0, 6975.0],
         [146, 15819.0, 6900.0],
         [147, 15825.0, 4575.0],
         [148, 15825.0, 4425.0],
         [149, 15819.0, 4350.0],
         [150, 15825.0, 1875.0],
         [151, 15825.0, 2025.0],
         [152, 15819.0, 1800.0]]


instances = [eli51, berlin52, rat99, pr152]

for instance in instances:
    fn_fitness = EvaluateTSM(instance)

    individual_length = len(fn_fitness.problem_instance)

    possible_values = range(0, individual_length)

    population_size = 8

    threshold = fn_fitness(possible_values) * 10
    print(threshold)
    # initial population
    population = init_population(population_size, possible_values, individual_length)

    start = time.perf_counter_ns()
    # run the algorithm
    solution = genetic_algorithm(population, fn_fitness, gene_pool=possible_values, ngen=3000,
                                 fn_thres=threshold, mutation=simple_inversion_mutation, selection=tournament_selection,
                                 recombine=partially_mapped_crossover, pmut=10/100)
    end = time.perf_counter_ns()
    # print the results
    print('Resulting solution:', solution)
    print('Value of resulting solution', fn_fitness(solution))
    print('Execution time:', str((end - start) / 1000000000))
    visited = []
    for city in solution:
        if city in visited:
            print("repetida:", city)
        else:
            visited.append(city)
    plot(fn_fitness.problem_instance, solution)
