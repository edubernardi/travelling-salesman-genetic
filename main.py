import math
import random as random
import bisect
from matplotlib import pyplot as plt


def genetic_algorithm(population, fn_fitness, gene_pool, mutation, recombine, selection, fn_thres=None, ngen=1000,
                      pmut=0.1):
    # for each generation
    for i in range(ngen):

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


# genetic operator for selection of individuals;
# this function implements roulette wheel selection, where individuals with
# higher fitness are selected with higher probability
def select(r, population, fn_fitness):
    fitnesses = map(fn_fitness, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]


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
        # each individual is represented as an array with size state_length,
        # where each position contains a value from gene_pool selected at random
        # new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        new_individual = random.sample(list(gene_pool), len(gene_pool))
        population.append(new_individual)

    return population


# evaluation class;
# since that a solution needs to be evaluated with respect to the problem instance
# in consideration, we created this class to store the problem instance and to
# allow the evaluation to be performed without having the problem instance at hand
class EvaluateTSM:
    # during initialization, store the problem instance
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance

    # compute the value of the received solution
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

        # voltar
        distance += euclidean_distance(self.problem_instance[origin], self.problem_instance[solution[0]])
        return 100000 - distance


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

    if len(x[0:cutoff_lower] + sublist + x[cutoff_upper:]) > 51:
        print("Deu ruim")
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


# def stochastic_universal_sampling(r, population, fn_fitness):
#    fitnesses = map(fn_fitness, population)
#    sampler = weighted_sampler(population, fitnesses)
#    return [sampler() for i in range(r)]


def tournament(r, population, fn_fitness):
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
    for city in tsm_instance:
        plt.plot(city[1], city[2], marker="o", markersize=3, markeredgecolor="red", markerfacecolor="red")
    origin = None
    for city in solution:
        if origin is not None:
            plt.plot([tsm_instance[origin][1], tsm_instance[city][1]],
                     [tsm_instance[origin][2], tsm_instance[city][2]],
                     color="black")
        origin = city
    plt.plot([tsm_instance[origin][1], tsm_instance[solution[0]][1]],
             [tsm_instance[origin][2], tsm_instance[solution[0]][2]],
             color="blue")

    plt.show()


tsm_instance = [[1, 37.0, 52.0],
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

fn_fitness = EvaluateTSM(tsm_instance)

individual_length = len(fn_fitness.problem_instance)

possible_values = range(0, individual_length)

population_size = 15

# initial population
population = init_population(population_size, possible_values, individual_length)

# run the algorithm
solution = genetic_algorithm(population, fn_fitness, gene_pool=possible_values, ngen=1000,
                             fn_thres=99500, mutation=simple_inversion_mutation, selection=tournament,
                             recombine=partially_mapped_crossover)

# print the results
print('Resulting solution: %s' % solution)
print('Value of resulting solution: %d' % fn_fitness(solution))
visited = []
for city in solution:
    if city in visited:
        print("repetida:", city)
    else:
        visited.append(city)

plot(tsm_instance, solution)
