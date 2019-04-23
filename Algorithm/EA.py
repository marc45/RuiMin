import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10  # DNA length
POP_SIZE = 100  # population size
CROSS_RATE = 0.8  # mating probability(DNA crossover)
MUTATION_RATE = 0.003  # mutation probability
N_GENRATIONS = 200
X_BOUND = [0, 5]  # x upper and lower bounds


# function
def F(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x  # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)


# 转换成01
def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1) * X_BOUND[1]


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


# 交叉
def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
    return parent


# 变异
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # initialize the pop DNA
plt.ion()  # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))


for _ in range(N_GENRATIONS):
    F_value = F(translateDNA(pop))
    # plt.cla()

    # something about plotting
    sca = plt.scatter(translateDNA(pop), F_value, s=200, lw=0, c='red', alpha=0.5)
    if sca in globals():
        sca.remove()
    plt.pause(0.05)

    fitness = get_fitness(F_value)
    print('most fitted DNA: ', pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child  # parent is replaced by its child

plt.ioff()
plt.show()
