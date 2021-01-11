from control import TransferFunction, feedback, step_info, step_response, series
from random import uniform, sample, randint, random
import matplotlib.pyplot as pyot

# Global Variables
IND = 50  # Number of individuals
GENE = 150  # Number of Generations
CROSS = 7  # Crossover odds (out of 10)
MUT = 25  # Mutation odds (out of 100)
POOL = 20  # Mating pool size
KEEP = 2  # Number of individuals carried between generations

# The amount each value needs to be stretched to cover the entire range allotted to it
KpFactor = 100*2048/1600
TiFactor = 100*1024/837
TdFactor = 100*256/211


def generate():
    Kp = round(uniform(2, 18), 2)
    Ti = round(uniform(1.05, 9.42), 2)
    Td = round(uniform(0.26, 2.37), 2)
    return Kp, Ti, Td


# converts floating point input parameters to binary
def f2b(value):
    Kp = bin(int((value[0]-2.00)*KpFactor)).lstrip("0b").zfill(11)
    Ti = bin(int((value[1]-1.05)*TiFactor)).lstrip("0b").zfill(10)
    Td = bin(int((value[2]-0.26)*TdFactor)).lstrip("0b").zfill(8)
    return Kp, Ti, Td


# converts the genetic sequence back to floating point
def b2f(value):
    Kp = round(int(value[0], 2)/KpFactor+2, 2)
    Ti = round(int(value[1], 2)/TiFactor+1.05, 2)
    Td = round(int(value[2], 2)/TdFactor+0.26, 2)
    return Kp, Ti, Td


def fitness(result):
    return result[0]


def mate(first, second):
    if randint(1, 10) < CROSS:
        first, second = crossover(first, second)
    if randint(1, 100) < MUT:
        first = mutate(first)
    if randint(1, 100) < MUT:
        second = mutate(second)
    return first, second


# inversion mutation should be more drastic than before
def mutate(result):
    merge = result[0] + result[1] + result[2]
    indices = sample(range(0, len(merge)), 2)
    lower = min(indices[0], indices[1])
    higher = max(indices[0], indices[1])
    mutated = merge[0:lower] + merge[higher:lower:-1] + merge[higher:]

    return mutated[0:11], mutated[11:21], mutated[21:]

    # single swap mutation, might want something more drastic, this rarely does anything
    # merge = result[0] + result[1] + result[2]
    # indices = sample(range(0, len(merge)), 2)
    # lower = min(indices[0], indices[1])
    # higher = max(indices[0], indices[1])
    # mutated = merge[0:lower] + merge[higher] + merge[lower+1:higher-1] + merge[lower] + merge[higher:]
    #
    # return mutated[0:11], mutated[11:21], mutated[21:]


# uniform crossover
def crossover(first, second):
    merge1 = first[0] + first[1] + first[2]
    merge2 = second[0] + second[1] + second[2]
    length = len(merge1)
    splice1 = ""
    splice2 = ""
    for i in range(length):
        if randint(0, 1) == 1:
            splice1 = splice1 + merge1[i]
            splice2 = splice2 + merge2[i]
        else:
            splice1 = splice1 + merge2[i]
            splice2 = splice2 + merge1[i]

    returned1 = (splice1[0:11], splice1[11:21], splice1[21:])
    returned2 = (splice2[0:11], splice2[11:21], splice2[21:])

    return returned1, returned2


def weigh_random(prob):
    ran = random()
    total = 0
    for index, value in enumerate(prob):
        total += value
        if ran <= total:
            return index
    return -1


# this adaptation of the perfFNC function was taken from the slack by Nick Shields
def q2_perfFNC(Kp, Ti, Td):
    G = Kp * TransferFunction([Ti * Td, Ti, 1], [Ti, 0])
    F = TransferFunction(1, [1, 6, 11, 6, 0])
    sys = feedback(series(G, F), 1)
    sysinf = step_info(sys)

    t = []
    i = 0
    while i < 100:
        t.append(i)
        i += 0.01

    T, y = step_response(sys, T=t)

    ISE = sum((y - 1) ** 2)
    t_r = sysinf['RiseTime']
    t_s = sysinf['SettlingTime']
    M_p = sysinf['Overshoot']

    return ISE, t_r, t_s, M_p


if __name__ == '__main__':

    pop = []
    fit = []
    best = []

    for i in range(IND):
        pop.append(generate())
        # this try/except arrangement means unstable systems are never in contention for the mating pool
        try:
            ind_fit = q2_perfFNC(pop[i][0], pop[i][1], pop[i][2])[0]
            fit.append((ind_fit, i))
        except IndexError:
            pass

    fit.sort()  # sort the fitness values, lowest (i.e. best) to highest

    # main generational loop
    for g in range(GENE):
        numerators = []
        denominator = 0
        new_pop = []
        new_fit = []
        remainder = int((IND - KEEP) / 2)
        # temp = fit[0:POOL]
        for h in range(KEEP):
            new_pop.append(pop[fit[h][1]])
            new_fit.append((fit[h][0], h))
        # for i in range(POOL):
        #     mating_pool.append(pop[temp[i][1]])

        # pop.clear()
        # fit.clear()

        # for j in range(KEEP):
        #     fit.append((best_fit[j], j))
        #     pop.append(mating_pool[j])

        for i in range(len(fit)):
            denominator += fit[i][0]

        for j in range(len(fit)):
            numerators.append(fit[j][0]/denominator)

        for k in range(remainder):
            parent1 = weigh_random(numerators)
            parent2 = weigh_random(numerators)
            # ensure the same two cannot be selected
            while parent1 == parent2:
                parent2 = weigh_random(numerators)
            child1, child2 = mate(f2b(pop[parent1]), f2b(pop[parent2]))
            child1 = b2f(child1)
            child2 = b2f(child2)

            new_pop.append(child1)
            try:
                ind_fit = q2_perfFNC(child1[0], child1[1], child1[2])[0]
                new_fit.append((ind_fit, k+KEEP))
            except IndexError:
                pass

            new_pop.append(child2)
            try:
                ind_fit = q2_perfFNC(child2[0], child2[1], child2[2])[0]
                new_fit.append((ind_fit, k+KEEP+1))
            except IndexError:
                pass

        new_fit.sort()
        print(new_fit[0][0])
        fit = new_fit
        pop = new_pop
        best.append(fit[0][0])

    pyot.plot(range(0, GENE), best)
    pyot.xlabel("Generation")
    pyot.ylabel("Best Fitness")
    pyot.show()


# # main generational loop
#     for g in range(GENE):
#         denominator = 0
#         best_fit = []  # temp array for best fitness to be preserved across generations
#         mating_pool = []  # array of mating individuals
#         temp = fit[0:POOL]  # temp array of most fit individuals from previous generation
#         for h in range(KEEP):
#             best_fit.append(fit[h][0])
#         for i in range(POOL):
#             mating_pool.append(pop[temp[i][1]])
#
#         pop.clear()
#         fit.clear()
#
#         for j in range(KEEP):
#             fit.append((best_fit[j], j))
#             pop.append(mating_pool[j])
#
#         remainder = int((IND-KEEP)/2)
#
#         for k in range(remainder):
#             partners = sample(range(0, POOL), 2)
#             child1, child2 = mate(f2b(mating_pool[partners[0]]), f2b(mating_pool[partners[1]]))
#             child1 = b2f(child1)
#             child2 = b2f(child2)
#
#             pop.append(child1)
#             try:
#                 ind_fit = q2_perfFNC(child1[0], child1[1], child1[2])[0]
#                 fit.append((ind_fit, k+KEEP))
#             except IndexError:
#                 pass
#
#             pop.append(child2)
#             try:
#                 ind_fit = q2_perfFNC(child2[0], child2[1], child2[2])[0]
#                 fit.append((ind_fit, k+KEEP+1))
#             except IndexError:
#                 pass
#
#         fit.sort()
#         print(fit[0][0])
#         best.append(fit[0][0])
#
#     pyot.plot(range(0, GENE), best)
#     pyot.xlabel("Generation")
#     pyot.ylabel("Best Fitness")
#     pyot.show()
