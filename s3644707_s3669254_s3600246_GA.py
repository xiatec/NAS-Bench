from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys, time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

DNA_SIZE = 26            # DNA length
POP_SIZE = 10          # population size
CROSS_RATE = 0.65        # mating probability (DNA crossover)
MUTATION_RATE = 0.03    # mutation probability

#get the seed inorder to reproduce the same results
np.random.seed(42)


def crossover(p1, p2):
   if(np.random.uniform(0,1) < CROSS_RATE):
        for i in range(len(p1)) :
            if np.random.uniform(0,1) < 0.5:
                t = p1[i]
                p1[i] = p2[i]
                p2[i] = t

def mutate(p):
    for i in range(len(p)) :
        if np.random.uniform(0,1) < MUTATION_RATE:
            if i >= 0 and i <= 20: p[i] = 1 - p[i]
            if i > 20: p[i] = int(np.random.choice([0,1,2],1))


def select(parent, parent_f):
    f_min = min(parent_f)
    f_sum = sum(parent_f) - (f_min - 0.001) * len(parent_f)

    rw = [(parent_f[0] - f_min + 0.001) / f_sum]
    for i in range(1, len(parent_f)):
        rw.append(rw[i - 1] + (parent_f[i] - f_min + 0.001) / f_sum)

    select_parent = []
    for i in range(len(parent)):
        r = np.random.uniform(0, 1)
        index = 0
        while (r > rw[index]):
            index = index + 1

        select_parent.append(parent[index].copy())
    return select_parent



def init(POP_SIZE):
    parent = []
    for i in range(POP_SIZE):
        parent.append(ga_sampling_model(nas_ioh.nasbench))
    parent = np.array(parent)
    return parent



def ga_sampling_model(bench):
    while True:
        matrix = np.random.choice([0, 1], size=(7, 7))
        matrix = np.triu(matrix, 1)
        # array([[0, 1, 0, 0, 1, 0, 0],
        # [0, 0, 0, 1, 1, 1, 0],
        # [0, 0, 0, 1, 1, 0, 1],
        # [0, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0]])
        operations = [CONV1X1, CONV3X3, MAXPOOL3X3]
        ops = np.random.choice(operations, 7)
        ops[0] = INPUT
        ops[6] = OUTPUT
        model_spec = api.ModelSpec(
            # Adjacency matrix of the module
            matrix=matrix,
            # Operations at the vertices of the module, matches order of matrix
            ops=list(ops))
        # check if the model is valid
        if bench.is_valid(model_spec):
            break
    x = np.empty(26, dtype=int)
    index = 0
    for i in range(7):
        for j in range(i + 1, 7):
            x[index] = matrix[i][j]
            index += 1
    for i in range(1, 6):
        if ops[i] == CONV1X1:
            x[index] = 1
            index += 1
        elif ops[i] == CONV3X3:
            x[index] = 2
            index += 1
        elif ops[i] == MAXPOOL3X3:
            x[index] = 0
            index += 1
    return x


def genetic_algorithm():
    f_opt = sys.float_info.min
    optimum = 1
    #Initialize
    budget = 5000
    parent = []
    parent_f = []

    parent = init(POP_SIZE)
    for i in range(POP_SIZE):
        parent_f.append(nas_ioh.f(parent[i]))
        budget = budget - 1
        if parent_f[i] > f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()

    while (f_opt < optimum and budget >= POP_SIZE):
        offspring = select(parent,parent_f)

        for i in range(0,POP_SIZE - (POP_SIZE%2),2) :
            crossover(offspring[i], offspring[i+1])


        for i in range(POP_SIZE):
            mutate(offspring[i])

        parent = offspring.copy()
        for i in range(POP_SIZE) :
            parent_f[i] = nas_ioh.f(parent[i])
            budget = budget - 1
            if parent_f[i] > f_opt:
                    f_opt = parent_f[i]
                    x_opt = parent[i].copy()
            if f_opt >= optimum:
                break
    print("best x:", x_opt, ", f :", f_opt)
    nas_ioh.f.reset()
    return f_opt, x_opt


def main(argv):
    del argv  # Unused
    for r in range(nas_ioh.runs):  # we execute the algorithm with 20 independent runs.
        genetic_algorithm()

if __name__ == '__main__':
    start = time.time()
    app.run(main)
    end = time.time()
    print("The program takes %s seconds" % (end - start))