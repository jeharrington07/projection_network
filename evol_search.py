import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer


def fitness_func(x):
    fitness = 1
    b1 = 1
    b2 = 1
    return fitness, b1, b2


#edit these fields according to the problem
archiveSize = [10, 10]
xrange = (-1, 1)
yrange = (-1, 1)
n_iter = 10
n_features = 10


archive = GridArchive(archiveSize, [xrange, yrange])
emitters = [ImprovementEmitter(archive, [0.0]*n_features, .1)]
optimizer = Optimizer(archive, emitters)

for itr in range(n_iter):
    solutions = optimizer.ask()

    results = [fitness_func(solution) for solution in solutions]
    objectives = [result[0] for result in results]
    bcs = [result[1:] for result in results]

    optimizer.tell(objectives, bcs)