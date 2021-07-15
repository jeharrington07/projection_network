import numpy as np
import torch as torch

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer

from random_network import create_random_dense, random_fake

def fitness_func(solution, projector):

    #dummy fitness function
    #TODO: replace with rastrigin funciton
    fitness = 1
    b1 = 1
    b2 = 1
    return fitness, b1, b2

#TODO: Put the below inside a main function, set up command line args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#edit these fields according to the problem
archiveSize = [10, 10]
xrange = (-1, 1)
yrange = (-1, 1)
n_iter = 10
n_features = 10

#setting up elements for cma-es (for cma-me, use ImprovementEmitter)
archive = GridArchive(archiveSize, [xrange, yrange])
emitters = [OptimizingEmitter(archive, [0.0]*n_features, .1)]
optimizer = Optimizer(archive, emitters)

#initialize projection network
#TODO: save weights
projector = create_random_dense(100, 200, device, 3)

with torch.no_grad():

    for itr in range(n_iter):
        solutions = optimizer.ask()
        results = [fitness_func(solution, projector) for solution in solutions]
        objectives = [result[0] for result in results]
        bcs = [result[1:] for result in results]

        optimizer.tell(objectives, bcs)