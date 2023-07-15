
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh
np.random.seed(42) 


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def random_sampling_model(bench):
  while True:
    matrix = np.random.choice([0, 1], size=(7, 7))
    matrix = np.triu(matrix, 1)

    operations = [CONV1X1,CONV3X3,MAXPOOL3X3]
    ops = np.random.choice(operations,7)
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
  
  x = np.empty(26,dtype=int)
  index = 0
  for i in range(7):
    for j in range(i+1,7):
      x[index] = matrix[i][j]
      index += 1
  for i in range (1,6):
    if ops[i] == CONV1X1:
      x[index] = 1
      index+=1
    elif ops[i] == CONV3X3:
      x[index] = 2
      index+=1
    elif ops[i] == MAXPOOL3X3:
      x[index] = 0
      index+=1
  return x

def init(mu):
    parent = []
    parent_sigma = []
    for i in range(mu):
        parent.append(random_sampling_model(nas_ioh.nasbench))
        parent_sigma.append(0.5)
    parent = np.array(parent)
    return parent,parent_sigma

def dis01(x):
    if abs(x-0)>=abs(x-1):
        x = 0
    else:
        x = 1
    return x

def dis012(x):
    if abs(x-0)<=abs(x-1) and abs(x-0)<=abs(x-2) :
        if np.random.uniform(0,1) < 0.5:
            x = 1
        else:
            x = 2
    elif abs(x-1)<=abs(x-0) and abs(x-1)<=abs(x-2) :
        if np.random.uniform(0,1) < 0.5:
            x = 2
        else:
            x = 0
    else:
        if np.random.uniform(0,1) < 0.5:
            x = 0
        else:
            x = 1
    return x

def mutation(parent, parent_sigma,tau):
    for i in range(len(parent)):
        parent_sigma[i] = parent_sigma[i] * np.exp(np.random.normal(0,tau))

        for j in range(21):
            parent[i][j] = parent[i][j] + np.random.normal(0,parent_sigma[i])
            parent[i][j] = dis01(parent[i][j])
  
        for j in range(21,26):
            parent[i][j] = parent[i][j] + np.random.normal(0,parent_sigma[i])
            parent[i][j] = dis012(parent[i][j])

def recombination(parent,parent_sigma):
    [p1,p2] = np.random.choice(len(parent),2,replace = True)
    offspring = (parent[p1] + parent[p2])/2
    for i in range(len(offspring)):
        if offspring[i] == 0.5:
            if np.random.uniform(0,1) < 0.5:
                offspring[i] = 0
            else:
                offspring[i] = 1
        elif  offspring[i] == 1.5:
            if np.random.uniform(0,1) < 0.5:
                offspring[i] = 2
            else:
                offspring[i] = 1
    # offspring = (parent[p1] + parent[p2])/2
    sigma = (parent_sigma[p1] + parent_sigma[p2])/2 
    return offspring,sigma  

def es():
    #   del argv# Unused

    mu_ = 50
    lambda_ = 150
    tau =  1/np.sqrt(26)
    parent = []
    parent_sigma = []
    f_opt = sys.float_info.min
    budget = 5000
    optimum = 1
    # init
    parent,parent_sigma = init(mu_)
    # print(parent)
    parent_f = []
    # print(parent)

    for i in range(mu_):
        parent_f.append(nas_ioh.f(parent[i]))
        # print(r,i,nas_ioh.f(parent[i]))
        budget = budget - 1
        if parent_f[i] > f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()
            
    while (f_opt < optimum and budget >= lambda_):        
        offspring = []
        offspring_sigma = []
        offspring_f = []

        # Recombination
        for i in range(lambda_):
            o, s = recombination(parent,parent_sigma)

            offspring.append(o)
            offspring_sigma.append(s)
        offspring = np.array(offspring)
        # Mutation
        mutation(offspring,offspring_sigma,tau)
            # print(offspring)
            #Evaluation
        for i in range(lambda_) : 
            offspring_f.append(nas_ioh.f(offspring[i].astype(int)))
            # print(offspring_f) # [0.0]
            budget = budget - 1
            if offspring_f[i] > f_opt:
                f_opt = offspring_f[i]
                x_opt = offspring[i].copy()
                    # print(f_opt)
            if (f_opt >= optimum) | (budget <= 0):
                break

            # Selection····
        # print(offspring_f)
        
        rank = np.argsort(offspring_f) #index
        # print(rank)
        parent = []
        parent_sigma = []
        parent_f = []
        i = 0
        while ((i < lambda_) & (len(parent) < mu_)):
            if (rank[i] >= mu_):
                parent.append(offspring[i])
                parent_f.append(offspring_f[i])
                parent_sigma.append(offspring_sigma[i])
            i = i + 1
    print("best x:", x_opt,", f :",f_opt)
    nas_ioh.f.reset() # Note that you must run the code after each independent run.
    return f_opt,x_opt

def main(argv):
    f_sum = 0
    for _ in range(20):
        f_opt,x_opt = es()    
        f_sum += f_opt
    print('average fitness:',f_sum/20)
# If you are passing command line flags to modify the default config values, you
# must use app.run(main)

        
if __name__ == '__main__':
  start = time.time()
  app.run(main)
  end = time.time()
  print("The program takes %s seconds" % (end-start))
