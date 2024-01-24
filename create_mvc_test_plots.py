import matplotlib.pyplot as plt
import numpy as np
import json

problem_type = 'min_cover'
training_graph_size = 20

with open('data/{}_test_data{}.txt'.format(problem_type, training_graph_size)) as f:
    solution_data = json.load(f)

with open('data/{}_test_times{}.txt'.format(problem_type, training_graph_size)) as f:
    solution_times = json.load(f)

algorithms = list(solution_data.keys())
vert_counts = list(solution_data[algorithms[0]].keys())

x = ["|V| = {}".format(i) for i in vert_counts]
ind = np.arange(len(x))
width = 0.12

# Want to get mean from every non-single graph test
for alg in solution_data:
    # If the solution_data's first graph size's first item is a list, then we want to change it to the mean solution
    if type(solution_data[alg]['20'][0]) == list:
        for vert in solution_data[alg]:
            for j in range(len(solution_data[alg][vert])):
                solution_data[alg][vert][j] = np.mean(solution_data[alg][vert][j])

for alg in solution_times:
    if type(solution_times[alg]['20'][0]) == list:
        for vert in solution_times[alg]:
            for j in range(len(solution_times[alg][vert])):
                solution_times[alg][vert][j] = np.mean(solution_times[alg][vert][j])

# Now we create approximation ratios with respect to the cplex solutions (always first in dict)
if 'cplex' in solution_data:
    cplex_solutions = solution_data['cplex']
else:
    cplex_solutions = None

if cplex_solutions != None:
    for alg in solution_data:
        for vert in solution_data[alg]:
            if problem_type == 'max_ind_set' or problem_type == 'max_cut':
                solution_data[alg][vert] = list(np.divide(cplex_solutions[vert], solution_data[alg][vert]))
            elif problem_type == 'min_cover' or problem_type == 'min_cut':
                solution_data[alg][vert] = list(np.divide(solution_data[alg][vert], cplex_solutions[vert]))

# Now we get the average approximation ratio or solution for every sub-list
for alg in solution_data:
    for vert in solution_data[alg]:
        solution_data[alg][vert] = np.mean(solution_data[alg][vert])

for alg in solution_times:
    for vert in solution_times[alg]:
        solution_times[alg][vert] = np.mean(solution_times[alg][vert])

bars = []
plt.figure(figsize=(20, 10))

for i, alg in enumerate(solution_data):
    next_data = []
    for vert in solution_data[alg]:
        next_data.append(solution_data[alg][vert])

    bars.append(plt.bar(ind + width * i, next_data, width))

for i, alg in enumerate(solution_data):
    for j, vert in enumerate(solution_data[alg]):
        plt.text(j + i * width, solution_data[alg][vert], round(solution_data[alg][vert], 2), ha='center', fontsize=8)


plt.xlabel("Validation Graph Size")
plt.ylabel("Mean Set Size")
plt.title("Mean Independent Set Size by Algorithm on Validation Graphs (Erdős-Rényi; p=0.15)")
plt.xticks(ind + width * len(solution_data) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('{}_test{}.png'.format(problem_type, training_graph_size))

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

bars = []
plt.figure(figsize=(20, 10))
if 'neural network random {}'.format(str(training_graph_size)) in solution_times:
    for vert in solution_times['neural network random {}'.format(str(training_graph_size))]:
        solution_times['neural network random {}'.format(str(training_graph_size))][vert] = np.divide(solution_times['neural network random {}'.format(str(training_graph_size))][vert], 50)

if 'neural network partial {}'.format(str(training_graph_size)) in solution_times:
    for vert in solution_times['neural network partial {}'.format(str(training_graph_size))]:
        solution_times['neural network partial {}'.format(str(training_graph_size))][vert] = np.divide(solution_times['neural network partial {}'.format(str(training_graph_size))][vert], 50)

for i, alg in enumerate(solution_times):
    next_data = []
    for vert in solution_times[alg]:
        next_data.append(solution_times[alg][vert])

    bars.append(plt.bar(ind + width * i, next_data, width))

for i, alg in enumerate(solution_times):
    for j, vert in enumerate(solution_times[alg]):
        plt.text(j + i * width, solution_times[alg][vert], round(solution_times[alg][vert], 2), ha='center', fontsize=8)


plt.xlabel("Validation Graph Size")
plt.ylabel("Mean Time to Solve")
plt.title("Mean Time to Solve by Algorithm on Validation Graphs (Erdős-Rényi; p=0.15)")
plt.xticks(ind + width * len(solution_times) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('{}_test_times{}.png'.format(problem_type, training_graph_size))
