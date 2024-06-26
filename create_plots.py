import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) != 3:
    print('Incorrect argument number') 
    print('Usage: python create_plots.py problem_type graph_size')
    exit(1)

self_filename = sys.argv[0]

problem_type = sys.argv[1]
if problem_type not in ['min_cover', 'max_ind_set', 'max_cut', 'min_cut', 'min_dom_set', 'max_clique']:
    print('Invalid problem type')
    print('Problem types: min_cover, max_ind_set, max_cut, min_cut, min_dom_set, max_clique')
    exit(1)

training_graph_size = sys.argv[2]
if int(training_graph_size) not in [20, 40, 60, 80, 100, 200, 500, 1000, 2000]:
    print('Invalid graph size')
    print('Graph sizes: 20, 40, 60, 80, 100, 200, 500, 1000, 2000')
    exit(1)

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
    first_key = list(solution_data[alg].keys())[0]
    if type(solution_data[alg][first_key][0]) == list:
        for vert in solution_data[alg]:
            for j in range(len(solution_data[alg][vert])):
                solution_data[alg][vert][j] = np.mean(solution_data[alg][vert][j])

for alg in solution_times:
    first_key = list(solution_data[alg].keys())[0]
    if type(solution_times[alg][first_key][0]) == list:
        for vert in solution_times[alg]:
            for j in range(len(solution_times[alg][vert])):
                solution_times[alg][vert][j] = np.mean(solution_times[alg][vert][j])

# Now we create approximation ratios with respect to the cplex solutions (always first in dict)
if 'cplex' in solution_data:
    cplex_solutions = solution_data['cplex'].copy()
else:
    cplex_solutions = None

if cplex_solutions != None:
    for alg in solution_data:
        for vert in solution_data[alg]:
            if 'max' in problem_type:
                print('========= CPLEX ============')
                print('size: {}, graph: {}'.format(len(cplex_solutions[vert]), vert))
                print(cplex_solutions[vert])
                print('======== SOL DATA ========')
                print('size: {}, graph: {}'.format(len(solution_data[alg][vert]), vert))
                print('algorithm: {}'.format(alg))
                solution_data[alg][vert] = list(np.divide(cplex_solutions[vert], [1 if item == 0 else item for item in solution_data[alg][vert]]))
            elif 'min' in problem_type: 
                print('========= CPLEX ============')
                print('size: {}, graph: {}'.format(len(cplex_solutions[vert]), vert))
                print(cplex_solutions[vert])
                print('======== SOL DATA ========')
                print('size: {}, graph: {}'.format(len(solution_data[alg][vert]), vert))
                print('algorithm: {}'.format(alg))
                print(solution_data[alg][vert])
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


problem_name = 'Independent Set'
if problem_type == 'min_cover':
    problem_name = 'Vertex Cover'
if problem_type == 'max_ind_set':
    problem_name = 'Independent Set'
if problem_type == 'max_cut' or problem_type == 'min_cut':
    problem_name = 'Cut'
if problem_type == 'min_dom_set':
    problem_name = 'Dominating Set'
if problem_type == 'max_clique':
    problem_name = 'Clique'

plt.xlabel("Validation Graph Size")
plt.ylabel("Mean Set Size")
plt.title("Mean {} Size by Algorithm on Validation Graphs (Erdős-Rényi; p=0.15)".format(problem_name))
plt.xticks(ind + width * len(solution_data) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('{}_test{}.png'.format(problem_type, training_graph_size))

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

bars = []
plt.figure(figsize=(20, 10))

### UNCOMMENT ONLY WHEN PARTIAL/RANDOM SOLUTIONS RUN 50 TIMES
# if 'neural network random {}'.format(str(training_graph_size)) in solution_times:
#     for vert in solution_times['neural network random {}'.format(str(training_graph_size))]:
#         solution_times['neural network random {}'.format(str(training_graph_size))][vert] = np.divide(solution_times['neural network random {}'.format(str(training_graph_size))][vert], 50)

# if 'neural network partial {}'.format(str(training_graph_size)) in solution_times:
#     for vert in solution_times['neural network partial {}'.format(str(training_graph_size))]:
#         solution_times['neural network partial {}'.format(str(training_graph_size))][vert] = np.divide(solution_times['neural network partial {}'.format(str(training_graph_size))][vert], 50)

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
plt.title("Mean Time to Solve {} by Algorithm on Validation Graphs (Erdős-Rényi; p=0.15)".format(problem_name))
plt.xticks(ind + width * len(solution_times) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('{}_test_times{}.png'.format(problem_type, training_graph_size))
