import matplotlib.pyplot as plt
import numpy as np

vert_counts = [20, 40, 60, 80, 100, 200, 500]


x = ["|V| = {}".format(i) for i in vert_counts]

ind = np.arange(len(x))
width = 0.12

with open('test_data20.txt') as f:
    solution_data : dict = eval(f.read())

with open('test_times20.txt') as f:
    solution_times : dict = eval(f.read())

algorithms = list(solution_data.keys())

# Want to get mean from every non-single graph test
for alg in solution_data:
    # If the solution_data's first graph size's first item is a list, then we want to change it to the mean solution
    if type(solution_data[alg][0][0]) == list:
        for i in range(len(solution_data[alg])):
            for j in range(len(solution_data[alg][i])):
                solution_data[alg][i][j] = np.mean(solution_data[alg][i][j])

for alg in solution_times:
    if type(solution_times[alg][0][0]) == list:
        for i in range(len(solution_times[alg])):
            for j in range(len(solution_times[alg][i])):
                solution_times[alg][i][j] = np.mean(solution_times[alg][i][j])

# Now we create approximation ratios with respect to the cplex solutions (always first in dict)
for i in solution_data:
    cplex_solutions = solution_data[i]
    break

for alg in solution_data:
    solution_data[alg] = list(np.divide(solution_data[alg], cplex_solutions))

# Now we get the average approximation ratio for every sub-list
for alg in solution_data:
    for i in range(len(solution_data[alg])):
        solution_data[alg][i] = np.mean(solution_data[alg][i])

for alg in solution_times:
    for i in range(len(solution_times[alg])):
        solution_times[alg][i] = np.mean(solution_times[alg][i])

bars = []
plt.figure(figsize=(20, 10))

for i, data in enumerate(solution_data):
    bars.append(plt.bar(ind + width * i, solution_data[data], width))

for i, alg in enumerate(solution_data):
    for j in range(len(solution_data[alg])):
        plt.text(j + i * width, solution_data[alg][j], round(solution_data[alg][j], 2), ha='center', fontsize=8)


plt.xlabel("Validation Graph Size")
plt.ylabel("Mean Cover Size")
plt.title("Mean Cover Size by Algorithm on Validation Graphs (Erdős-Rényi; p=0.15)")
plt.xticks(ind + width * len(solution_data) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('test20.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

plt.figure(figsize=(20, 10))
solution_times['neural_network_random_start_times'] = np.divide(solution_times['neural_network_random_start_times'], 50)

for i in solution_times:
    cplex_times = solution_times[i]
    break

for alg in solution_times:
    solution_times[alg] = list(np.divide(solution_times[alg], cplex_times))

for i, data in enumerate(solution_times):
    bars.append(plt.bar(ind + width * i, solution_times[data], width))

for i, alg in enumerate(solution_times):
    for j in range(len(solution_times[alg])):
        plt.text(j + i * width, solution_times[alg][j], round(solution_times[alg][j], 2), ha='center', fontsize=8)


plt.xlabel("Validation Graph Size")
plt.ylabel("Mean Time to Solve")
plt.title("Mean Time to Solve by Algorithm on Validation Graphs (Erdős-Rényi; p=0.15)")
plt.xticks(ind + width * len(solution_times) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('test_times20.png')