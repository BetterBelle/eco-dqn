import matplotlib.pyplot as plt
import numpy as np

vert_counts = [20, 40, 60, 80, 100, 200]


x = ["|V| = {}".format(i) for i in vert_counts]

ind = np.arange(len(x))
width = 0.12

with open('test_data20.txt') as f:
    solution_data : dict = eval(f.read())

algorithms = list(solution_data.keys())

# Want to get mean from every non-single graph test
for alg in solution_data:
    # If the solution_data's first graph size's first item is a list, then we want to change it to the mean solution
    if type(solution_data[alg][0][0]) == list:
        for i in range(len(solution_data[alg])):
            for j in range(len(solution_data[alg][i])):
                solution_data[alg][i][j] = int(np.mean(solution_data[alg][i][j]))

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