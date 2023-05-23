import matplotlib.pyplot as plt
import numpy as np

vert_counts = [20,40,60,80,100]


x = ["|V| = {}".format(i) for i in vert_counts]
algorithms = ["CPLEX", 
              "MaxMatching", 
              "Greedy Empty Start", 
              "Greedy Random Start", 
              "NetworkX Min Weighted Cover",
              "Neural Network ER |V|=20, Empty Start",
              "Neural Network ER |V|=20, Full Start",
              "Neural Network ER |V|=20, Random Start"]

ind = np.arange(len(x))
width = 0.12

with open('test_data.txt') as f:
    solution_data = eval(f.read())

bars = []
plt.figure(figsize=(20, 10))

for i, data in enumerate(solution_data):
    bars.append(plt.bar(ind + width * i, data, width))

for i in range(len(solution_data)):
    for j in range(len(solution_data[i])):
        plt.text(j + i * width, solution_data[i][j], round(solution_data[i][j], 2), ha='center', fontsize=8)


plt.xlabel("Validation Graph Size")
plt.ylabel("Mean Cover Size")
plt.title("Mean Cover Size by Algorithm on Validation Graphs (Erdős–Rényi; p=0.15)")
plt.xticks(ind + width * len(solution_data) / 2, x)
plt.legend(tuple(bars), tuple(algorithms))
plt.savefig('test.png')