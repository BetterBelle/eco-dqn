import pickle
from src.envs.utils import RandomErdosRenyiGraphGenerator, EdgeType
import sys

if len(sys.argv) != 3:
    print('Incorrect argument number') 
    print('Usage: python create_plots.py problem_type graph_size')
    exit(1)

num_nodes = sys.argv[1]
if not num_nodes.isnumeric():
    print('Incorrect argument usage: num_nodes must be a number') 
    print('Usage: python create_graphs.py num_nodes graph_type')
    exit(1)

graph_type = sys.argv[2]
if graph_type not in ['ER', 'BA']:
    print('Incorrect argument usage: graph_type must be either ER or BA') 
    print('Usage: python create_graphs.py num_nodes graph_type')
    exit(1)

er_generator = RandomErdosRenyiGraphGenerator(n_spins=2000, p_connection=[0.15,0], edge_type=EdgeType.DISCRETE)
graphs = []
for i in range(100):
    graphs.append(er_generator.get())

pickle.dump(graphs, open("_graphs/validation/{}_{}spin_p15_100graphs.pkl".format(graph_type, 2000), 'bw'))
