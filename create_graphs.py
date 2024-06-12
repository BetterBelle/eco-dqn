import pickle
from src.envs.utils import RandomErdosRenyiGraphGenerator, EdgeType, RandomBarabasiAlbertGraphGenerator
import sys

if len(sys.argv) != 3:
    print('Incorrect argument number') 
    print('Usage: python create_plots.py graph_size graph_type')
    exit(1)

num_nodes = sys.argv[1]
if not num_nodes.isnumeric():
    print('Incorrect argument usage: num_nodes must be a number') 
    print('Usage: python create_graphs.py num_nodes graph_type')
    exit(1)

num_nodes = int(num_nodes)

graph_type = sys.argv[2]
if graph_type not in ['ER', 'BA']:
    print('Incorrect argument usage: graph_type must be either ER or BA') 
    print('Usage: python create_graphs.py num_nodes graph_type')
    exit(1)

graph_edge_parameter = 'p15' if graph_type == 'ER' else 'm4'

graph_generator = None
if graph_type == 'ER':
    graph_generator = RandomErdosRenyiGraphGenerator(n_spins=num_nodes, p_connection=[0.15,0], edge_type=EdgeType.DISCRETE)
elif graph_type == 'BA':
    graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=num_nodes, m_insertion_edges=4, edge_type=EdgeType.DISCRETE)

graphs = []
for i in range(100):
    graphs.append(graph_generator.get())

pickle.dump(graphs, open("_graphs/validation/{}_{}spin_{}_100graphs.pkl".format(graph_type, num_nodes, graph_edge_parameter), 'bw'))
