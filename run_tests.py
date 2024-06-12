from experiments.test_eco import run_with_params 
import sys

if len(sys.argv) != 7:
    print('Incorrect argument number') 
    print('Usage: python run_tests vertex_count problem_type graph_train_type graph_test_type network_type stopping_type')
    exit(1)

self_filename = sys.argv[0]

num_verts = int(sys.argv[1])
if num_verts not in [20, 40, 60, 100, 200]:
    print('Invalid vertex count')
    exit(1)

problem_type = sys.argv[2]
if problem_type not in ['max_ind_set', 'min_cover', 'max_cut', 'min_cut', 'max_clique', 'min_dom_set']:
    print('Invalid problem type')
    exit(1) 

graph_train_type = sys.argv[3]
if graph_train_type not in ['ER', 'BA']:
    print('Invalid graph train type')
    exit(1)

graph_test_type = sys.argv[4]
if graph_test_type not in ['ER', 'BA']:
    print('Invalid graph test type')
    exit(1)

network_type = sys.argv[5]
if network_type not in ['eco', 's2v']:
    print('Invalid network type')
    exit(1)

stopping_type = sys.argv[6]
if stopping_type not in ['normal', 'quarter', 'early']:
    print('Invalid stopping type')

run_with_params(num_verts, problem_type, graph_train_type, graph_test_type, network_type, stopping_type)
