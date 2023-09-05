from experiments.test_eco import run_with_params 
import sys

if len(sys.argv) != 5:
    num_verts = 20
    problem_type = 'max_ind_set'
    graph_type = 'ER'
    network_type = 'eco'
    print('Incorrect argument number') 
    run_with_params(num_verts, problem_type, graph_type, network_type)

self_filename = sys.argv[0]

num_verts = int(sys.argv[1])
if num_verts not in [20, 40, 60, 100, 200]:
    print('Invalid vertex count')
    exit(1)

problem_type = sys.argv[2]
if problem_type not in ['max_ind_set', 'min_cover', 'max_cut', 'min_cut']:
    print('Invalid problem type')
    exit(1) 

graph_type = sys.argv[3]
if graph_type not in ['ER', 'BA']:
    print('Invalid graph type')
    exit(1)

network_type = sys.argv[4]
if network_type not in ['eco', 's2v']:
    print('Invalid network type')
    exit(1)

run_with_params(num_verts, problem_type, graph_type, network_type)
