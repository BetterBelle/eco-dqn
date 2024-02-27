import pandas as pd
import json
import sys

if len(sys.argv) != 3:
    print('Incorrect argument number') 
    print('Usage: python create_histories.py problem_type graph_size')
    exit(1)

self_filename = sys.argv[0]

problem_type = sys.argv[1]
if problem_type not in ['min_cover', 'max_ind_set', 'max_cut', 'min_cut']:
    print('Invalid problem type')
    print('Problem types: min_cover, max_ind_set, max_cut, min_cut')
    exit(1)

training_graph_size = sys.argv[2]
if int(training_graph_size) not in [20, 40, 60, 80, 100, 200, 500, 1000, 2000]:
    print('Invalid training graph size')
    print('Graph sizes: 20, 40, 60, 80, 100, 200, 500, 1000, 2000')
    exit(1)

problem_init = ['empty', 'partial', 'full']

for init in problem_init:
    with open('data/{}_histories{}_neural network {} {}.txt'.format(problem_type, training_graph_size, init, training_graph_size)) as f:
        solution_data = json.load(f)

# action, solution, reward, qs, spins, score_mask, validity
    columns = [
            'best_solution',
            'best_solution_step',
            'min_time_between_best',
            'max_time_between_best',
            'avg_time_between_best',
            'first_solution',
            'last_solution',
            'valid_states',
            'invalid_states',
            'repeated_valid_states',
            'repeated_invalid_states',
            'first_invalid_state',
            'last_invalid_state',
            'actions',
            'repeated_actions',
            'local_optimums_found',
            'best_found_local_optimum',
            'graph_size',
    ]
    df = pd.DataFrame(columns=columns)

    for size in solution_data:
        for graph in solution_data[size]:
            best_solution = 1000000 if 'min' in problem_type else -100000
            best_solution_step = 0
            first_solution = 0
            last_solution = 0 
            valid_states = {}
            invalid_states = {}
            first_invalid_state = -1 
            last_invalid_state = 0
            actions = {}
            local_opt = 0
            best_found_local_opt = False
            best_found_times = []

            for index, step in enumerate(graph):
                action = step[0]
                solution = step[1]
                reward = step[2]
                qs = step[3]
                spins = str(step[4])
                score_mask = step[5]
                validity = True if step[6] == 'True' else False

                # setup dictionaries
                if validity:
                    if spins in valid_states:
                        valid_states[spins] += 1
                    else:
                        valid_states[spins] = 1 
                else:
                    if spins in invalid_states:
                        invalid_states[spins] += 1
                    else:
                        invalid_states[spins] = 1

                if action in actions:
                    actions[action] += 1
                else:
                    actions[action] = 1

                # finding local optimum
                flag = True 
                for score in score_mask:
                    if score > 0:
                        flag = False 
                if flag:
                    local_opt += 1

                # best_solution and step
                if ('min' in problem_type and solution < best_solution) or ('max' in problem_type and solution > best_solution):
                    best_solution = solution 
                    best_solution_step = index
                    best_found_local_opt = flag
                    best_found_times.append(index)
                        
                # first solution
                if index == 0:
                    first_solution = solution 
                # last solution
                if index == len(graph) - 1:
                    last_solution = solution
                # first and last invalid
                if not validity and first_invalid_state == -1:
                    first_invalid_state = index
                if not validity:
                    last_invalid_state = index

            num_valid_states = len(valid_states)
            num_invalid_states = len(invalid_states)
            num_actions = len(actions)
            repeated_valid_states = 0
            for s in valid_states:
                if valid_states[s] > 1:
                    repeated_valid_states += 1

            repeated_invalid_states = 0
            for s in invalid_states:
                if invalid_states[s] > 1:
                    repeated_invalid_states += 1

            repeated_actions = 0
            for a in actions:
                if actions[a] > 1:
                    repeated_actions += 1

            best_found_time_gaps = []
            for index, time in enumerate(best_found_times):
                if index != len(best_found_times) - 1:
                    best_found_time_gaps.append(best_found_times[index + 1] - time)

            if len(best_found_times) <= 1:
                best_found_time_gaps.append(0)

            row = [
                    best_solution,
                    best_solution_step,
                    min(best_found_time_gaps),
                    max(best_found_time_gaps),
                    sum(best_found_time_gaps)/len(best_found_time_gaps),
                    first_solution,
                    last_solution,
                    num_valid_states,
                    num_invalid_states,
                    repeated_valid_states,
                    repeated_invalid_states,
                    first_invalid_state,
                    last_invalid_state,
                    num_actions,
                    repeated_actions,
                    local_opt,
                    best_found_local_opt,
                    size,
            ]
            df.loc[len(df)] = row

    df.to_csv('{}_solution_start_history_data.csv'.format(init))
