"""
Input v_source : Un marquage quelconque du reachability graph.
Input v_target : Un autre marquage atteignable depuis v_source.
Label (Cible) : Le vecteur alpha idéal qui représente le chemin le plus court.
"""
from reachability_graph import *
import heapq

def get_shortest_path_sequence(graph, start_node, end_node, t_name_to_idx):
    # queue: (distance, current_marking, sequence_of_indices)
    queue = [(0, start_node, [])]
    visited = {start_node: 0}
    
    while queue:
        (dist, current, seq) = heapq.heappop(queue)
        
        if current == end_node:
            return seq # Retourne par ex: [0, 4, 5] (indices de t1, t5, t6)
        
        if current in graph:
            for t_name, mode, next_marking in graph[current]:
                new_dist = dist + 1
                if next_marking not in visited or new_dist < visited[next_marking]:
                    visited[next_marking] = new_dist
                    new_seq = seq + [t_name_to_idx[t_name]]
                    heapq.heappush(queue, (new_dist, next_marking, new_seq))
    return None

dataset = []

import random

def create_split_dataset(graph, all_markings, t_name_to_idx, train_ratio=0.3):
    direct_paths = []    # length 1
    complex_paths = []   # length > 1
    
    for m_src in all_markings:
        for m_tgt in all_markings:
            if m_src == m_tgt: continue
            
            seq = get_shortest_path_sequence(graph, start_node=m_src, end_node=m_tgt, t_name_to_idx=t_name_to_idx)
            
            if seq:  
                length = len(seq)
                data_point = {
                    'v_src_idx': marking_to_idx[m_src],
                    'v_tgt_idx': marking_to_idx[m_tgt],
                    'alphas_seq': seq,
                    'length': length
                }
                
                if length == 1:
                    direct_paths.append(data_point)
                else:
                    complex_paths.append(data_point)

    complex_by_src = {}
    complex_by_tgt = {}
    for path in complex_paths:
        src = path['v_src_idx']
        tgt = path['v_tgt_idx']
        complex_by_src.setdefault(src, []).append(path)
        complex_by_tgt.setdefault(tgt, []).append(path)
    train_set = list(direct_paths)
    forced_into_train = set()

    for tgt, paths in complex_by_tgt.items():
        chosen = paths[0]
        train_set.append(chosen)
        forced_into_train.add(id(chosen))

    for src, paths in complex_by_src.items():
        for path in paths:
            if id(path) not in forced_into_train:
                train_set.append(path)
                forced_into_train.add(id(path))
                break
    remaining = [p for p in complex_paths if id(p) not in forced_into_train]
    random.shuffle(remaining)
    split = int(len(remaining) * train_ratio)
    train_set += remaining[:split]
    test_set = remaining[split:]

    return train_set, test_set

train_data, test_data = create_split_dataset(reachability_graph, all_markings, t_name_to_idx)
print(f"Train: {len(train_data)} | Test: {len(test_data)}")

# print("\n========= train dataset ==============")
# for dataset_element in train_data:
#     print(f"Source: {all_markings[dataset_element['v_src_idx']]}")
#     print(f"Target: {all_markings[dataset_element['v_tgt_idx']]}")
#     print(f"Alphas cibles (transitions à tirer): {dataset_element['alphas']}")
# print("\n========= test dataset ==============")
# for dataset_element in test_data:
#     print(f"Source: {all_markings[dataset_element['v_src_idx']]}")
#     print(f"Target: {all_markings[dataset_element['v_tgt_idx']]}")
#     print(f"Alphas cibles (transitions à tirer): {dataset_element['alphas']}")



def prepare_tensors(data, num_m, num_t):
    v_src_list, v_tgt_list, seq_alphas_list = [], [], []
    
    for item in data:
        src = torch.zeros(num_m); src[item['v_src_idx']] = 1.0
        tgt = torch.zeros(num_m); tgt[item['v_tgt_idx']] = 1.0
        
        indices = item['alphas_seq']  # ex: [0, 4, 5]
        
        step_transition_list = []  
        for t_idx in indices:
            step_tensor = torch.zeros(num_t)
            step_tensor[t_idx] = 1.0
            step_transition_list.append(step_tensor)
        
        seq_tensor = torch.stack(step_transition_list)  # [L, num_t]
        
        v_src_list.append(src)
        v_tgt_list.append(tgt)
        seq_alphas_list.append(seq_tensor) 

    return torch.stack(v_src_list), torch.stack(v_tgt_list), seq_alphas_list

X_src_train, X_tgt_train, y_alphas_train = prepare_tensors(train_data, num_m, num_t)
X_src_test,  X_tgt_test,  y_alphas_test  = prepare_tensors(test_data,  num_m, num_t)