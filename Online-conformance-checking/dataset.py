"""
Input v_source : Un marquage quelconque du reachability graph.
Input v_target : Un autre marquage atteignable depuis v_source.
Label (Cible) : Le vecteur alpha idéal qui représente le chemin le plus court.
"""
from reachability_graph_construction import *
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

# Génération du Dataset
dataset = []

import random

def create_split_dataset(graph, all_markings, t_name_to_idx, train_ratio=0.6):
    direct_paths = []    # Longueur 1
    complex_paths = []   # Longueur > 1
    
    for m_src in all_markings:
        for m_tgt in all_markings:
            if m_src == m_tgt: continue
            
            # On récupère le chemin le plus court
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

    # Mélange des chemins complexes
    random.shuffle(complex_paths)
    split_idx = int(len(complex_paths) * train_ratio)
    
    # Construction des sets
    # TRAIN : 100% des briques de base + une partie du complexe
    train_set = direct_paths + complex_paths[:split_idx]
    # TEST : Uniquement des chemins complexes jamais vus
    test_set = complex_paths[split_idx:]
    
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



def prepare_tensors(data, num_m, num_t, max_steps=5): # to do : remove max steps , but think about batching
    v_src_list, v_tgt_list, seq_alphas_list = [], [], []
    
    for item in data:
        # 1. Marquages (One-hot)
        src = torch.zeros(num_m); src[item['v_src_idx']] = 1.0
        tgt = torch.zeros(num_m); tgt[item['v_tgt_idx']] = 1.0
        
        # 2. Séquence d'alphas (Matrice [max_steps, num_t])
        # On crée une matrice vide (zéro)
        seq_tensor = torch.zeros((max_steps, num_t)) 
        indices = item['alphas_seq'] # La liste ordonnée [0, 4, 5]
        
        for step, t_idx in enumerate(indices):
            if step < max_steps:
                seq_tensor[step, t_idx] = 1.0
        
        v_src_list.append(src)
        v_tgt_list.append(tgt)
        seq_alphas_list.append(seq_tensor)
    # print(f"v_src_list : {v_src_list}")
    # print(f"v_tgt_list : {v_tgt_list}")
    # print(f"seq_alphas_list : {seq_alphas_list}")

    return torch.stack(v_src_list), torch.stack(v_tgt_list), torch.stack(seq_alphas_list)

# Conversion
X_src_train, X_tgt_train, y_alphas_train = prepare_tensors(train_data, num_m, num_t)
X_src_test, X_tgt_test, y_alphas_test = prepare_tensors(test_data, num_m, num_t)