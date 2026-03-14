import sys 
sys.path.append("Online-conformance-checking")
from collections import deque

# from example_2.graph_example_2 import *
from example_1.graph_example_1 import *

def build_reachability_graph(net):
    graph = {}  # {marking: [(transition, mode, next_marking)]}
    
    initial = net.get_marking()
    queue = deque([initial])
    visited = {initial}
    
    while queue:
        marking = queue.popleft()
        graph[marking] = []
        
        net.set_marking(marking)  # restaure le marking courant
        
        for t_name in [t.name for t in net.transition()]:
            t = net.transition(t_name)
            net.set_marking(marking)
            for mode in t.modes():
                print("Firing transition:", t_name, "with mode:", mode)
                # tire la transition
                net.set_marking(marking)
                t.fire(mode)
                next_marking = net.get_marking()
                
                graph[marking].append((t_name, mode, next_marking))
                
                if next_marking not in visited:
                    visited.add(next_marking)
                    queue.append(next_marking)
    
    return graph


reachability_graph = build_reachability_graph(n)
print("\n=== Reachability Graph ===")
for marking, transitions in reachability_graph.items():
    print(f"\nFrom: {marking}")
    for t_name, mode, next_marking in transitions:
        print(f"  --[{t_name}]--> {next_marking}")

print(f"\nTotal states: {len(reachability_graph)}")




all_markings = list(reachability_graph.keys())
marking_to_idx = {m: i for i, m in enumerate(all_markings)}

all_transition_names = sorted([t.name for t in n.transition()])
t_name_to_idx = {name: i for i, name in enumerate(all_transition_names)}

num_m = len(all_markings)
num_t = len(all_transition_names)

import torch
import math

# reachability graph as a tensor of shape (Transitions, num_markings, num_markings)
def build_reachability_graph_tensor(num_t, num_m, reachability_graph):
  reachability_tensor = torch.zeros((num_t, num_m, num_m))
  angle = math.pi / 2
  tansition_used_indices = {t_idx: set() for t_idx in range(num_t)}
  for marking, transitions in reachability_graph.items():
      src_marking_idx = marking_to_idx[marking]

      for t_name, mode, next_marking in transitions:
          dst_marking_idx = marking_to_idx[next_marking]
          t_idx = t_name_to_idx[t_name]

          # check even if it ddoesn't happen by definition in the process graph
          used = tansition_used_indices[t_idx]
          if src_marking_idx in used or dst_marking_idx in used:
              print(f"[warning] transition {t_name}: index collision")
          used.add(src_marking_idx)
          used.add(dst_marking_idx)
          # end check
          
          reachability_tensor[t_idx, src_marking_idx, dst_marking_idx] = -angle 
          reachability_tensor[t_idx, dst_marking_idx, src_marking_idx] = angle
  return reachability_tensor

reachability_tensor = build_reachability_graph_tensor(num_t=num_t, num_m=num_m, reachability_graph=reachability_graph)
# print(f"reachability tensor : {reachability_tensor}")


transition_to_enabled_markings = {}

for marking, transitions in reachability_graph.items():
    for t_name, mode, next_marking in transitions:
        if t_name not in transition_to_enabled_markings:
            transition_to_enabled_markings[t_name] = []
        transition_to_enabled_markings[t_name].append({
            'from': marking,
            'to':   next_marking,
            'from_idx': marking_to_idx[marking],
            'to_idx':   marking_to_idx[next_marking]
        })

# C matrix (num_p, num_t)
place_names = sorted([p.name for p in n.place()])
place_to_idx = {name: i for i, name in enumerate(place_names)}
num_p = len(place_names)

marking_vectors = torch.zeros(num_m, num_p)
for marking, idx in marking_to_idx.items():
    for p_name in place_names:
        try:
            tokens = marking(p_name)   
            marking_vectors[idx, place_to_idx[p_name]] = len(tokens)
        except:
            marking_vectors[idx, place_to_idx[p_name]] = 0

C = torch.zeros(num_p, num_t)
for marking, transitions in reachability_graph.items():
    for t_name, mode, next_marking in transitions:
        t_idx = t_name_to_idx[t_name]
        src_vec = marking_vectors[marking_to_idx[marking]]
        dst_vec = marking_vectors[marking_to_idx[next_marking]]
        col = dst_vec - src_vec  
        C[:, t_idx] = col         

C = C.detach()
marking_vectors = marking_vectors.detach()