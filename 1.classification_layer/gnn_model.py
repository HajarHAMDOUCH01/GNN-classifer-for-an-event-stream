"""
GNN Model for Online Conformance Checking
Predicts whether an incoming event is conformant given current marking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, global_mean_pool


def build_n1_petri_net_structure():
    """
    Build the N1 Petri net structure as a heterogeneous graph
    This represents the STATIC structure of the process model
    
    Places: pi, p1, p2, p3, p4, p5, po (indices 0-6)
    Transitions: t1(a), t2(b), t3(c), t4(d), t5(d), t6(τ), t7(e), t8(f) (indices 0-7)
    
    Returns edge_index for pre and post arcs
    """
    # Pre-arcs: (place, pre, transition)
    # Format: which places must have tokens to enable which transitions
    pre_arcs = [
        # t1(a): pi -> t1
        [0, 0],
        # t2(b): p1 -> t2
        [1, 1],
        # t3(c): p2 -> t3
        [2, 2],
        # t4(d): p1 + p4 -> t4 (first variant of d)
        [1, 3], [4, 3],
        # t5(d): p3 + p4 -> t5 (second variant of d)
        [3, 4], [4, 4],
        # t6(τ): p5 -> t6 (silent transition)
        [5, 5],
        # t7(e): p5 -> t7
        [5, 6],
        # t8(f): p5 -> t8
        [5, 7]
    ]
    
    # Post-arcs: (transition, post, place)
    # Format: which places receive tokens when transitions fire
    post_arcs = [
        # t1 -> p1, p2
        [0, 1], [0, 2],
        # t2 -> p3
        [1, 3],
        # t3 -> p4
        [2, 4],
        # t4 -> p5
        [3, 5],
        # t5 -> p5
        [4, 5],
        # t6 -> (loop back - structure depends on full model)
        [5, 1], [5, 2],  # simplified
        # t7 -> po
        [6, 6],
        # t8 -> po
        [7, 6]
    ]
    
    pre_edge_index = torch.tensor(pre_arcs, dtype=torch.long).t().contiguous()
    post_edge_index = torch.tensor(post_arcs, dtype=torch.long).t().contiguous()
    
    return pre_edge_index, post_edge_index


def create_hetero_data_from_sample(marking, event, label, transition_labels=['a', 'b', 'c', 'd', 'e', 'f']):
    """
    Convert a dataset sample into HeteroData format
    
    Args:
        marking: List of 7 integers [pi, p1, p2, p3, p4, p5, po]
        event: String, one of ['a', 'b', 'c', 'd', 'e', 'f']
        label: Int, 0 (conformant) or 1 (non-conformant)
    
    Returns:
        HeteroData object
    """
    data = HeteroData()
    
    # Place node features: token counts (current marking)
    data['place'].x = torch.tensor(marking, dtype=torch.float).unsqueeze(1)  # [7, 1] state of a current marking
    
    # Transition node features: one-hot encoding of incoming event
    num_transitions = len(transition_labels)
    event_idx = transition_labels.index(event) # an event occured
    transition_features = torch.zeros((num_transitions, 1), dtype=torch.float)
    transition_features[event_idx] = 1.0  # Mark the incoming transition
    data['transition'].x = transition_features  # [6, 1]
    
    # Add Petri net structure (static)
    pre_edges, post_edges = build_n1_petri_net_structure()
    data['place', 'pre', 'transition'].edge_index = pre_edges
    data['transition', 'post', 'place'].edge_index = post_edges
    
    # Label
    data.y = torch.tensor([label], dtype=torch.long)
    
    return data


class ConformanceGNN(nn.Module):
    """
    Heterogeneous GNN for conformance checking
    
    Architecture:
    1. Message passing between places and transitions
    2. Learn the enabling logic of the Petri net
    3. Global pooling to get graph-level representation
    4. Binary classification: conformant vs non-conformant
    """
    
    def __init__(self, hidden_dim=64, num_layers=2):
        super(ConformanceGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projections
        self.place_proj = nn.Linear(1, hidden_dim)  # marking value -> hidden
        self.transition_proj = nn.Linear(1, hidden_dim)  # incoming event indicator -> hidden
        
        # Heterogeneous graph convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('place', 'pre', 'transition'): SAGEConv(hidden_dim, hidden_dim),
                ('transition', 'post', 'place'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)
        
        # Classifier head
        # We'll pool both place and transition features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: HeteroData with:
                - data['place'].x: [num_places, 1] marking
                - data['transition'].x: [num_transitions, 1] incoming event
                - edges: Petri net structure
        
        Returns:
            logits: [batch_size, 2] class logits
        """
        # Project to hidden dimension
        x_dict = {
            'place': self.place_proj(data['place'].x),  # [7, hidden_dim]
            'transition': self.transition_proj(data['transition'].x)  # [6, hidden_dim]
        }
        
        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Global pooling
        # Pool place features
        place_pooled = torch.mean(x_dict['place'], dim=0)  # [hidden_dim]
        # Pool transition features
        trans_pooled = torch.mean(x_dict['transition'], dim=0)  # [hidden_dim]
        
        # Concatenate
        graph_repr = torch.cat([place_pooled, trans_pooled], dim=0)  # [hidden_dim * 2]
        
        # Classify
        logits = self.classifier(graph_repr.unsqueeze(0))  # [1, 2]
        
        return logits