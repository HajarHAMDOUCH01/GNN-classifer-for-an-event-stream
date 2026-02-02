"""
Dataset Generation for GNN-based Online Conformance Checking
Goal: Train a classifier to predict if an incoming event is conformant given current marking
This replaces the A* algorithm with a learned model
"""

import random
from typing import List, Tuple
from collections import defaultdict

# Transitions with activity labels
TRANSITIONS = ['a', 'b', 'c', 'd', 'e', 'f']
T_MAP = {label: i for i, label in enumerate(TRANSITIONS)}


class N1Simulator:
    """Simulator for the N1 Petri net"""
    
    def __init__(self):
        # Initial marking: [pi, p1, p2, p3, p4, p5, po]
        self.initial_marking = [1, 0, 0, 0, 0, 0, 0]
        self.place_names = ['pi', 'p1', 'p2', 'p3', 'p4', 'p5', 'po']
    
    def get_enabled_transitions(self, marking: List[int]) -> List[str]:
        """Get all enabled transitions for current marking"""
        enabled = []
        pi, p1, p2, p3, p4, p5, po = marking
        
        if pi > 0:
            enabled.append('a')
        if p1 > 0:
            enabled.append('b')
        if p2 > 0:
            enabled.append('c')
        if (p3 > 0 and p4 > 0) or (p1 > 0 and p4 > 0):
            enabled.append('d')
        if p5 > 0:
            enabled.append('e')
            enabled.append('f')
        
        return enabled
    
    def fire(self, marking: List[int], transition: str) -> Tuple[List[int], bool]:
        """Fire a transition and return new marking"""
        new_m = marking.copy()
        pi, p1, p2, p3, p4, p5, po = new_m
        
        if transition == 'a' and pi > 0:
            new_m[0] -= 1  # pi
            new_m[1] += 1  # p1
            new_m[2] += 1  # p2
            return new_m, True
        
        elif transition == 'b' and p1 > 0:
            new_m[1] -= 1  # p1
            new_m[3] += 1  # p3
            return new_m, True
        
        elif transition == 'c' and p2 > 0:
            new_m[2] -= 1  # p2
            new_m[4] += 1  # p4
            return new_m, True
        
        elif transition == 'd':
            if p3 > 0 and p4 > 0:
                new_m[3] -= 1
                new_m[4] -= 1
                new_m[5] += 1
                return new_m, True
            elif p1 > 0 and p4 > 0:
                new_m[1] -= 1
                new_m[4] -= 1
                new_m[5] += 1
                return new_m, True
        
        elif transition == 'e' and p5 > 0:
            new_m[5] -= 1
            new_m[6] += 1
            return new_m, True
        
        elif transition == 'f' and p5 > 0:
            new_m[5] -= 1
            new_m[6] += 1
            return new_m, True
        
        return marking, False
    
    def generate_conforming_trace(self, max_length: int = 20) -> List[str]:
        """Generate a random conforming trace"""
        marking = self.initial_marking.copy()
        trace = []
        
        for _ in range(max_length):
            enabled = self.get_enabled_transitions(marking)
            if not enabled:
                break
            
            transition = random.choice(enabled)
            marking, success = self.fire(marking, transition)
            
            if success:
                trace.append(transition)
            
            if marking[6] > 0:  # Reached final state
                break
        
        return trace
    
    def get_marking_after_prefix(self, prefix: List[str]) -> List[int]:
        """Get marking after replaying a prefix"""
        marking = self.initial_marking.copy()
        
        for event in prefix:
            marking, success = self.fire(marking, event)
            if not success:
                break
        
        return marking


def inject_noise(trace: List[str], noise_type: str, noise_level: float) -> List[str]:
    """
    Inject noise into a trace
    noise_type: 'remove', 'add', 'swap'
    """
    if random.random() > noise_level or len(trace) == 0:
        return trace.copy()
    
    noisy_trace = trace.copy()
    
    if noise_type == 'remove' and len(noisy_trace) > 0:
        idx = random.randint(0, len(noisy_trace) - 1)
        noisy_trace.pop(idx)
    
    elif noise_type == 'add':
        random_activity = random.choice(TRANSITIONS)
        idx = random.randint(0, len(noisy_trace))
        noisy_trace.insert(idx, random_activity)
    
    elif noise_type == 'swap' and len(noisy_trace) >= 2:
        idx = random.randint(0, len(noisy_trace) - 2)
        noisy_trace[idx], noisy_trace[idx + 1] = noisy_trace[idx + 1], noisy_trace[idx]
    
    return noisy_trace


def generate_online_conformance_dataset(
    simulator: N1Simulator,
    num_traces: int = 1000,
    noise_type: str = None,
    noise_level: float = 0.0,
    balance_dataset: bool = True
) -> List[Tuple[List[int], str, int]]:
    """
    Generate dataset for GNN training to replace A* algorithm
    
    For each event in each trace, generate:
    - Current marking (before event)
    - Incoming event
    - Label: 0 if conformant (event enabled), 1 if non-conformant
    
    This simulates the online scenario where events arrive one by one
    """
    samples = []
    
    for _ in range(num_traces):
        # Generate conforming trace
        trace = simulator.generate_conforming_trace()
        
        if len(trace) == 0:
            continue
        
        # Apply noise if specified
        if noise_type:
            trace = inject_noise(trace, noise_type, noise_level)
        
        # Generate samples for EACH event in the trace
        # This simulates the online setting: event arrives → check conformance
        marking = simulator.initial_marking.copy()
        
        for event in trace:
            # Get enabled transitions from current marking
            enabled = simulator.get_enabled_transitions(marking)
            
            # Label: 0 if event is enabled (conformant), 1 otherwise (non-conformant)
            label = 0 if event in enabled else 1
            
            # Store: (current_marking, incoming_event, label)
            samples.append((marking.copy(), event, label))
            
            # Update marking by firing the event (if possible)
            # In real online scenario, the system continues even with deviations
            new_marking, success = simulator.fire(marking, event)
            if success:
                marking = new_marking
            # If event is non-conformant, marking stays the same
            # this is just a classifier , (in practice, the system might handle this differently)
    
    # Balance dataset if requested
    if balance_dataset:
        samples = balance_samples(samples)
    
    return samples


def balance_samples(samples: List[Tuple[List[int], str, int]]) -> List[Tuple[List[int], str, int]]:
    """Balance the dataset by undersampling the majority class"""
    conformant = [s for s in samples if s[2] == 0]
    non_conformant = [s for s in samples if s[2] == 1]
    
    min_count = min(len(conformant), len(non_conformant))
    
    if min_count == 0:
        return samples
    
    # Undersample majority class
    conformant_balanced = random.sample(conformant, min_count)
    non_conformant_balanced = random.sample(non_conformant, min_count)
    
    balanced = conformant_balanced + non_conformant_balanced
    random.shuffle(balanced)
    
    return balanced


def generate_negative_samples(
    simulator: N1Simulator,
    num_samples: int = 1000
) -> List[Tuple[List[int], str, int]]:
    """
    Generate additional negative samples by explicitly creating non-conformant scenarios
    This ensures the model learns to detect various types of violations
    """
    samples = []
    
    for _ in range(num_samples):
        # Generate a random conforming trace
        trace = simulator.generate_conforming_trace()
        if len(trace) == 0:
            continue
        
        # Pick a random position in the trace
        if len(trace) > 1:
            pos = random.randint(0, len(trace) - 1)
        else:
            pos = 0
        
        # Get marking at that position
        prefix = trace[:pos]
        marking = simulator.get_marking_after_prefix(prefix)
        
        # Get enabled transitions
        enabled = simulator.get_enabled_transitions(marking)
        
        # Pick a NON-enabled transition
        all_transitions = TRANSITIONS.copy()
        not_enabled = [t for t in all_transitions if t not in enabled]
        
        if not_enabled:
            event = random.choice(not_enabled)
            samples.append((marking.copy(), event, 1))  # Non-conformant
    
    return samples


# ============================================================================
# Save/Load Dataset with Polars
# ============================================================================
def save_dataset_polars(samples: List[Tuple[List[int], str, int]], filepath: str):
    """
    Save dataset to Parquet format using Polars
    Each row contains: marking (7 columns), event, label
    """
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed. Install with: pip install polars")
        return None
    
    # Convert samples to structured format
    data = {
        'pi': [],
        'p1': [],
        'p2': [],
        'p3': [],
        'p4': [],
        'p5': [],
        'po': [],
        'event': [],
        'label': []
    }
    
    for marking, event, label in samples:
        data['pi'].append(marking[0])
        data['p1'].append(marking[1])
        data['p2'].append(marking[2])
        data['p3'].append(marking[3])
        data['p4'].append(marking[4])
        data['p5'].append(marking[5])
        data['po'].append(marking[6])
        data['event'].append(event)
        data['label'].append(label)
    
    # Create Polars DataFrame
    df = pl.DataFrame(data)
    
    # Save to Parquet
    df.write_parquet(filepath)
    print(f"Dataset saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Schema: {df.schema}")
    
    return df


def load_dataset_polars(filepath: str) -> List[Tuple[List[int], str, int]]:
    """
    Load dataset from Parquet format using Polars
    Returns list of (marking, event, label) tuples
    """
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed. Install with: pip install polars")
        return []
    
    df = pl.read_parquet(filepath)
    print(f"Loaded dataset from {filepath}")
    print(f"Shape: {df.shape}")
    
    # Convert back to list of tuples
    samples = []
    for row in df.iter_rows(named=True):
        marking = [
            row['pi'], row['p1'], row['p2'], row['p3'], 
            row['p4'], row['p5'], row['po']
        ]
        event = row['event']
        label = row['label']
        samples.append((marking, event, label))
    
    return samples


def save_dataset_polars_compact(samples: List[Tuple[List[int], str, int]], filepath: str):
    """
    Save dataset in a more compact format with marking as list column
    """
    try:
        import polars as pl
    except ImportError:
        print("Polars not installed. Install with: pip install polars")
        return None
    
    # Convert to list format
    markings = [marking for marking, _, _ in samples]
    events = [event for _, event, _ in samples]
    labels = [label for _, _, label in samples]
    
    df = pl.DataFrame({
        'marking': markings,
        'event': events,
        'label': labels
    })
    
    df.write_parquet(filepath)
    print(f"Compact dataset saved to {filepath}")
    print(f"Shape: {df.shape}")
    
    return df


# ============================================================================
# Build PyG Dataset (if using PyTorch Geometric)
# ============================================================================
def build_pyg_dataset(samples: List[Tuple[List[int], str, int]]):
    """
    Build PyTorch Geometric HeteroData dataset
    Requires: torch, torch_geometric
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        print("PyTorch Geometric not installed. Returning raw samples.")
        return samples
    
    dataset = []
    
    for marking, event, label in samples:
        data = HeteroData()
        
        # Place node features: marking vector
        marking_tensor = torch.tensor(marking, dtype=torch.float).unsqueeze(1)  # [7, 1]
        data['place'].x = marking_tensor
        
        # Transition node features: one-hot encoding of incoming event
        event_idx = T_MAP[event]
        event_features = torch.zeros((len(TRANSITIONS), 1))
        event_features[event_idx] = 1.0
        data['transition'].x = event_features
        
        # Label: 0 = conformant, 1 = non-conformant
        data.y = torch.tensor([label], dtype=torch.long)
        
        dataset.append(data)
    
    return dataset


# ============================================================================
# Demo and Statistics
# ============================================================================
if __name__ == "__main__":
    random.seed(42)
    simulator = N1Simulator()
    
    print("=" * 80)
    print("GNN Dataset Generation for Online Conformance Checking")
    print("Replacing A* algorithm with learned classifier")
    print("=" * 80)
    
    # Generate dataset with no noise
    print("\n1. CONFORMING traces (no noise):")
    print("-" * 80)
    conforming_samples = generate_online_conformance_dataset(
        simulator,
        num_traces=1000,
        noise_type=None,
        balance_dataset=False
    )
    
    conf_count = sum(1 for _, _, label in conforming_samples if label == 0)
    print(f"Total samples: {len(conforming_samples)}")
    print(f"Conformant: {conf_count} ({100*conf_count/len(conforming_samples):.1f}%)")
    print(f"Non-conformant: {len(conforming_samples)-conf_count}")
    
    # Generate dataset with noise
    print("\n2. NOISY traces (30% remove noise):")
    print("-" * 80)
    noisy_samples = generate_online_conformance_dataset(
        simulator,
        num_traces=1000,
        noise_type='remove',
        noise_level=0.3,
        balance_dataset=False
    )
    
    conf_count = sum(1 for _, _, label in noisy_samples if label == 0)
    non_conf_count = len(noisy_samples) - conf_count
    print(f"Total samples: {len(noisy_samples)}")
    print(f"Conformant: {conf_count} ({100*conf_count/len(noisy_samples):.1f}%)")
    print(f"Non-conformant: {non_conf_count} ({100*non_conf_count/len(noisy_samples):.1f}%)")
    
    # Generate additional negative samples
    print("\n3. Additional NEGATIVE samples:")
    print("-" * 80)
    negative_samples = generate_negative_samples(simulator, num_samples=200)
    print(f"Generated {len(negative_samples)} non-conformant samples")
    
    # Combined dataset
    print("\n4. COMBINED dataset:")
    print("-" * 80)
    all_samples = conforming_samples + noisy_samples + negative_samples
    conf_count = sum(1 for _, _, label in all_samples if label == 0)
    print(f"Total samples: {len(all_samples)}")
    print(f"Conformant: {conf_count} ({100*conf_count/len(all_samples):.1f}%)")
    print(f"Non-conformant: {len(all_samples)-conf_count} ({100*(len(all_samples)-conf_count)/len(all_samples):.1f}%)")
    
    # Balanced dataset
    print("\n5. BALANCED dataset:")
    print("-" * 80)
    balanced_samples = balance_samples(all_samples)
    conf_count = sum(1 for _, _, label in balanced_samples if label == 0)
    print(f"Total samples: {len(balanced_samples)}")
    print(f"Conformant: {conf_count} ({100*conf_count/len(balanced_samples):.1f}%)")
    print(f"Non-conformant: {len(balanced_samples)-conf_count} ({100*(len(balanced_samples)-conf_count)/len(balanced_samples):.1f}%)")
    
    # Show example samples
    print("\n6. Example samples:")
    print("-" * 80)
    for i in range(5):
        marking, event, label = balanced_samples[i]
        enabled = simulator.get_enabled_transitions(marking)
        status = "CONFORMANT" if label == 0 else "NON-CONFORMANT"
        print(f"Sample {i+1}:")
        print(f"  Marking: {marking}")
        print(f"  Enabled: {enabled}")
        print(f"  Event: {event}")
        print(f"  Label: {status}")
        print()
    
    # Save datasets to Parquet files
    print("\n7. Saving datasets to Parquet format:")
    print("-" * 80)
    
    # Save balanced dataset (recommended for training)
    save_dataset_polars(balanced_samples, "/kaggle/working/conformance_balanced.parquet")
    
    # Save full unbalanced dataset
    save_dataset_polars(all_samples, "/kaggle/working/conformance_full.parquet")
    
    # Save compact version (with marking as list column)
    save_dataset_polars_compact(balanced_samples, "conformance_balanced_compact.parquet")
    
    # Test loading
    print("\n8. Testing dataset loading:")
    print("-" * 80)
    loaded_samples = load_dataset_polars("/kaggle/working/conformance_balanced.parquet")
    print(f"Successfully loaded {len(loaded_samples)} samples")
    print(f"First sample: {loaded_samples[0]}")
    
    print("\n" + "=" * 80)
    print("Key Points:")
    print("=" * 80)
    print(" Each sample = (current_marking, incoming_event, conformance_label)")
    print(" Label 0: Event is enabled from current marking (conformant)")
    print(" Label 1: Event is NOT enabled (non-conformant, deviation detected)")
    print(" This simulates online scenario: events arrive one-by-one")
    print(" GNN will learn: marking + event → conformant or not")
    print(" This replaces A* shortest path algorithm with learned classifier")
    print(" Datasets saved in Parquet format for efficient storage and loading")