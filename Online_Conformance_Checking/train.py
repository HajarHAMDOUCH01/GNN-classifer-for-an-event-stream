import torch.optim as optim
import torch
import torch.nn.functional as F
import random
from model import PetriNetAlignmentPredictor
from reachability_graph import *
from dataset import *

model = PetriNetAlignmentPredictor(reachability_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def training_step(model, X_src_train, X_tgt_train, y_alphas_train, epoch):
    model.train()
    epoch_loss = 0

    tau = max(0.001, 1.0 - epoch * 0.005)  

    for i in range(len(X_src_train)):
        optimizer.zero_grad()

        v_s = X_src_train[i:i+1]
        v_t = X_tgt_train[i:i+1]
        y_true = y_alphas_train[i]

        if y_true.dim() == 3:
            y_true = y_true.squeeze(1)   # → [L, num_t]

        target_indices = y_true.argmax(dim=-1).long()  # [L]

        v_pred, pred_seq = model(v_s, v_t, training=True, tau=tau)
        

        min_len = min(pred_seq.size(0), target_indices.size(0))

        if min_len > 0:
            x_pred = torch.softmax(pred_seq, dim=-1).sum(dim=0)  # [num_t]
            # C = C.detach()
            # marking_vectors = marking_vectors.detach()
            m_src_places = marking_vectors[v_s.argmax().item()]  # [num_p]
            m_tgt_places = marking_vectors[v_t.argmax().item()]  # [num_p]
            residual = (m_tgt_places - m_src_places) - C @ x_pred
            loss_ce    = F.cross_entropy(pred_seq[:min_len], target_indices[:min_len])
            loss_pos   = F.mse_loss(v_pred, v_t)
            loss_state = residual.pow(2).mean()

            loss = loss_ce + 0.1 * loss_pos + 0.3 * loss_state
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(X_src_train)

epochs = 200
for epoch in range(epochs):
    avg_loss = training_step(model, X_src_train, X_tgt_train, y_alphas_train, epoch)  
    if epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

# Évaluation
model.eval()

total_runs = 10
global_success = 0
global_total = 0
global_failed_cases = []

for run in range(total_runs):
    # reshuffle test set for this run
    indices = list(range(len(X_src_test)))
    random.shuffle(indices)

    run_success = 0
    run_total = 0
    run_failed = []

    with torch.no_grad():
        for i in indices:
            v_src_test = X_src_test[i:i+1]
            v_tgt_test = X_tgt_test[i:i+1]

            src_idx = v_src_test.argmax().item()
            tgt_idx = v_tgt_test.argmax().item()

            # dead state guard
            src_marking = all_markings[src_idx]
            tgt_marking = all_markings[tgt_idx]
            if src_marking not in reachability_graph or len(reachability_graph[src_marking]) == 0:
                continue  # already handled, skip silently

            run_total += 1
            global_total += 1

            v_pred, pred_logits = model(v_src_test, v_tgt_test, training=False)

            cos = F.cosine_similarity(v_pred, v_tgt_test, dim=-1).item()
            success = cos >= 0.99

            if success:
                run_success += 1
                global_success += 1
            else:
                seq = [pred_logits[s].argmax().item() for s in range(pred_logits.size(0))]
                run_failed.append({
                    'src': src_marking,
                    'tgt': tgt_marking,
                    'cos': cos,
                    'seq': seq
                })
                global_failed_cases.append((run, src_idx, tgt_idx, cos))

    print(f"\nRun {run+1:2d} | Success: {run_success}/{run_total} "
          f"({100*run_success/run_total:.1f}%)")

    for f in run_failed:
        t_names = [all_transition_names[t] for t in f['seq']]
        print(f"  FAILED src={f['src']} -> tgt={f['tgt']} "
              f"cos={f['cos']:.4f} seq={t_names}")

# global summary
print(f"\n{'='*50}")
print(f"Global success: {global_success}/{global_total} "
      f"({100*global_success/global_total:.1f}%) over {total_runs} runs")

if global_failed_cases:
    print(f"\nAll failed pairs (run, src, tgt, cos):")
    for case in global_failed_cases:
        print(f"  run={case[0]+1} src={case[1]} tgt={case[2]} cos={case[3]:.4f}")
else:
    print("\nNo failures across all runs.")