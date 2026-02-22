import torch.optim as optim
import torch 
import torch.nn.functional as F
from model import PetriNetAlignmentPredictor

import torch.optim as optim

from reachability_graph_construction import *
from dataset import *

model = PetriNetAlignmentPredictor(reachability_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0018)

import torch.optim as optim
import torch 
import torch.nn.functional as F

# try with batch size = 1 in training and inference : chaque observation de la trace

def training_step(model, v_src, v_tgt, y_true_seq, epoch):
    model.train()
    optimizer.zero_grad()
    train_data_length = len(v_src)
    for i in range(train_data_length):
        v_src_element = v_src[i:i+1]
        v_tgt_element = v_tgt[i:i+1]
        v_src_element = v_src_element.squeeze(0)
        v_tgt_element = v_tgt_element.squeeze(0)
        # v_pred: [Batch, Num_M], pred_seq: [Batch, Steps, Num_T]
        v_pred, pred_seq = model(v_src_element, v_tgt_element)
        break
        
        loss_alpha = F.mse_loss(pred_seq, y_true_seq)# to do : this should not be mse and loss should not be for total sequence
        
        total_loss = loss_alpha
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        return total_loss.item()


# Boucle d'entraînement
epochs = 1 
for epoch in range(epochs):
    loss = training_step(model, X_src_train, X_tgt_train, y_alphas_train, epoch)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f} ")


model.eval()
with torch.no_grad():
    for i in range(len(X_src_test)):
        v_src_test = X_src_test[i:i+1]
        v_tgt_test = X_tgt_test[i:i+1]
        
        # v_pred est le résultat de : R_n * ... * R_1 * v_src
        v_pred, pred_alphas = model(v_src_test, v_tgt_test)
        
        print(f"\n--- Test Chemin {i} ---")
        print(f"source (Index) : {v_src_test.argmax().item()}")
        print(f"Cible réelle (Index) : {v_tgt_test.argmax().item()}")
        print(f"Prédit (Index)       : {v_pred.argmax().item()}")
        
        if pred_alphas.dim() > 2:
            for step in range(pred_alphas.size(1)):
                step_alphas = pred_alphas[0, step]
                top_t = step_alphas.argmax().item()
                if step_alphas[top_t] > 0.2:
                    print(f"  Pas {step+1}: Transition t{top_t+1} (alpha={step_alphas[top_t]:.2f})")
                break
        else:
            print(f"Alphas globaux : {pred_alphas.squeeze().tolist()}")
        break
