import torch.optim as optim
import torch 
import torch.nn.functional as F
from model import PetriNetAlignmentPredictor

import torch.optim as optim

from reachability_graph_construction import *
from dataset import *
# Initialisation
model = PetriNetAlignmentPredictor(reachability_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0018)

def training_step(model, v_src, v_tgt, y_true_alphas, lambda_l1=0.05):
    model.train()
    optimizer.zero_grad()
    
    # Forward
    v_pred, pred_alphas = model(v_src, v_tgt)
    
    # 1. Perte de position (Reconstruction)
    loss_pos = F.mse_loss(v_pred, v_tgt)
    lambda_l1 = 0.0 if epoch < 1000 else 0.02
    # 2. Perte de parcimonie (on veut des alphas petits/rares)
    loss_sparsity = torch.mean(torch.sum(pred_alphas, dim=1))
    
    # 3. Optionnel : Perte sur les alphas eux-mêmes (Supervision directe)
    loss_alpha_direct = F.mse_loss(pred_alphas, y_true_alphas)
    
    total_loss = loss_pos + loss_alpha_direct
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_pos.item()


epochs = 5000
for epoch in range(epochs):
    loss, pos_err = training_step(model, X_src_train, X_tgt_train, y_alphas_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Total Loss: {loss:.6f} | Pos Error: {pos_err:.6f}")

print("\nEntraînement terminé.")


model.eval()
with torch.no_grad():
    # On prend le premier exemple du test set
    # v_src = X_src_test[0:1]
    # v_tgt = X_tgt_test[0:1]
    # print(f"marking source : {v_src}")
    # print(f"marking cible : {v_tgt}")
    # target_alphas = y_alphas_test[0]
    for i in range(len(X_src_test)):
        v_src_test_element = X_src_test[i:i+1]
        v_tgt_test_element = X_tgt_test[i:i+1]
        y_alphas_test_element = y_alphas_test[i]
        v_pred, pred_alphas = model(v_src_test_element, v_tgt_test_element)
        
        print(f"\n--- Test sur un chemin plus court entre {v_src_test_element} et {v_tgt_test_element}---")
        formatted_real_alphas = [f"{x:.1f}" for x in y_alphas_test_element.tolist()]
        print(f"Alphas attendus (Dijkstra) : {formatted_real_alphas}")
        formatted_pred_alphas = [f"{x:.1f}" for x in pred_alphas.squeeze().tolist()]
        print(f"Alphas prédits (IA)       : {formatted_pred_alphas}")
        
        # On vérifie si le marquage final est le bon
        print(f"Index du marquage cible : {v_tgt_test_element.argmax().item()}")
        # print("\nv_pred :\n", v_pred)
        print(f"Index du marquage prédit: {v_pred.argmax().item()}")