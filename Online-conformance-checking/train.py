import torch.optim as optim
import torch 
import torch.nn.functional as F
from model import PetriNetAlignmentPredictor

import torch.optim as optim

from reachability_graph_construction import *
from dataset import *
# Initialisation
model = PetriNetAlignmentPredictor(reachability_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def training_step(model, v_src, v_tgt, y_true_alphas, lambda_l1=0.05):
    model.train()
    optimizer.zero_grad()
    
    # Forward
    v_pred, pred_alphas = model(v_src, v_tgt)
    
    # 1. Perte de position (Reconstruction)
    loss_pos = F.mse_loss(v_pred, v_tgt)
    
    # 2. Perte de parcimonie (on veut des alphas petits/rares)
    loss_sparsity = torch.mean(torch.sum(pred_alphas, dim=1))
    
    # 3. Optionnel : Perte sur les alphas eux-mêmes (Supervision directe)
    loss_alpha_direct = F.mse_loss(pred_alphas, y_true_alphas)
    
    total_loss = loss_pos + (lambda_l1 * loss_sparsity) + loss_alpha_direct
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), loss_pos.item()


epochs = 1000
for epoch in range(epochs):
    loss, pos_err = training_step(model, X_src_train, X_tgt_train, y_alphas_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Total Loss: {loss:.6f} | Pos Error: {pos_err:.6f}")

print("\nEntraînement terminé.")


model.eval()
with torch.no_grad():
    v_test_pred, pred_alphas_test = model(X_src_test, X_tgt_test)
    # On compare v_test_pred avec X_tgt_test
    test_err = F.mse_loss(v_test_pred, X_tgt_test)
    print(f"Erreur moyenne sur le Test Set (chemins inconnus): {test_err.item():.6f}")