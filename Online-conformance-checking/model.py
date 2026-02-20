import torch
import torch.nn as nn
import torch.nn.functional as F

class PetriNetAlignmentPredictor(nn.Module):
    def __init__(self, reachability_tensor):
        super().__init__()
        self.register_buffer('omegas', reachability_tensor)
        self.num_m = reachability_tensor.shape[1]
        self.num_t = reachability_tensor.shape[0]

        # On concatène v_source et v_target en entrée
        self.network = nn.Sequential(
            nn.Linear(self.num_m * 2, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_t),
            nn.Softplus() 
        )

    def forward(self, v_current, v_target):
            # 1. Prédire les intensités (alphas)
            x = torch.cat([v_current, v_target], dim=-1)
            alphas = self.network(x) # [batch, num_t]
            # alphas = alphas * 2.0
            # 2. Initialiser la matrice de rotation totale comme l'Identité
            batch_size = v_current.size(0)
            R_total = torch.eye(self.num_m).unsqueeze(0).repeat(batch_size, 1, 1).to(v_current.device)
            
            # 3. Appliquer les rotations une par une (Produit matriciel)
            # On itère sur les transitions dans l'ordre de leur index (ou un ordre topologique)
            for t in range(self.num_t):
                # Omega individuel pour la transition t
                omega_t = self.omegas[t] # [num_m, num_m]
                
                # alpha pour cette transition
                a_t = alphas[:, t].view(-1, 1, 1) # [batch, 1, 1]
                
                # Matrice de rotation pour cette transition précise
                R_t = torch.matrix_exp(a_t * omega_t)
                
                # Composition : R_total = R_t * R_total (Ordre important !)
                R_total = torch.bmm(R_t, R_total)
            
            # 4. Appliquer la rotation finale cumulée
            v_final = torch.bmm(R_total, v_current.unsqueeze(-1)).squeeze(-1)
            
            return v_final, alphas
