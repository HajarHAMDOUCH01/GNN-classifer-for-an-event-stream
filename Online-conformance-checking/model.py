import torch
import torch.nn as nn
import torch.nn.functional as F

class PetriNetAlignmentPredictor(nn.Module):
    def __init__(self, reachability_tensor):
        super().__init__()
        # self.omegas: [num_t, num_m, num_m]
        self.register_buffer('omegas', reachability_tensor)
        self.num_m = reachability_tensor.shape[1]
        self.num_t = reachability_tensor.shape[0]

        # Le "step_network" décide de la transition à tirer à chaque pas
        self.network_step = nn.Sequential(
            nn.Linear(self.num_m * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_t),
            # nn.Softplus(),
            nn.Softmax(dim=0)
        )

    def forward(self, v_src, v_target, max_steps=5):
        # now v_src and v_target are one element from the dataset
        batch_size = v_src.shape[0]
        v_current = v_src
        predicted_sequence = []
        for k in range(max_steps):
            # 1. Analyse de l'état actuel vs cible
            x = torch.cat([v_current, v_target], dim=0)
            
            # 2. Prédiction des alphas pour ce pas précis
            alpha_k = self.network_step(x) # [batch, num_t]
            alpha_k = alpha_k.squeeze(0)
            print(alpha_k)
            # omega_k = torch.einsum('bt, tmk -> bmk', alpha_k, self.omegas)
            
            # omega_k pour un seul generateur de transition et deux markings src et target
            transition_idx = alpha_k.argmax().item()
            print(transition_idx)
            # 4. Exponentielle de matrice (Passage au Groupe de Lie)
            break
            R_k = torch.matrix_exp(omega_k)
            
            v_current = torch.bmm(R_k, v_current.unsqueeze(-1)).squeeze(-1)

            predicted_sequence.append(alpha_k)
            # print(f"v_current : {v_current}")
            # print(f"v_trgt : {v_target}")
            # if v_current == v_target:
            # break
        full_alpha_seq = torch.stack(predicted_sequence, dim=1)
        
        return v_current, full_alpha_seq