import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad

# Questa è la Loss del paper CoMM
class CoMMLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in CoMM [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None): # NB di default si fa una media semplice tra embeddings mettendo weights=None
        super().__init__()
        self.temperature = temperature # Temperatura per scalare le similarità (più bassa => distribuzione più "sharp")
        self.weights = weights         # Pesi opzionali per combinare le diverse perdite (utile se voglio ponderare gli embeddings)
        self.INF = 1e8                 # Costante grande per mascherare la diagonale (auto-similarità)

    def infonce(self, z1, z2):
        """
        Calcola la InfoNCE loss (contrastiva) tra due batch di embeddings z1 e z2.
        """
        N = len(z1)

        # Similarità intra-modale (z1 vs z1 e z2 vs z2)
        # sim intra-z1
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        # sim intra-z2
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        
        # Similarità inter-modale (z1 vs z2)
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        
        # Rimuove la diagonale (simile a "self-similarity") penalizzandola con -inf (softmax poi => 0)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)

        # Costruisce la matrice combinata di similarità (sim_Z):
        # | z1 vs z2 | z1 vs z1 |
        # | z2 vs z2 | z2 vs z1 |
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        
        # Applica la log-softmax riga per riga
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        # La loss si ottiene considerando la diagonale come ground-truth
        loss = - torch.diag(log_sim_Z).mean()

        # Calcola anche l'accuratezza SSL (self-supervised learning):
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug. --> lista di embeddings da prima augmentazione/modalità
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug. --> lista di embeddings da seconda augmentazione/modalità
                - "prototype", integer indicating where the multimodal representation Z --> indice dell'embedding considerato come "prototipo" per l'allineamento.
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        assert len(z1) == len(z2)
        n_emb = len(z1)  # numero di diversi embeddings/modalità

        # Normalizza tutti gli embeddings (L2)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]

        # Concatena tutte le modalità e distribuisce su tutti i processi (multi-GPU)
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:] # split in aug1 e aug2

        # Calcola la InfoNCE loss tra ciascun embedding e il prototipo
        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb):
            loss1, acc1 = self.infonce(z1[i], z2[prototype]) # aug1[i] vs aug2[prototype]
            loss2, acc2 = self.infonce(z2[i], z1[prototype]) # aug2[i] vs aug1[prototype]
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
        
        # Crea dict per loggare le metriche separatamente
        ssl_acc = {"ssl_acc_%i"%i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i"%i: l for i, l in enumerate(loss)}

        # Se presenti, applica i pesi alle loss --> di default weights = None
        if self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss)) # Fa una media delle Loss
        acc = torch.mean(torch.stack(acc))

        # Ritorna tutto per il logger
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)