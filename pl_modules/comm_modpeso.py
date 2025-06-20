from torch import nn
import torch
from collections import OrderedDict
from typing import Dict, List
# Local imports
from pl_modules.base import BaseModel
from losses.comm_loss import CoMMLoss
from models.mmfusion import MMFusion

from demo.cc3m_sim_mat_recallk import compute_similarity_matrix
from demo.cc3m_sim_mat_recallk import compute_recall_at_k
from torchmetrics.retrieval import RetrievalRecall # Aggiunto per ottenere la RecallK di pytorch

# Contrastive MultiModal learning (CoMM): Permette di apprendere rappresentazioni multimodali attraverso una singola rete (encoder multimodale) e una loss contrastiva.
class CoMM(BaseModel):
    """ Contrastive MultiModal learning allowing the communication between modalities 
    in a single multimodal space [1].
    
    It encodes a pair of mulitmodal data and outputs a pair of representations through
    a single multimodal encoder.

    [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self,
                 encoder: MMFusion,     # Encoder multimodale (fonda le modalità in uno spazio condiviso)
                 projection: nn.Module, # MLP per proiettare le features nello spazio latente, ovvero la rappresentazione di dati dove le features importanti sono distillate e compresse in uno spazio dimensionale più piccolo
                 optim_kwargs: Dict,    # # Dizionario di iperparametri per l'ottimizzazione (learning rate, etc.)
                 loss_kwargs: Dict):    # Dizionario di iperparametri per la loss CoMM
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        # Costruttore del modello: 
        super(CoMM, self).__init__(optim_kwargs)

        # create the encoder
        # Encoder multimodale: fonde le diverse modalità
        self.encoder = encoder # Inizializza l'encoder multimodale. Encoders hanno dei parametri freezati

        # build a 3-layers projector
        # Testa di proiezione: MLP che proietta le features in uno spazio latente
        self.head = projection # Costruisce la testa di proiezione (MLP). Il projectHead ha dei parametri da addestrare

        # Build the loss
        # Loss contrastiva per multimodalità
        self.loss = CoMMLoss(**loss_kwargs) # Inizializza la loss CoMM. La loss non ha parametri

        # Aggiunto Serve per ottenere la Recall@K cpm K = 1, 5, 10
        self.val_recalls_i2t = {
            1: RetrievalRecall(top_k=1),
            5: RetrievalRecall(top_k=5),
            10: RetrievalRecall(top_k=10),
        }
        self.val_recalls_t2i = {
            1: RetrievalRecall(top_k=1),
            5: RetrievalRecall(top_k=5),
            10: RetrievalRecall(top_k=10),
        }


    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim): # Noi usiamo:  projection=CoMM._build_mlp(768, 512, 256)
        """
        Costruisce un MLP a 3 layer con BatchNorm e ReLU.
        Usato per la testa di proiezione.
        Esempio chiamata: CoMM._build_mlp(768, 512, 256) # quindi in_dim=768, mlp_dim=512, out_dim=226
        """
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)), # BatchNorm sincronizzato utile nei modelli distribuiti
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))


    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]):
        # compute features for all modalities
        """
        Forward pass:
        - Prende due viste augmentate (x1 e x2) di dati multimodali
        - Calcola features multimodali tramite l'encoder
        - Passa le features attraverso la testa di proiezione
        - Restituisce un dizionario con:
            - 'aug1_embed' e 'aug2_embed' = features proiettate
            - 'prototype' = placeholder per eventuali clustering (non usato qui)
        """
        # Calcola tutte le possibili maschere per le modalità (masking multimodale)
        all_masks = self.gen_all_possible_masks(len(x1))

        # Encoder multimodale per x1 e x2
        z1 = self.encoder(x1, mask_modalities=all_masks)
        z2 = self.encoder(x2, mask_modalities=all_masks)

        # Passaggio attraverso la testa di proiezione
        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]

        # Restituisce un dizionario con embeddings
        return {'aug1_embed': z1,
                'aug2_embed': z2,
                "prototype": -1}
    
    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Genera tutte le possibili maschere di modalità:
        - Ogni maschera ha un solo True (modalità attiva) e gli altri False (modalità spente)
        - L'ultima maschera ha tutti True (tutte le modalità attive)
        Esempio:
            Per n_mod = 2 => masks ==[[True, False], [False, True], [True, True]]
            Per n_mod = 3 => masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks
    
    # Cambiato Extrcat_features perchè richiedeva etichette (modificato rispetto all'originale)
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
        Extract multimodal features (embedding) from the encoder for all data in the loader.
        Supporta sia batch in formato dizionario (default) che tuple usate nel contrastive learning.
        
        Returns:
            X: extracted features (tensor)
            names: image names (list of strings)
        """
        X = []
        names = []

        for batch in loader:
            # Supporto per entrambe le modalità di batch: dict o tuple
            if isinstance(batch, dict):
                images = batch.get("image", None)
                texts = batch.get("text", None)
                image_names = batch.get("image_name", None)
            elif isinstance(batch, tuple):
                x1, _ = batch  # batch = (x1, x2)
                images, texts = x1  # x1 = [images, texts]
                image_names = None
            else:
                raise ValueError("Formato batch non riconosciuto. Atteso dict o tuple.")

            modalities = []
            if images is not None:
                images = images.to(self.device)
                modalities.append(images)
            if texts is not None:
                if isinstance(texts, dict):  # Se è un batch tokenizzato
                    texts = {k: v.to(self.device) for k, v in texts.items()}
                else:
                    texts = texts.to(self.device)
                modalities.append(texts)

            with torch.inference_mode():
                output = self.encoder(modalities, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())

                if image_names is not None:
                    names.extend(image_names)

        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), names

    
    # Sovrascrive on_validation_epoch_end (di LightningModule) in CoMM
    # Sfrutta le funzioni 'compute_similarity_matrix' e 'compute_recall_at_k' per ottenere la recallK
    """
    def on_validation_epoch_end(self):
        val_loader = self.trainer.datamodule.val_dataloader()

        image_embeds, _ = self.extract_features(val_loader, mask_modalities=[[True, False]]) # Ritornerebbero anche i nomi ma non ci servono per il calolco delle similarità
        text_embeds, _ = self.extract_features(val_loader, mask_modalities=[[False, True]])  # Ritornerebbero anche i nomi ma non ci servono per il calolco delle similarità

        sim_matrix = compute_similarity_matrix(image_embeds, text_embeds)  # NxN

        # I2T = ogni riga: un'immagine come query, colonne = testi
        scores_i2t = sim_matrix
        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)  # Ground truth: diagonale

        for k, metric in self.val_recalls_i2t.items():
            metric.update(scores_i2t, targets)
            self.log(f"val/image_to_text/recall@{k}", metric.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
            metric.reset()

        # T2I = ogni riga: un testo come query, colonne = immagini
        scores_t2i = sim_matrix.T  # trasponi la matrice
        for k, metric in self.val_recalls_t2i.items():
            metric.update(scores_t2i, targets)
            self.log(f"val/text_to_image/recall@{k}", metric.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
            metric.reset()
    """



