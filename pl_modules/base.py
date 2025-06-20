from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Tuple, Dict
from abc import ABC, abstractmethod
import torch
import math
import sys
from utils import set_weight_decay_per_param, LinearWarmupCosineAnnealingLR

import torch.nn.functional as F # Aggiunto

class BaseModel(ABC, LightningModule):
    """
        Modello base per modelli Self-Supervised Learning (SSL), Vision-Language (VL) o Language-Guided (LG).
        È pensata per essere estesa da classi figlie che implementano un estrattore di features.
        We expect any `BaseModel` to implement a features extractor.
    """

    def __init__(self, optim_kwargs: Dict):
        """
        Args:
            optim_kwargs: Dizionario contenente iperparametri di ottimizzazione come learning rate e weight decay.
        """
        super().__init__()
        self.optim_kwargs = optim_kwargs

    def configure_optimizers(self):
        """
        Definisce l'ottimizzatore e l'eventuale scheduler del learning rate.
        """
        # Imposta AdamW come ottimizzatore e applica il weight decay in modo selettivo ai parametri del modello.
        optimizer = torch.optim.AdamW(
            set_weight_decay_per_param(
                self, weight_decay=self.optim_kwargs["weight_decay"]),
            lr=self.optim_kwargs["lr"])

        # Se specificato, aggiunge un learning rate scheduler (warmup + cosine annealing)
        if "lr_scheduler" in self.optim_kwargs:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.optim_kwargs["lr_scheduler"]["warmup_epochs"],
                max_epochs=self.trainer.max_epochs,
                warmup_start_lr=self.optim_kwargs["lr_scheduler"]["start_warmup_value"],
                eta_min=self.optim_kwargs["lr_scheduler"]["final_value"]
            )
            # Ritorna la lista di ottimizzatori e scheduler da usare
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Step di training: calcola la loss, la logga e ritorna il valore.
        """
        outputs = self.forward(*batch) # Forward pass del modello (input e/o augmented)
        out_dict = self.loss(outputs)  # Calcola la loss
        loss = out_dict['loss']        # Estrae la loss principale
        
        # Se la loss non è finita, interrompe l'allenamento
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # Calcola il batch_size correttamente per logging
        # Per essere sicuro che prenda il batch_size giusto (o meglio, usa la dimensione effettiva del batch)-ed eliminare gli warnings
        batch_size = batch[0][0].size(0)  # Prende la dimensione del primo tensore nel batch
        #print(f"[INFO] batch_size usato nel train: {batch_size}") # Print per essere sicuro del Batch_size

        # Logga tutte le metriche (loss + eventuali altre) con il batch_size corretto
        self.log_dict(out_dict, on_step=True, sync_dist=True, prog_bar=True, batch_size=batch_size)
        #self.log_dict(out_dict, on_step=True, sync_dist=True, prog_bar=True)
        
        return loss
    
    # Validation modificato rispetto all'originale al fine di poter calcolare anche Recall@K
    def validation_step(self, batch, batch_idx):
        # batch = [images, texts]
        # Duplico la vista per simulare le due viste richieste dal forward
        modalities = batch
        outputs = self.forward(modalities, modalities)

        out_dict = self.loss(outputs)
        val_loss = out_dict['loss']

        batch_size = modalities[0].size(0)
        self.log_dict(
            {f"val_{k}": v for k, v in out_dict.items()},
            on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch_size
        )

        img_embeds = outputs['aug1_embed'][0]  # Immagini
        txt_embeds = outputs['aug2_embed'][1]  # Testi

        output = {
            'val_loss': val_loss,
            'img_embeds': img_embeds,
            'txt_embeds': txt_embeds,
        }

        self.val_outputs.append(output)

        return output

    def test_step(self, batch, batch_idx): # ToDo --> implementare la stessa logica di validation, ovvero il calcolo della metrica Recall@K
        """
        Step di test: calcola la loss e la logga.
        """
        outputs = self.forward(*batch)
        out_dict = self.loss(outputs)
        test_loss = out_dict['loss']
        # Logga tutte le metriche (loss + eventuali altre)
        self.log_dict({"test_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True)
        return test_loss

    @abstractmethod
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs) \
            -> Tuple[Tensor, Tensor]:
        """
        Metodo astratto da implementare nelle classi figlie: estrae features visive (o multimodali).
        Extract global average pooled visual features.
        Args:
            loader: Dataset loader to serve ``(image, label)`` tuples. Dataloader che fornisce tuple (X, y)
        Returns:
            Pair (X,y) corresponding to extracted features and corresponding labels
            X: Tensor di features estratte
            y: Tensor di etichette corrispondenti
        """
        pass

    # New sovrascrive la funzione di LightningModule
    def on_validation_start(self):
        self.val_img_embeds = []
        self.val_txt_embeds = []
        self.val_outputs = []  # conterrà tutti i dizionari restituiti da validation_step

    # New serve per il calcolo della Recall@k e per loggare le metriche
    def compute_and_log_recall(self, sim_matrix, k_list=[1, 5, 10]):
        # Assunzioni importanti:
        # - batch è ordinato e allineato: img[i] ↔ txt[i]
        # - sim_matrix è simmetrica nel caso ideale
        # - embedding sono già normalizzati prima del calcolo con @ (dot product)
        targets = torch.arange(sim_matrix.size(0))  # target corretti (posizione originale), ie targets = [0, 1, 2, ..., N-1] = indici delle corrispondenze corrette # Serve come etichetta “corretta”: si assume che l'immagine i corrisponda al testo i

        # TEXT -> IMAGE retrieval
        sim_matrix_T = sim_matrix.T  # [N, N] # inverte righe e colonne così ora ogni testo è una riga
        _, retrieved_img = sim_matrix_T.topk(max(k_list), dim=1)  # [N, max_k] # trova le topk immagini più simili rispetto a un testo

        recalls = {}
        for k in k_list:
            # retrieved_img[i, :] = top-k immagini per il testo i
            correct = (retrieved_img[:, :k] == targets.unsqueeze(1)).any(dim=1).float()  # Si controlla se trai topk rientra quello corretto
            recalls[f"Recall@{k}_text2img"] = correct.mean().item() # La media su tutti gli esempi fornisce Recall@K per text→img

        # IMAGE -> TEXT retrieval
        _, retrieved_txt = sim_matrix.topk(max(k_list), dim=1)  # [N, max_k] # trova le topk testi più simili rispetto a una immagine

        for k in k_list:
            correct = (retrieved_txt[:, :k] == targets.unsqueeze(1)).any(dim=1).float() # Si controlla se trai topk rientra quello corretto
            recalls[f"Recall@{k}_img2text"] = correct.mean().item()

        # Recall@K = % di volte in cui il testo corretto è tra i primi k risultati

        # Logging (Lightning logger sempre, print solo dopo epoca 0)
        for key, val in recalls.items():
            self.log(key, val, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Print dei risultati
        if not self.trainer.sanity_checking and self.trainer.is_global_zero: # Evita stampe delle metriche durante il sanity check
            for key, val in recalls.items():
                print(f"{key}: {val:.4f}")
        

    # New sovrascrive la funzione di LightningModule
    def on_validation_epoch_end(self):
        # Raggruppa tutti gli embeddings accumulati durante validation_step
        img_embeds = torch.cat([x["img_embeds"] for x in self.val_outputs], dim=0)  # [N, D]
        txt_embeds = torch.cat([x["txt_embeds"] for x in self.val_outputs], dim=0)  # [N, D]
        
        # Normalizza (consigliato visto che usiamo dot-product come similarità)
        img_embeds = F.normalize(img_embeds, dim=1)
        txt_embeds = F.normalize(txt_embeds, dim=1)

        # Sposta tutto su CPU per salvataggio, GPU per il calcolo
        txt_embeds = txt_embeds.to(self.device)  # solo una volta, sta fermo
        sim_matrix_rows = []

        batch_size_sim = 256   # regola questo valore in base alla tua GPU (es. 256 o 512). Da tenere il più alto possibile in base alla GPU

        # Calcolo matrice di similarità per strisce (ie. alcune img per tutti i testi)
        for i in range(0, img_embeds.size(0), batch_size_sim):
            img_batch = img_embeds[i:i+batch_size_sim].to(self.device)  # [B, D]
            sim_batch = img_batch @ txt_embeds.T  # [B, N]
            sim_matrix_rows.append(sim_batch.cpu())  # sposta su CPU e salva

            # Libera la GPU subito
            del img_batch, sim_batch
            torch.cuda.empty_cache()

        # Ricostruisci la matrice finale [N, N] in CPU
        sim_matrix = torch.cat(sim_matrix_rows, dim=0)
        diag = sim_matrix.diag()
        print(f"[DEBUG] Similarità diagonale: min={diag.min():.4f}, max={diag.max():.4f}, mean={diag.mean():.4f}") # Per vedere cosa c'è sulla diagonale della matrice

        # Ora puoi calcolare le metriche di retrieval, es. Recall@K
        self.compute_and_log_recall(sim_matrix)




