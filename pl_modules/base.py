from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Tuple, Dict
from abc import ABC, abstractmethod
import torch
import math
import sys
from utils import set_weight_decay_per_param, LinearWarmupCosineAnnealingLR


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

    def validation_step(self, batch, batch_idx):
        """
        Step di validazione: calcola la loss e la logga.
        """
        outputs = self.forward(*batch)
        out_dict = self.loss(outputs)
        val_loss = out_dict['loss']

        # Calcola il batch_size correttamente per logging
        # Per essere sicuro che prenda il batch_size giusto (o meglio, usa la dimensione effettiva del batch)-ed eliminare gli warnings
        batch_size = batch[0][0].size(0)  # Prende la dimensione del primo tensore nel batch
        #print(f"[INFO] batch_size usato nel validation: {batch_size}") # Print per essere sicuro del Batch_size
        
        # Logga tutte le metriche (loss + eventuali altre) con il batch_size corretto
        self.log_dict({"val_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=batch_size)
        #self.log_dict({"val_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
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

