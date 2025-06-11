import sys
import os
sys.path.append("../") # aggiunge la directory padre del progetto all'elenco dei path da cui Python può importare moduli
import numpy as np
import torch
import matplotlib.pyplot as plt
import textwrap # modulo standard usato per formatting del testo
import torch.nn as nn
import time
import datetime
from pytorch_lightning import Trainer # Importa il Trainer di PyTorch Lightning, un framework che semplifica il training loop
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Callback # Serve per fare callback durante il training per ottenere print
from pytorch_lightning.callbacks import EarlyStopping # Da implementare
from torch.utils.data import random_split # Serve per splittare il dataset
from pytorch_lightning.loggers import TensorBoardLogger # Serve per loggare la loss in modo da monitorarla
from pytorch_lightning.callbacks import ModelCheckpoint # Serve per salvare i checkpoint

# Import locali necessari NB: presi dal progtto CoMM
from pl_modules.comm import CoMM # Apprendimento multimodale contrastivo che consente la comunicazione tra più modalità in un unico spazio multimodale (Modello multimodale principale) [Importa classe CoMM da pl_modules.comm]
from models.mmfusion import MMFusion # Classe che fonde le rappresentazioni visive e testuali in una rappresentazione multimodale (Fusione avanzata di feature multimodali). [Importa MMFusion]
from models.vit import VisionTransformer # Encoder per immagini basato su ViT. [Importa una versione Pre-trained vision transformers personalizzata del Vision Transformer (ViT)]
from models.transformer import LanguageEncoder # Encoder per testo. [Importa LanguageEncoder, probabilmente una variante di Transformer per l'elaborazione del testo]

# Import aggiuntivi
import clip # Per farlo funzionare bisogna eseguire: pip install git+https://github.com/openai/CLIP.git 
from torch.utils.data import DataLoader
from cc3m_llava import CC3MLLaVaDataset  # Sfrutta il custom dataset/dataloader CC3M fornito da MISTRETTA

from transformers import logging as hf_logging
hf_logging.set_verbosity_error() # Silenzia warning di Hugging Face # --> NB ci sono degli warning dovuti al fatto che Some weights of the model checkpoint at openai/clip-vit-base-patch32 were not used when initializing CLIPTextModel

import warnings
warnings.filterwarnings("ignore", message=".*nvrtc.so.*")

torch.manual_seed(42) # Seed casuali per riproducibilità (seme randomico globale di PyTorch a 42)
np.random.seed(42) 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid HF's warning 

_, clip_preprocess = clip.load('ViT-B/32') # Carica il preprocessore CLIP

# Funzione per preparare i batch nel formato che CoMM si aspetta
# CoMM si aspetta coppie di viste aumentate di ogni modalità per contrastive learning
# Per semplicità, usiamo le stesse immagini e testi per entrambe le viste
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    # Ritorna (x1, x2) dove ogni xi è una lista di modalità
    return ([images, texts], [images, texts]) # Due viste identiche (augmentation simulata)

# Crea il DataModule per CC3M # --> similmente a come gli autori di CoMM definivano MMIMDBDataModule (in mmimdb.py)
# Configura i DataLoader per training/validation con batch size e multi-threading.
class CC3MDataModule(LightningDataModule):
    def __init__(self, dataroot, preprocess, batch_size=64, num_workers=16, val_split=0.1):
        super().__init__()
        self.dataroot = dataroot
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
    def setup(self, stage=None):
        full_dataset = CC3MLLaVaDataset(
            dataroot=self.dataroot,
            preprocess=self.preprocess,
            return_image=True
        )
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        train_size = total_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Per riproducibilità
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn
        )


# Istanziazione e configurazione Dataloader
data_module = CC3MDataModule(
    dataroot="/andromeda/datasets",
    preprocess=clip_preprocess,
    batch_size=64,
    num_workers=16,
    val_split=0.1  # Il 10% del dataset sarà usato per fare per validation
)

# Crea il modello CoMM standard
comm = CoMM(
    # MMFusion unisce i token multimodali in uno spazio comune (768D)
    encoder=MMFusion(
        encoders=[
            VisionTransformer( 
                model_name="vit_base_patch32_clip_224.openai", # ViT preaddestrato (CLIP) con output a 768 dimensioni
                pretrained=True,
                output_value="token_embeddings",
                freeze=True
            ),
            LanguageEncoder( 
                #model_name="openai/clip-vit-base-patch32", # CLIP text encoder con output a 512 dimensioni → proiettato a 768 con ProjectionAdapter
                model_name="clip-ViT-B-32-multilingual-v1", # Originale
                output_value="token_embeddings",
                normalize_embeddings=True,
                use_dataset_cache=False,
                mask_prob=0.15,
                freeze=True
            )
        ],
        input_adapters=[None, None], # originale
        #input_adapters=[None, ProjectionAdapter(512, 768)], # --> necessario un adapter per il LenguageEncoder che riporti ad una dimensione congrua con embed_dim=768
        embed_dim=768 # Originale # Questo 768 è dovuto a come è strutturato il nostro codice
    ),
    # MLP che mappa 768D → 256D per la loss contrastiva (Projection Head).
    projection=CoMM._build_mlp(768, 512, 256), # Originale
    optim_kwargs=dict(lr=1e-4, weight_decay=1e-2),
    loss_kwargs=dict(temperature=0.1) # Iperparametro per loss contrastiva di tipo InfoNCE
)

#print(f"[INFO] Struttura della head di comm: {comm.head}")
#print(f"[INFO] Struttura encoder di comm: {comm.encoder}")
"""
for name, param in comm.encoder.named_parameters():
    if param.requires_grad:
        print(name, param.numel()) # Print per vedere esattamente quali layer sono allenabili in PyTorch
"""

# Definisce callback di pytorch_lightning per avere i printo su Batch size, e info sui parametri
class PrintParamsCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        batch_size = trainer.datamodule.batch_size
        print(f"[INFO] Batch size: {batch_size}")

        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        print(f"[INFO] Totale parametri: {total_params}")
        print(f"[INFO] Parametri addestrabili: {trainable_params}")
        print(f"[INFO] Parametri frozen: {total_params - trainable_params}")

# Inizio calcolo del tempo
start_time = time.time()

# Prima di chiamare trainer.fit(), chiama setup esplicitamente (opzionale, ma consigliato)
data_module.setup()

# Training del modello
#trainer = Trainer(max_epochs=1)  #(max_epochs=70) # 70 epoche di Train (NB: PyTorch Lightning gestisce automaticamente il loop di training)
num_epochs=1 # Indica il numero di epoche che verranno eseguite
# Provate num_epochs= 10, 20
trainer = Trainer(
    max_epochs=num_epochs, # Numero di epoche che verrano eseguite
    callbacks=[PrintParamsCallback()] # Serve per avere i print durante l'esecuzione
)
trainer.fit(comm, datamodule=data_module) # Usa loss contrastiva (implicitamente definita in CoMM). NB: avendo fornito validation_step e val_dataloader nel DataModule, la validazione viene eseguita automaticamente ad ogni fine epoca durante il training

# Fine calcolo del tempo
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time))) # Rende più leggibile il tempo trascorso
print(f"[INFO] Tempo totale di esecuzione: {elapsed_str}")

print(f"CoMM end")


