import sys
import os
sys.path.append("../") # aggiunge la directory padre del progetto all'elenco dei path da cui Python può importare moduli
import numpy as np
import torch
import matplotlib.pyplot as plt
import textwrap # modulo standard usato per formatting del testo
import torch.nn as nn
from pytorch_lightning import Trainer # Importa il Trainer di PyTorch Lightning, un framework che semplifica il training loop
from pytorch_lightning import LightningDataModule

# Import locali necessari NB: presi dal progtto CoMM
from pl_modules.comm import CoMM # Apprendimento multimodale contrastivo che consente la comunicazione tra più modalità in un unico spazio multimodale (Modello multimodale principale) [Importa classe CoMM da pl_modules.comm]
from models.mmfusion import MMFusion # Classe che fonde le rappresentazioni visive e testuali in una rappresentazione multimodale (Fusione avanzata di feature multimodali). [Importa MMFusion]
from models.vit import VisionTransformer # Encoder per immagini basato su ViT. [Importa una versione Pre-trained vision transformers personalizzata del Vision Transformer (ViT)]
from models.transformer import LanguageEncoder # Encoder per testo. [Importa LanguageEncoder, probabilmente una variante di Transformer per l'elaborazione del testo]

# MMFusion si occupa di orchestrare la fusione dei token provenienti da più modalità.
# Gli encoders elaborano le caratteristiche grezze dei dati (immagini, testo).
# Gli input_adapters (ProjectionAdapter) convertono i tensori di feature estratti dagli encoder in token con dimensioni uniformi.

# Import aggiuntivi
import clip # Per farlo funzionare bisogna eseguire: pip install git+https://github.com/openai/CLIP.git 
from torch.utils.data import DataLoader
from cc3m_llava import CC3MLLaVaDataset  # Sfrutta il custom dataset/dataloader CC3M fornito da MISTRETTA

from transformers import logging as hf_logging
hf_logging.set_verbosity_error() # Silenzia warning di Hugging Face # --> NB ci sono degli warning dovuti al fatto che Some weights of the model checkpoint at openai/clip-vit-base-patch32 were not used when initializing CLIPTextModel
    
torch.manual_seed(42) # Seed casuali per riproducibilità (seme randomico globale di PyTorch a 42)
np.random.seed(42) 
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid HF's warning 

_, clip_preprocess = clip.load('ViT-B/32') # Carica il preprocessore CLIP

# Crea il DataModule per CC3M # --> similmente a come gli autori di CoMM definivano MMIMDBDataModule (in mmimdb.py)
# Configura i DataLoader per training/validation con batch size e multi-threading.
class CC3MDataModule(LightningDataModule):
    def __init__(self, dataroot, preprocess, batch_size=64, num_workers=16): # Batch size di 64
        super().__init__()
        self.dataroot = dataroot
        self.preprocess = preprocess # Trasformazioni per immagini (CLIP)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        self.dataset = CC3MLLaVaDataset(
            dataroot=self.dataroot,
            preprocess=self.preprocess,
            return_image=True
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

# Istanziazione e configurazione Dataloader
data_module = CC3MDataModule(
    dataroot="/andromeda/datasets",
    preprocess=clip_preprocess,
    batch_size=64, # Batch size di 64
    num_workers=16
)

# Adapter necessario per poter utiizzare il Language encoder (In: 512 --> Out: 768)
# Serve per allineare dimensioni diverse (ie. testo da 512 a 768 per matchare l'encoder immagini).
class ProjectionAdapter(torch.nn.Module):
    def __init__(self, in_dim=512, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) # Proiezione lineare 512 → 768
        
    def forward(self, x):
        return self.proj(x)

# Funzione per preparare i batch nel formato che CoMM si aspetta
# CoMM si aspetta coppie di viste aumentate di ogni modalità per contrastive learning
# Per semplicità, usiamo le stesse immagini e testi per entrambe le viste
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    # Ritorna (x1, x2) dove ogni xi è una lista di modalità
    return ([images, texts], [images, texts]) # Due viste identiche (augmentation simulata)

# Aggiorna i DataLoader per usare la collate_fn (grazie a lambda per fare un override veloce)
data_module.train_dataloader = lambda: DataLoader(
    data_module.dataset,
    batch_size=data_module.batch_size,
    num_workers=data_module.num_workers,
    shuffle=True,
    collate_fn=collate_fn
)

data_module.val_dataloader = lambda: DataLoader(
    data_module.dataset,
    batch_size=data_module.batch_size,
    num_workers=data_module.num_workers,
    shuffle=False,
    collate_fn=collate_fn
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
                model_name="openai/clip-vit-base-patch32", # CLIP text encoder con output a 512 dimensioni → proiettato a 768 con ProjectionAdapter
                #model_name="sentence-transformers/clip-ViT-B-32", # Originale
                #model_name="clip-ViT-B-32-multilingual-v1",
                output_value="token_embeddings",
                normalize_embeddings=True,
                use_dataset_cache=False,
                mask_prob=0.15,
                freeze=True
            )
        ],
        # input_adapters=[None, None], originale
        input_adapters=[None, ProjectionAdapter(512, 768)], # --> necessario un adapter per il LenguageEncoder che riporti ad una dimensione congrua con embed_dim=768
        embed_dim=768 # Originale # Questo 768 è dovuto a come è strutturato il nostro codice
    ),
    # MLP che mappa 768D → 256D per la loss contrastiva (Projection Head).
    projection=CoMM._build_mlp(768, 512, 256), # Originale
    optim_kwargs=dict(lr=1e-4, weight_decay=1e-2),
    loss_kwargs=dict(temperature=0.1)
)

# Funzione per valutazione contrastiva 
# Funzione evaluate_recall_at_k misura quanto bene il modello riesce a recuperare la modalità target corretta (immagine o testo) in un retrieval task contrastivo
def evaluate_recall_at_k(model, dataloader, k_values=[10, 50]):
    model.eval() # Modello viene messo in modalità eval per disattivare dropout, batch norm, e altre funzioni di training
    img_embeddings, text_embeddings = [], [] # Serve a collezionare tutti gli embeddings di immagini e testi estratti dal dataloader

    # Loop sul dataloader (no gradienti)
    with torch.no_grad():
        # NB: La seconda vista (x2) rimane lì per coerenza con il dataloader definito in fase di training. # Per la valutazione serve solo una vista (x1)
        for (x1, x2) in dataloader:
            images = x1[0].to(model.device)
            texts = x1[1]
            # x1 è una lista con due elementi: x1[0] batch di immagini, x1[1] batch di testi
            
            # Estrazione embeddings:
            # Image embeddings
            img_features = model.encoder([images, texts], mask_modalities=[[True, False]])[0] # mask_modalities=[[True, False]] indica che vogliamo solo la modalità immagine
            img_features = model.head(img_features)
            img_embeddings.append(img_features)

            # Text embeddings
            #text_inputs = model.encoder.encoders[1].tokenize(texts).to(model.device) # model.encoder.encoders[1] è il LanguageEncoder
            text_inputs = texts
            text_features = model.encoder([images, text_inputs], mask_modalities=[[False, True]])[0] # mask_modalities=[[False, True]] seleziona solo il testo
            text_features = model.head(text_features)
            text_embeddings.append(text_features)

    # Concatenazione dei batch --> Creo due matrici: una per immagini (N, D) e una per testi (N, D)
    img_embeddings = torch.cat(img_embeddings)
    text_embeddings = torch.cat(text_embeddings)

    # Similarità coseno
    sim = img_embeddings @ text_embeddings.T # Calcola la matrice di similarità tra immagini e testi, usando il prodotto scalare

    # Label ground-truth
    labels = torch.arange(len(img_embeddings)).to(model.device) # Costruisce le etichette target: immagine i corrisponde al testo i

    results = {}

    # Calcolo Recall@k
    for k in k_values:
        # Image-to-Text Recall@k
        _, indices_img2txt = sim.topk(k, dim=1) # Prende i k testi più simili per ogni immagine
        recall_img2txt = (indices_img2txt == labels.view(-1, 1)).any(dim=1).float().mean().item() # Verifica se il testo corretto è tra questi

        # Text-to-Image Recall@k
        _, indices_txt2img = sim.topk(k, dim=0) # Prende le k immagini più simili per ogni testo
        recall_txt2img = (indices_txt2img == labels.view(1, -1)).any(dim=0).float().mean().item() # Verifica se l’immagine corretta è tra queste

        # Salvataggio dei risultati
        results[f"Recall@{k}"] = {
            "img2txt": recall_img2txt,
            "txt2img": recall_txt2img,
            "mean": (recall_img2txt + recall_txt2img) / 2 # Mean Recall
        }

    # Ritorna un dizionario
    return results

# Training del modello
trainer = Trainer(max_epochs=1) #(max_epochs=70) # 70 epoche di Train (NB: PyTorch Lightning gestisce automaticamente il loop di training)
trainer.fit(comm, datamodule=data_module) # Usa loss contrastiva (implicitamente definita in CoMM)

# Valutazione del modello
# evaluate_recall_at_k calcola l'accuracy di retrieval (quanto bene il modello abbina immagini a testi corrispondenti)
# Metrica usata è la Recall@k con k =10 e k=50.
recall_results = evaluate_recall_at_k(comm.to("cuda"), data_module.val_dataloader(), k_values=[10, 50])
for k, metrics in recall_results.items():
    print(f"Recall@{k}: img2txt={metrics['img2txt']*100:.2f}%, txt2img={metrics['txt2img']*100:.2f}%, mean={metrics['mean']*100:.2f}%")

print(f"CoMM end")

# ToDo:
# Aggiungere data augmentation per immagini/testi.
# Per testo --> devo far funzionare lo script 'panoptic-segment-anything' da poi passare a DAM





