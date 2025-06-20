from pytorch_lightning import LightningDataModule
import torch
from cc3m_llava import CC3MLLaVaDataset  # Sfrutta il custom dataset/dataloader CC3M fornito da MISTRETTA
from torch.utils.data import random_split # Serve per splittare il dataset
from torch.utils.data import DataLoader


# Funzione per preparare i batch nel formato che CoMM si aspetta
# CoMM si aspetta coppie di viste aumentate di ogni modalità per contrastive learning
# Per semplicità, usiamo le stesse immagini e testi per entrambe le viste
# ToDo aggiungi augmentation
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    
    # Ritorna (x1, x2) dove ogni xi è una lista di modalità
    return ([images, texts], [images, texts]) # Due viste identiche (augmentation simulata)

# At inference time CoMM utilizza viste non augmentate di ogni modalità
def collate_fn_noaugm(batch):
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]

    # Ritorna solo una vista: una lista di modalità [immagini, testi]
    return [images, texts]

# Crea il DataModule per CC3M # --> similmente a come gli autori di CoMM definivano MMIMDBDataModule (in mmimdb.py)
# Configura i DataLoader per training/validation con batch size e multi-threading.
class CC3MDataModule(LightningDataModule):
    def __init__(self, dataroot, preprocess, batch_size=64, num_workers=16, val_split=0.1, test_split=0.1):
        super().__init__()
        self.dataroot = dataroot
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        
    def setup(self, stage=None):
        full_dataset = CC3MLLaVaDataset(
            dataroot=self.dataroot,
            preprocess=self.preprocess,
            return_image=True
        )
        total_size = len(full_dataset)
        test_size = int(self.test_split * total_size)
        val_size = int(self.val_split * (total_size - test_size))  # Val su ciò che resta
        train_size = total_size - test_size - val_size
        print(f"[INFO] total: {total_size}, train: {train_size}, val: {val_size}, test: {test_size}") # Messo perchè secondo me ci sono problemi nel dataset
        # print da usare CC3MLLaVaDataset(...) NON sta caricando 595k dati, ma circa la metà (~297k).

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
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
            collate_fn=collate_fn_noaugm # non dovrei voler estarre features contrastive la validazione quindi non serve
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn_noaugm # non ho bisogno di features contrastive quindi no collate_fn
        )
