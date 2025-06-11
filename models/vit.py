import timm
import torch
import torch.nn as nn

# VisionTransformer
class VisionTransformer(nn.Module):
    """Pre-trained vision transformers supported by timm library. """
    """
    Wrapper per i Vision Transformer pre-addestrati supportati dalla libreria timm.

    Questo modulo permette di usare diversi Vision Transformer (ViT, DeiT, ecc.)
    con opzioni di pooling e di ritorno di token embedding intermedi.
    """

    def __init__(self, model_name: str,
                 pretrained: bool = True,
                 freeze: bool = False,
                 output_value: str = "embedding"):
        """
        Args:
            model_name (str): Nome del modello ViT supportato da timm (es. 'vit_base_patch16_224').
            pretrained (bool): Se usare pesi pre-addestrati (True) o no (False).
            freeze (bool): Se bloccare i pesi (utile per feature extraction).
            output_value (str): Se 'embedding', restituisce l'embedding globale;
                                se 'token_embeddings', restituisce i token embedding intermedi.
        """
        super().__init__()
        assert output_value in {'embedding', "token_embeddings"} # Verifica che l'output richiesto sia supportato

        # --- NB! è qui che rimuoviamo la head del modello per usare una identity ---
        if output_value == "token_embeddings":
            # Crea il modello senza pooling globale, così restituisce i token embedding
            self.model = timm.create_model(model_name, global_pool="", pretrained=pretrained)
            # Rimuove la testa finale (classificatore) e la sostituisce con un'identità
            self.model.head = nn.Identity() # get token embeddings  # questo permette di ottenere i token embedding (con dim 768)
        else:
            # Crea il modello standard (con pooling)
            self.model = timm.create_model(model_name, pretrained=pretrained)
        # Salva i parametri di configurazione
        self.model_name = model_name
        self.freeze = freeze
        self.pretrained = pretrained

        if freeze: # no grad computed
            # Disattiva il calcolo del gradiente per tutti i parametri (utile per feature extraction)
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Esegue l'inferenza sul modello.

        Args:
            x (torch.Tensor): Batch di immagini in input.

        Returns:
            torch.Tensor: Embedding globale (o token embedding intermedi) a seconda della configurazione.
        """
        if self.freeze:
            # Se i pesi sono congelati, disabilita la computazione dei gradienti
            with torch.no_grad():
                return self.model(x)
        # Altrimenti, calcola normalmente
        return self.model(x)