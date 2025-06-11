import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from einops import repeat
from collections import OrderedDict
# Local import
from models.mlp import MLP

# Implementa una versione approssimata e più veloce di GELU
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# Implementa un blocco di cross-attention residua (da un input a un altro) + feedforward
class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention module between 2 inputs. """
    def __init__(self, d_model: int, n_heads: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()

        # Modulo di cross-attention: x attende y
        self.attn = nn.MultiheadAttention(d_model, n_heads, add_bias_kv=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1x = nn.LayerNorm(d_model) # Normalizzazione input x
        self.ln_1y = nn.LayerNorm(d_model) # Normalizzazione input y

        # Feedforward network dopo cross-attention
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)), # Espansione dimensionale
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)) # Proiezione back to d_model
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                  attn_mask: torch.Tensor = None):
         # x: query; y: key e value
        return self.attn(x, y, y, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        # Residual connection: x + attn(x, y, y)
        x = x + self.attention(self.ln_1x(x), self.ln_1y(y), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # Residual connection: x + MLP(x)
        x = x + self.mlp(self.ln_2(x))
        return x

# Implementa un blocco di self-attention residua (classico Transformer) + feedforward
class ResidualAttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, d_model: int, n_head: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, add_bias_kv=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # Self-attention: x attende x
        return self.attn(x.clone(), x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

# FusionTransformer: Implementa la logica di fusione vera e propria
# I parametri al suo interno verranno addestrati # TRAINABLES
# Modulo di fusione multimodale basato su self-attention o cross-attention
class FusionTransformer(nn.Module):
    """
    Fusion of features from multiple modalities using attention.
    in_shape: (N, L1, E), (N, L2, E), out_shape: (N, E)
    We use either (modalità):
        - "concat": concatenation over tokens + self-attention module [concatena i token delle modalità e applica ResidualAttentionBlock [self-attention]]
        - "x-attn": cross-attention between two sets of tokens + concatenation over tokens [cross-attention (ResidualCrossAttentionBlock) tra due insiemi di token e concatena i risultati [cross-attention]]
    An attention mask can be applied eventually for each modality with shape (N, Li) for modality i.
    """
    def __init__(self, width: int,
                 n_heads: int,
                 n_layers: int,
                 fusion: str = "concat",
                 pool: str = "cls",
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = True):
        """
        :param width: embedding size
        :param n_heads: number of heads in multi-head attention blocks
        :param n_layers: number of attention blocks
        :param fusion: "concat" or "x-attn"
        :param pool: "cls" or "pool"
        :param add_bias_kv: If specified, adds bias to the key and value sequences at dim=0.
        :param dropout: Dropout probability on `attn_output_weights`
        :param batch_first: input tensor is either (batch, tokens, features) if `True` or (tokens, batch, features)
        """
        super().__init__()

        self.fusion = fusion
        self.width = width
        self.layers = n_layers
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 if batch_first else 0
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, width)) if self.pool == "cls" else None

        # Costruzione dei blocchi di attention in base al tipo di fusione
        if fusion == "concat":
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                       dropout=dropout, batch_first=batch_first)
                for _ in range(n_layers)])
        elif fusion == "x-attn":
            self.resblocks = [
                nn.Sequential(*[
                    ResidualCrossAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                                dropout=dropout, batch_first=batch_first)
                    for _ in range(n_layers)])
                for _ in range(2)] # due flussi di cross-attention simmetrici
        else:
            raise ValueError("Unknown fusion %s" % fusion)
        self.initialize()

    def initialize(self):
        """
        Inizializza i pesi dei moduli (multihead attention, MLP) con distribuzioni normali
        scalate in base alla dimensione del modello.
        """
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: List[torch.Tensor], key_padding_mask: List[torch.Tensor] = None):
        """
        Esegue la fusione dei token:
        - Concatena i token e applica self-attention ("concat")
        - Applica cross-attention tra due set di token e concatena ("x-attn")
        :param x: input tensors
        :param key_padding_mask: torch mask of type bool. `True` indicates unattended tokens.
        :return:
        """
        # Concatenate over tokens + self-attention
        if self.fusion == "concat":
            x = torch.cat(x, dim=self.token_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
            if self.pool == "cls": # append cls token at the beginning
                cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_token, x), dim=self.token_dim)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=self.token_dim)

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf")).float()

            for layer in self.resblocks:
                x = layer(x, key_padding_mask=key_padding_mask)

            x = self.norm(x)

            if self.pool == "cls":
                x = x[:, 0] if self.token_dim == 1 else x[0]
            else:
                x = x.mean(dim=self.token_dim)
            return x
        # Cross-attention + concatenate over tokens
        elif self.fusion == "x-attn":
            if self.pool == "cls":
                raise ValueError("Only `mean` pool is implemented for cross-attention.")
            if len(x) != 2:
                raise ValueError("Only 2 modalities are currently accepted for cross-attention")
            if key_padding_mask is not None:
                raise NotImplementedError()
            x1, x2 = x
            x = torch.cat([self.resblocks[0](x1, x2, key_padding_mask),
                           self.resblocks[1](x2, x1, key_padding_mask)], dim=self.token_dim)
            x = self.norm(x).mean(dim=self.token_dim)
            return x

# MMFusion: orchestratore finale di tutte le modalità. Gestisce l’intero processo di fusione multimodale
class MMFusion(nn.Module):
    """
    Riceve una lista di encoders e adapters per ogni modalità:
    1. Codifica ciascun input tramite l'encoder
    2. Tokenizza (opzionale) tramite l'adapter
    3. Applica FusionTransformer per la fusione multimodale
    Permette anche la modalità di mascheramento dei moduli (mask_modalities).
    """
    def __init__(self,
                 encoders: List[nn.Module],
                 input_adapters: List[nn.Module],
                 embed_dim: int = 512,
                 fusion: str = "concat",
                 pool: str = "cls",
                 n_heads: int = 8,
                 n_layers: int = 1,
                 add_bias_kv: bool = False,
                 dropout: float = 0.):
        """ Multi-Modal (MM) fusion model using `FusionTransformer` in the latent space.
        It can handle an arbitrary number of input modalities.
        Each modality is encoded through either a:
            - Transformer (e.g. for text or audio) -> no adapters
            - CNN (e.g. for images) -> `PatchedInputAdapter` for tokenization
            - MLP (e.g. tabular data) -> `FeaturesInputAdapter` for tokenization
        Once each modality is encoded and tokenized, it then goes to `FusionTransformer` to output
        the final embedding.

        :param encoders: List of Torch encoders (CNN, Transformer, MLP, etc.) for each modality. Lista di encoder PyTorch (uno per modalità)
        :param input_adapters: List of Torch adapters for each modality (can be None if not required). Lista di adapter PyTorch per ciascuna modalità (può essere None)
        :param embed_dim: Embedding size. Dimensione dell'embedding latente
        :param fusion: "concat" or "x-attn". For "x-attn", only "mean" pool is accepted. Strategia di fusione ('concat' o 'x-attn')
        :param pool: "cls" or "mean", pooling strategy for the tokens. Strategia di pooling ('cls' o 'mean')
        :param n_heads: Number of heads in multi-heads attention blocks. Numero di teste di attenzione nella FusionTransformer
        :param n_layers: Number of attention layers in latent fusion. Numero di strati di attenzione
        :param add_bias_kv: If `True`, add bias term in key/values mapping. Se True, aggiunge bias a key/value
        :param dropout: attention matrix dropout rate. Dropout per la matrice di attenzione
        """
        super().__init__()

        # Controllo di coerenza: ogni encoder deve avere il corrispettivo adapter
        assert len(encoders) == len(input_adapters), "Each encoder must have an adapter."

        # Controllo di validità del pooling
        assert pool in {'cls', 'mean'}, "pool type must be either cls (cls token) or mean (mean pooling)"

        # Lista di adapter per ciascuna modalità (nn.ModuleList è comoda per register_module automatico)
        self.input_adapters = nn.ModuleList(input_adapters) # Non dovrebbero servire # Tokenizza (solo se serve)
        # Lista di encoder (pre-addestrati nel nostro caso, frozen oppure non)
        self.encoders = nn.ModuleList(encoders) # Non Trainables # Encoder pre-addestrati nel nostro caso
        self.pool = pool
        self.num_modalities = len(self.encoders)

        # FusionTransformer: modulo di fusione multimodale nel latent space
        # Riceve tutti i token e li fonde secondo la strategia di fusione scelta
        self.fusion_transformer = FusionTransformer(embed_dim, n_heads, n_layers,
                                                    fusion, pool, add_bias_kv, dropout,
                                                    batch_first=True) # Tranables

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):
        """
        :param x: List of tensors (uno per modalità)
        :param mask_modalities: Mask indicating which modalities are given. (maschera per indicare quali modalità sono presenti)
            By default, `x` should have all modalities.
            If a list of lists is given, assume `x` has all modalities and computes
            a list of output by masking out modalitites according to `mask_modalities`.
        :return: a latent vector z or list of vector if `mask_modalities` is a list of list. (embedding finale multimodale)
        """
        list_mask_mod = None
        # Se la maschera non è fornita, considera tutte le modalità come presenti
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        # Caso speciale: maschera a livello di sample (es. lista di liste)
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]

        # Check consistenza maschera
        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")

        # Conta il numero di modalità effettivamente presenti
        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

         # Filtra encoder e adapter in base alla maschera
        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        input_adapters = [adapter for (adapter, m) in zip(self.input_adapters, mask_modalities) if m]
        attn_mask = []

        # 1. Encode input modalities (Encoding di ciascuna modalità)
        z = []
        for (enc, xi) in zip(encoders, x):
            embedding = enc(xi)
            attn_mask_ = None
            # Se l'encoder restituisce un dict (es. token_embeddings + attention_mask)
            if isinstance(embedding, dict):  # attention mask must be considered # se l'encoder restituisce attenzione
                attn_mask_ = embedding["attention_mask"]
                embedding = embedding["token_embeddings"]
            z.append(embedding)
            attn_mask.append(attn_mask_)

        # 2. Tokenize each latent features (Tokenizzazione con gli adapter (solo se definiti))
        latent_tokens = [adapter(zi) if adapter is not None else zi
                         for (adapter, zi) in zip(input_adapters, z)]
        # Costruzione dell'attention mask
        # Se non definita dal modello, la creiamo come maschera di zeros_like
        attn_mask = [attn_mask_ if attn_mask_ is not None else torch.zeros_like(zi[:,:,0]).bool()
                     for (attn_mask_, zi) in zip(attn_mask, latent_tokens)]
        
        # Fusione multimodale
        if list_mask_mod is None:
            # 3. FusionTransformer forward pass
            # Caso standard: una sola chiamata al FusionTransformer
            z = self.fusion_transformer(latent_tokens, key_padding_mask=attn_mask)
        else:
            # 3.bis Drop modalities according to `mask_modalities`
            # Caso: lista di maschere a livello di sample
            z = []
            for mask_mod in list_mask_mod:
                # Filtra le modalità per questo sample
                latent_tokens_ = [z for (z, m) in zip(latent_tokens, mask_mod) if m]
                attn_mask_ = [attn for (attn, m) in zip(attn_mask, mask_mod) if m]
                # 3. FusionTransformer forward pass
                z.append(self.fusion_transformer(latent_tokens_))
        return z

    def encode_single_mod(self, x: torch.Tensor, mod: int):
        """
        Codifica una singola modalità (utile per debug o feature extraction)
        :param x: tensore di input per la modalità richiesta
        :param mod: indice della modalità
        :return: embedding prodotto dall'encoder
        """
        assert 0 <= mod < self.num_modalities, "Wrong input modality"
        return self.encoders[mod](x)

# Fusione lineare semplice
class LinearFusion(nn.Module):
    """
    Implementa una fusione multimodale lineare.
    Ogni modalità è encodata separatamente e proiettata in uno spazio comune (embed_dim) tramite un layer Linear.
    Se ci sono due modalità, le concateno e le progetto tramite head_projector.
    """
    def __init__(self,
                 encoders: List[nn.Module],
                 mod_dims: List[int],
                 embed_dim: int = 512,
                 **kwargs):
        """
        :param encoders: lista di moduli PyTorch, uno per ciascuna modalità
        :param mod_dims: lista di dimensioni di output degli encoders (per ogni modalità)
        :param embed_dim: dimensione comune di embedding nello spazio latente
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.mod_dims = mod_dims
        assert len(self.mod_dims) == len(self.encoders)
        self.embed_dim = embed_dim
        self.num_modalities = len(self.encoders)
        # projector for each modality to common space # Proiettori lineari per portare ciascuna modalità nello spazio comune
        self.projectors = nn.ModuleList([nn.Linear(mod_dim, embed_dim) for mod_dim in mod_dims])
        # projector for all modalities to common space  # Proiettore combinato per fondere tutte le modalità concatenate
        self.head_projector = nn.Linear(int(sum(mod_dims)), embed_dim)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):
        """
        :param x: lista di tensori (uno per modalità)
        :param mask_modalities: maschera per abilitare/disabilitare modalità (anche lista di liste)
        :return: embedding comune
        """
        list_mask_mod = None
        # Prepara la maschera (default: tutte le modalità presenti)
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]
        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")
        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

        # Filtra gli encoder secondo la maschera
        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        Z = [enc(xi) for enc, xi in zip(encoders, x)] # Applica l'encoder a ciascun input
        if list_mask_mod is not None:
            # Modalità diverse per ogni sample
            Z_ = []
            for mask_mod in list_mask_mod:
                Z_.append(self.get_common_embedding(Z, mask_mod))
            return Z_
        return self.get_common_embedding(Z, mask_modalities)

    def get_common_embedding(self, z: List[torch.Tensor], mask_modalities: List[bool]):
        """
        Combina i tensori encodati proiettandoli nello spazio comune:
        - Se 1 modalità: proiettore individuale
        - Se 2 modalità: concatenazione + proiettore combinato
        """
        if np.sum(mask_modalities) == 1:
            idx = int(np.nonzero(mask_modalities)[0][0])
            return self.projectors[idx](z[idx])
        elif np.sum(mask_modalities) == 2:
            return self.head_projector(torch.cat(z, dim=-1))
        raise NotImplementedError()


class MLPFusion(nn.Module):
    """
    Fusione multimodale non lineare:
    Ogni modalità viene encodata e proiettata tramite un MLP invece di un Linear.
    """
    def __init__(self,
                 encoders: List[nn.Module],
                 mod_dims: List[int],
                 embed_dim: int = 512,
                 **kwargs):
        """
        :param encoders: lista di moduli PyTorch (uno per modalità)
        :param mod_dims: lista di dimensioni di output degli encoders
        :param embed_dim: dimensione embedding comune
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.mod_dims = mod_dims
        assert len(self.mod_dims) == len(self.encoders)
        self.embed_dim = embed_dim
        self.num_modalities = len(self.encoders)
        # non-linear projector for each modality to common space # Proiettori non lineari per ciascuna modalità
        self.projectors = nn.ModuleList([MLP(mod_dim, embed_dim, embed_dim) for mod_dim in mod_dims])
        # non-linear projector for all modalities to common space # Proiettore non lineare combinato per tutte le modalità concatenate
        self.head_projector = MLP(int(sum(mod_dims)), embed_dim, embed_dim)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):
        """
        :param x: lista di tensori (uno per modalità)
        :param mask_modalities: maschera per abilitare/disabilitare modalità (anche lista di liste)
        :return: embedding comune
        """
        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]
        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")
        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        Z = [enc(xi) for enc, xi in zip(encoders, x)]
        if list_mask_mod is not None:
            Z_ = []
            for mask_mod in list_mask_mod:
                Z_.append(self.get_common_embedding(Z, mask_mod))
            return Z_
        return self.get_common_embedding(Z, mask_modalities)

    def get_common_embedding(self, z: List[torch.Tensor], mask_modalities: List[bool]):
        """
        Stesso principio di LinearFusion, ma usando proiettori non lineari:
        - Se 1 modalità: proiettore individuale
        - Se 2 modalità: concatenazione + head_projector
        """
        if np.sum(mask_modalities) == 1:
            idx = int(np.nonzero(mask_modalities)[0][0])
            return self.projectors[idx](z[idx])
        elif np.sum(mask_modalities) == 2:
            return self.head_projector(torch.cat(z, dim=-1))
        raise NotImplementedError()


if __name__ == "__main__":
    # Esempio di test rapido di FusionTransformer
    width = 10
    batch = 3
    fusion = FusionTransformer(width, 2, 2)
    x = [torch.randn((batch, 2, width)), torch.randn((batch, 3, width))]
    # preserve modality 1
    mask = [torch.ones((batch, 2)).bool(), torch.ones((batch, 3)).bool()]
    print(fusion(x, mask))
    print(fusion([x[1]]))



