import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.

        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)              # Primo layer lineare: input -> hidden
        self.fc2 = nn.Linear(hiddim, outdim)            # Secondo layer lineare: hidden -> output        
        self.dropout_layer = torch.nn.Dropout(dropoutp) # Livello di dropout (disattivato se non specificato)
        self.dropout = dropout        
        self.output_each_layer = output_each_layer      # Se True, restituisce anche gli output intermedi (utile per debugging o analisi)
        self.lklu = nn.LeakyReLU(0.2)                   # Attivazione non lineare LeakyReLU (usata solo se output_each_layer=True)

    def forward(self, x):
        """Apply MLP to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # Applica il primo layer con attivazione ReLU
        output = F.relu(self.fc(x))

        # Applica il dropout se abilitato
        if self.dropout:
            output = self.dropout_layer(output)

        # Applica il secondo layer (lineare) per ottenere l'output finale
        output2 = self.fc2(output)

        # Applica il dropout anche dopo il secondo layer (usando ancora output)
        if self.dropout:
            output2 = self.dropout_layer(output)

        # Se richiesto, restituisce anche l'input e i risultati intermedi
        if self.output_each_layer:
            # output intermedi: input originale, output primo layer, output secondo layer
            return [0, x, output, self.lklu(output2)]
        return output2 # Se no, restituisce solo l'output finale
