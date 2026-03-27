import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 50,          # количество временных рядов
        window_size: int = 100,       # длина окна
        hidden_dim: int = 128,        # размер скрытого состояния RNN
        latent_dim: int = 32,         # размер скрытого представления
        num_layers: int = 2,          # количество слоев RNN
        rnn_type: str = "GRU",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        
        rnn_class = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Sigmoid()
        )

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        # self.from_latent = nn.Linear(latent_dim, hidden_dim)
        # self.to_latent = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
        self.decoder_hidden = None
        
    def encode(self, x):
        _, hidden = self.encoder_rnn(x)
        
        if isinstance(hidden, tuple):  # LSTM
            h = hidden[0] 
        else:  # GRU
            h = hidden
        
        last_layer_hidden = h[-1]  # (batch, hidden_dim)
        
        latent = self.to_latent(last_layer_hidden)  # (batch, latent_dim)
        return latent
    
    def decode(self, latent, target_length):
        # latent: (batch, latent_dim)
        decoder_input = self.from_latent(latent)  # (batch, hidden_dim)
        
        # Тут повтор вектора для каждого временного шага
        decoder_input = decoder_input.unsqueeze(1)  # (batch, 1, hidden_dim)
        decoder_input = decoder_input.repeat(1, target_length, 1)  # (batch, window_size, hidden_dim)
        
        batch_size = latent.size(0)
        hidden_dim = self.decoder_rnn.hidden_size
        num_layers = self.decoder_rnn.num_layers
        
        if isinstance(self.decoder_rnn, nn.GRU):
            h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=latent.device)
            hidden = h0
        else:  # LSTM
            h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=latent.device)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim, device=latent.device)
            hidden = (h0, c0)
        
        decoder_output, _ = self.decoder_rnn(decoder_input, hidden)
        
        output = self.output_layer(decoder_output)  # (batch, window_size, input_dim)
        return output
    
    def forward(self, x):
        latent = self.encode(x)  # (batch, latent_dim)
        
        reconstructed = self.decode(latent, x.size(1))  # (batch, window_size, input_dim)
        
        return reconstructed, latent
