import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    input_dim: int
    window_size: int
    hidden_dim: int
    latent_dim: int
    num_layers: int
    rnn_type: str
    dropout: float
    model_version: str = "1.0"
    description: Optional[str] = None
    num_layers_linear: int = 1

class RecurrentAutoencoder(nn.Module):
    def __init__(
        self,
        config=None,
        input_dim: int = 50,          # количество временных рядов
        window_size: int = 100,       # длина окна
        hidden_dim: int = 128,        # размер скрытого состояния RNN
        latent_dim: int = 32,         # размер скрытого представления
        num_layers: int = 2,          # количество слоев RNN
        rnn_type: str = "GRU",
        dropout: float = 0.1,
        num_layers_linear = 1,
    ):
        super().__init__()

        if config is not None:
            input_dim = config.input_dim
            window_size = config.window_size
            hidden_dim = config.hidden_dim
            latent_dim = config.latent_dim
            num_layers = config.num_layers
            rnn_type = config.rnn_type
            dropout = config.dropout
            num_layers_linear = config.num_layers_linear
            self.config = config

        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_layers_linear = num_layers_linear
        
        rnn_class = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.enc_norm = nn.LayerNorm(hidden_dim) 

        if window_size > 50 or num_layers_linear > 1:
            self.to_latent = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, latent_dim),
                nn.Tanh()
            )
            self.from_latent = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        else:
            self.from_latent = nn.Linear(latent_dim, hidden_dim)
            self.to_latent = nn.Linear(hidden_dim, latent_dim)
        
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
        last_layer_hidden = self.enc_norm(last_layer_hidden)
        
        latent = self.to_latent(last_layer_hidden)  # (batch, latent_dim)
        return latent
    
    def decode(self, latent, target_length):
        # latent: (batch, latent_dim)
        decoder_input = self.from_latent(latent)  # (batch, hidden_dim)
        
        # Тут повтор вектора для каждого временного шага
        decoder_input = decoder_input.unsqueeze(1).expand(-1, target_length, -1)  # (batch, window_size, hidden_dim)

        
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

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: Optional[str] = None):
        """Загрузка модели из файлов"""
        if config_path is None:
            config_path = model_path.replace('.pth', '_config.json')
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        
        model = cls(config)
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model

    def save_pretrained(self, save_path: str, config=None):
        """Сохранение модели и конфига"""
        if not config:
            config = self.config
        # Сохраняем веса
        torch.save(self.state_dict(), save_path)
        
        # Сохраняем конфиг
        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)
        
        # print(f"Model saved to {save_path}")
        # print(f"Config saved to {config_path}")