import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

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


class VariationalRecurrentAutoencoder(nn.Module):
    """
    Variational Recurrent Autoencoder (VRAE).

    Отличие от RAE:
      - Энкодер предсказывает не точку в латентном пространстве,
        а параметры распределения: mu и log_var.
      - Сэмплирование z ~ N(mu, exp(log_var)) через reparameterization trick.
      - В функцию потерь добавляется KL-дивергенция.
    """

    def __init__(
        self,
        config = None,
        input_dim: int = 53,
        window_size: int = 100,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        rnn_type: str = "GRU",
        dropout: float = 0.1,
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

        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        rnn_class = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.encoder_rnn = rnn_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.enc_norm = nn.LayerNorm(hidden_dim)

        if window_size > 50:
            self.pre_latent = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fc_mu      = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim)

            self.from_latent = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
        else:
            self.pre_latent  = nn.Identity()
            self.fc_mu       = nn.Linear(hidden_dim, latent_dim)
            self.fc_log_var  = nn.Linear(hidden_dim, latent_dim)
            self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder_rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        z = mu + eps * std,  eps ~ N(0, I)

        Во время инференса (eval-режим) можно использовать только mu,
        передав use_mean=True в encode().
        """
        if self.training:
            std = torch.exp(0.5 * log_var)   # std = exp(log_var / 2)
            eps = torch.randn_like(std)       # eps ~ N(0, I)
            return mu + eps * std
        else:
            return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Возвращает (z, mu, log_var).
        z   — сэмплированный латентный вектор
        mu  — среднее распределения
        log_var — логарифм дисперсии
        """
        _, hidden = self.encoder_rnn(x)

        if isinstance(hidden, tuple):   # LSTM: hidden = (h, c)
            h = hidden[0]
        else:                           # GRU
            h = hidden

        last_layer_hidden = h[-1]                   # (batch, hidden_dim)
        last_layer_hidden = self.enc_norm(last_layer_hidden)

        h_pre = self.pre_latent(last_layer_hidden)  # (batch, hidden_dim // 2) или (batch, hidden_dim)

        mu      = self.fc_mu(h_pre)       # (batch, latent_dim)
        log_var = self.fc_log_var(h_pre)  # (batch, latent_dim)

        # Ограничиваем log_var для численной стабильности
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        z = self.reparameterize(mu, log_var)  # (batch, latent_dim)
        return z, mu, log_var

    def decode(self, z: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        z: (batch, latent_dim)
        target_length: длина восстанавливаемой последовательности
        """
        decoder_input = self.from_latent(z)                              # (batch, hidden_dim)
        decoder_input = decoder_input.unsqueeze(1).expand(-1, target_length, -1)  # (batch, T, hidden_dim)

        batch_size = z.size(0)
        hidden_dim = self.decoder_rnn.hidden_size
        num_layers = self.decoder_rnn.num_layers

        if isinstance(self.decoder_rnn, nn.GRU):
            hidden = torch.zeros(num_layers, batch_size, hidden_dim, device=z.device)
        else:
            h0 = torch.zeros(num_layers, batch_size, hidden_dim, device=z.device)
            c0 = torch.zeros(num_layers, batch_size, hidden_dim, device=z.device)
            hidden = (h0, c0)

        decoder_output, _ = self.decoder_rnn(decoder_input, hidden)
        output = self.output_layer(decoder_output)   # (batch, T, input_dim)
        return output

    def forward(self, x: torch.Tensor):
        """
        Возвращает (reconstructed, z, mu, log_var).
        """
        z, mu, log_var = self.encode(x)
        reconstructed = self.decode(z, x.size(1))
        return reconstructed, z, mu, log_var

    @classmethod
    def from_pretrained(cls, model_path: str, config_path: Optional[str] = None):
        if config_path is None:
            config_path = model_path.replace('.pth', '_config.json')

        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)

        model = cls(
            input_dim=config.input_dim,
            window_size=config.window_size,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            rnn_type=config.rnn_type,
            dropout=config.dropout,
        )
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_path: str, config: ModelConfig):
        torch.save(self.state_dict(), save_path)

        config_path = save_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=4)

        print(f"Model saved  → {save_path}")
        print(f"Config saved → {config_path}")


def vrae_loss(
    x: torch.Tensor,
    reconstructed: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
    free_bits=2.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = Reconstruction loss + beta * KL divergence

    ----------
    total_loss, recon_loss, kl_loss
    """
    # ── Reconstruction loss (MSE) ────────────────────────────────────────────
    recon_loss = F.mse_loss(reconstructed, x, reduction=reduction)

    # ── KL divergence: KL( N(mu, sigma²) || N(0, I) ) ───────────────────────
    # Аналитическая формула:
    #   KL = -0.5 * sum( 1 + log_var - mu² - exp(log_var) )
    kl_per_sample = -0.5 * torch.sum(
        1.0 + log_var - mu.pow(2) - log_var.exp(),
        dim=-1,                           # суммируем по latent_dim
    )                                     # (batch,)

    kl_with_min = torch.max(kl_per_sample, torch.full_like(kl_per_sample, free_bits))
    if reduction == "mean":
        kl_loss = kl_with_min.mean()
    else:
        kl_loss = kl_with_min.sum()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def train_one_epoch(
    model: VariationalRecurrentAutoencoder,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
    beta_warmup: bool = False,
    current_step: int = 0,
    warmup_steps: int = 10_000,
) -> dict:
    model.train()
    total, recon_total, kl_total = 0.0, 0.0, 0.0

    for batch in dataloader:
        x = batch.to(device)                          # (batch, T, input_dim)

        # KL Annealing / Warm-up
        if beta_warmup:
            effective_beta = beta * min(1.0, current_step / warmup_steps)
            current_step += 1
        else:
            effective_beta = beta

        optimizer.zero_grad()

        reconstructed, z, mu, log_var = model(x)

        loss, recon_loss, kl_loss = vrae_loss(
            x, reconstructed, mu, log_var,
            beta=effective_beta,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total      += loss.item()
        recon_total += recon_loss.item()
        kl_total   += kl_loss.item()

    n = len(dataloader)
    return {
        "loss":       total       / n,
        "recon_loss": recon_total / n,
        "kl_loss":    kl_total    / n,
        "beta":       effective_beta,
    }



def train_vrae(
    model: VariationalRecurrentAutoencoder,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    beta: float = 0.1,
    beta_warmup: bool = False,
    warmup_steps: int = 10_000,
    device: torch.device = torch.device('cpu'),
    saved_path: Optional[str] = None,
    patience: int = 10,
    gradient_clip: float = 1.0,
) -> Tuple[List[dict], List[dict]]:
    """
    Обучение VRAE модели на несколько эпох
    
    Args:
        model: VRAE модель
        train_loader: DataLoader для обучения
        test_loader: DataLoader для валидации
        epochs: количество эпох
        lr: learning rate
        beta: коэффициент KL divergence
        beta_warmup: использовать ли разогрев beta
        warmup_steps: количество шагов для разогрева beta
        device: устройство для обучения
        saved_path: путь для сохранения лучшей модели
        patience: patience для early stopping
        gradient_clip: значение для clip_grad_norm
    
    Returns:
        train_metrics_list: список словарей с метриками на каждой эпохе обучения
        test_metrics_list: список словарей с метриками на каждой эпохе валидации
    """
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_loss = float("inf")
    no_improve = 0
    best_state = None
    
    train_metrics_list = []
    test_metrics_list = []
    current_step = 0
    
    print(f"Начинаем обучение VRAE на {epochs} эпох")
    print(f"Device: {device}, Learning rate: {lr}, Beta: {beta}, Beta warmup: {beta_warmup}")
    print("-" * 80)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=beta,
            beta_warmup=beta_warmup,
            current_step=current_step,
            warmup_steps=warmup_steps
        )
        
        current_step += len(train_loader)
        
        model.eval()
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch.to(device)  # (batch, T, input_dim)
                reconstructed, z, mu, log_var = model(x)
                
                loss, recon_loss, kl_loss = vrae_loss(
                    x, reconstructed, mu, log_var,
                    beta=train_metrics['beta']  # используем тот же beta что и в обучении
                )
                
                val_total_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        n_val = len(test_loader)
        test_metrics = {
            "loss": val_total_loss / n_val,
            "recon_loss": val_recon_loss / n_val,
            "kl_loss": val_kl_loss / n_val,
            "beta": train_metrics['beta']
        }
        
        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} [{epoch_time:.1f}s]")
        print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
              f"Recon: {train_metrics['recon_loss']:.6f}, "
              f"KL: {train_metrics['kl_loss']:.6f}, "
              f"Beta: {train_metrics['beta']:.4f}")
        print(f"  Test  - Loss: {test_metrics['loss']:.6f}, "
              f"Recon: {test_metrics['recon_loss']:.6f}, "
              f"KL: {test_metrics['kl_loss']:.6f}")
        
        # Early stopping check
        if test_metrics['loss'] < best_loss - 1e-6:
            best_loss = test_metrics['loss']
            no_improve = 0
            
            # Сохраняем лучшую модель
            if saved_path is not None:
                # Сохраняем состояние модели
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'best_loss': best_loss,
                }, saved_path)
                print(f"  Модель сохранена в {saved_path} (loss: {best_loss:.6f})")
        else:
            no_improve += 1
            print(f"  Нет улучшения {no_improve}/{patience} ")
        
        print(f"  Ratio test/train: {test_metrics['loss'] / train_metrics['loss']:.3f}")
        print("-" * 80)
        
        # Ранняя остановка
        if no_improve >= patience:
            print(f"\n Early stopping на эпохе {epoch+1}/{epochs}")
            print(f"Лучшее значение loss: {best_loss:.6f}")
            break
    
    # Финальная статистика
    print("\n" + "="*80)
    print("Обучение завершено!")
    print(f"Лучшая валидационная loss: {best_loss:.6f}")
    print(f"Финальная train loss: {train_metrics_list[-1]['loss']:.6f}")
    print(f"Финальная test loss: {test_metrics_list[-1]['loss']:.6f}")
    print("="*80)
    
    return train_metrics_list, test_metrics_list


def plot_training_history(train_metrics_list, test_metrics_list):
    """Визуализация истории обучения VRAE"""
    epochs = range(1, len(train_metrics_list) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(epochs, [m['loss'] for m in train_metrics_list], 'b-', label='Train')
    axes[0, 0].plot(epochs, [m['loss'] for m in test_metrics_list], 'r-', label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss over epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, [m['recon_loss'] for m in train_metrics_list], 'b-', label='Train')
    axes[0, 1].plot(epochs, [m['recon_loss'] for m in test_metrics_list], 'r-', label='Test')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss over epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL divergence
    axes[1, 0].plot(epochs, [m['kl_loss'] for m in train_metrics_list], 'b-', label='Train')
    axes[1, 0].plot(epochs, [m['kl_loss'] for m in test_metrics_list], 'r-', label='Test')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence over epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Beta value (if warmup was used)
    betas = [m['beta'] for m in train_metrics_list]
    axes[1, 1].plot(epochs, betas, 'g-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta')
    axes[1, 1].set_title('Beta Annealing')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
