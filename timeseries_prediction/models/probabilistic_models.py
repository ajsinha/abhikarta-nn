"""
Probabilistic Time Series Models

Copyright © 2025-2030, All Rights Reserved
Ashutosh Sinha | Email: ajsinha@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Tuple, Optional
from .base import TimeSeriesModel


class DeepARModel(TimeSeriesModel):
    """
    DeepAR: Probabilistic forecasting with autoregressive RNN.
    
    Generates probabilistic forecasts by learning the distribution parameters
    at each time step using an autoregressive approach.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        distribution: str = 'gaussian',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize DeepAR model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension
            output_size: Number of outputs
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            distribution: Output distribution ('gaussian', 'negative_binomial')
            device: Device to run on
        """
        super(DeepARModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.distribution = distribution
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size + output_size,  # Input + previous prediction
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Distribution parameter networks
        if distribution == 'gaussian':
            # Mean and standard deviation
            self.mu_layer = nn.Linear(hidden_size, output_size)
            self.sigma_layer = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Softplus()  # Ensure positive sigma
            )
        elif distribution == 'negative_binomial':
            # Mean and dispersion
            self.mu_layer = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Softplus()
            )
            self.alpha_layer = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Softplus()
            )
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        y_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DeepAR.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            y_prev: Previous predictions (batch_size, seq_len, output_size)
            
        Returns:
            Tuple of (mean, variance/dispersion)
        """
        batch_size, seq_len, _ = x.size()
        
        # If no previous predictions, use zeros
        if y_prev is None:
            y_prev = torch.zeros(batch_size, seq_len, self.output_size).to(self.device)
        
        # Concatenate input with previous predictions
        lstm_input = torch.cat([x, y_prev], dim=-1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_input)
        
        # Last time step
        last_out = lstm_out[:, -1, :]
        
        # Generate distribution parameters
        if self.distribution == 'gaussian':
            mu = self.mu_layer(last_out)
            sigma = self.sigma_layer(last_out)
            return mu, sigma
        
        elif self.distribution == 'negative_binomial':
            mu = self.mu_layer(last_out)
            alpha = self.alpha_layer(last_out)
            return mu, alpha
    
    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        Generate samples from the predictive distribution.
        
        Args:
            x: Input tensor
            num_samples: Number of samples to generate
            
        Returns:
            Samples of shape (num_samples, batch_size, output_size)
        """
        self.eval()
        with torch.no_grad():
            param1, param2 = self.forward(x)
            
            if self.distribution == 'gaussian':
                distribution = dist.Normal(param1, param2)
            elif self.distribution == 'negative_binomial':
                distribution = dist.NegativeBinomial(param1, param2)
            
            samples = distribution.sample((num_samples,))
        
        return samples
    
    def predict(
        self,
        x: torch.Tensor,
        return_numpy: bool = True
    ) -> torch.Tensor:
        """Return mean prediction."""
        mu, _ = self.forward(x)
        if return_numpy:
            return mu.cpu().numpy()
        return mu


class VAETimeSeriesModel(TimeSeriesModel):
    """
    Variational Autoencoder for time series.
    
    Models time series in a latent space, useful for anomaly detection
    and probabilistic forecasting.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """Initialize VAE model."""
        super(VAETimeSeriesModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        self.latent_size = latent_size
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_size, hidden_size)
        
        self.decoder_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.to(self.device)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mu, logvar)
        """
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation.
        
        Args:
            z: Latent tensor
            seq_len: Sequence length to generate
            
        Returns:
            Decoded sequence
        """
        # Project latent to hidden
        h = self.fc_decode(z)
        
        # Repeat for sequence length
        h = h.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM decode
        out, _ = self.decoder_lstm(h)
        
        # Output projection
        reconstruction = self.fc_out(out)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        batch_size, seq_len, _ = x.size()
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decode(z, seq_len)
        
        return reconstruction, mu, logvar
    
    def vae_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        VAE loss combining reconstruction and KL divergence.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL divergence
            
        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss
    
    def predict(self, x: torch.Tensor, return_numpy: bool = True):
        """Generate prediction."""
        reconstruction, _, _ = self.forward(x)
        
        # Take last time step
        pred = reconstruction[:, -1, :]
        
        if return_numpy:
            return pred.cpu().detach().numpy()
        return pred


class QuantileRegressionModel(TimeSeriesModel):
    """
    Quantile Regression for probabilistic forecasting.
    
    Predicts multiple quantiles to estimate prediction intervals.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        quantiles: list = None,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize quantile regression model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension
            output_size: Output dimension
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
            num_layers: Number of layers
            dropout: Dropout rate
            device: Device to run on
        """
        super(QuantileRegressionModel, self).__init__(
            input_size, hidden_size, output_size, num_layers, dropout, device, **kwargs
        )
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Separate heads for each quantile
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size)
            )
            for _ in range(self.num_quantiles)
        ])
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning all quantile predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor of shape (batch_size, num_quantiles * output_size)
        """
        # LSTM encoding
        _, (h_n, _) = self.lstm(x)
        encoded = h_n[-1]
        
        # Predict each quantile
        quantile_preds = []
        for head in self.quantile_heads:
            quantile_preds.append(head(encoded))
        
        # Stack quantiles
        output = torch.cat(quantile_preds, dim=-1)
        
        return output
    
    def quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantile loss function.
        
        Args:
            predictions: Predicted quantiles
            targets: True values
            
        Returns:
            Quantile loss
        """
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.num_quantiles, -1)
        
        total_loss = 0
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, i, :]
            errors = targets - pred_q
            
            loss_q = torch.max(
                (q - 1) * errors,
                q * errors
            )
            total_loss += loss_q.mean()
        
        return total_loss / self.num_quantiles
    
    def predict(self, x: torch.Tensor, return_numpy: bool = True):
        """Return median prediction (0.5 quantile)."""
        output = self.forward(x)
        batch_size = output.size(0)
        output = output.view(batch_size, self.num_quantiles, -1)
        
        # Find median quantile
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else self.num_quantiles // 2
        median_pred = output[:, median_idx, :]
        
        if return_numpy:
            return median_pred.cpu().detach().numpy()
        return median_pred
