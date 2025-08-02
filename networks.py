import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import math

class LayerNorm(nn.Module):
    """Layer normalization for improved training stability."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class SpectralNorm(nn.Module):
    """Spectral normalization for improved training stability."""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)  # Residual connection with layer norm
        return x

class AttentionLayer(nn.Module):
    """Self-attention layer for state representation learning."""
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Create queries, keys, values
        q = self.query(x).unsqueeze(1)  # (batch_size, 1, attention_dim)
        k = self.key(x).unsqueeze(2)    # (batch_size, attention_dim, 1)
        v = self.value(x).unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # Compute attention scores
        attention_scores = torch.bmm(q, k) / math.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.bmm(attention_weights, v).squeeze(1)
        output = self.output(attended)
        
        return output + x  # Residual connection

class ImprovedActor(nn.Module):
    """
    Enhanced Actor network with advanced architectural features.
    Incorporates residual connections, attention, and layer normalization.
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 hidden_dims: List[int] = [400, 300], use_attention: bool = False,
                 use_residual: bool = False, use_layer_norm: bool = True,
                 dropout: float = 0.1, use_spectral_norm: bool = False):
        super(ImprovedActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input processing
        if use_attention:
            self.attention = AttentionLayer(state_dim)
        
        # Build network layers
        dims = [state_dim] + hidden_dims + [action_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            
            # Apply spectral normalization if requested
            if use_spectral_norm and i < len(dims) - 2:  # Not on output layer
                layer = SpectralNorm(layer)
            
            self.layers.append(layer)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                LayerNorm(dims[i + 1]) for i in range(len(dims) - 2)
            ])
        else:
            self.layer_norms = None
        
        # Residual blocks
        if use_residual:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(dim, dropout) for dim in hidden_dims
            ])
        else:
            self.residual_blocks = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using appropriate schemes."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):  # Handle spectral norm case
                if i < len(self.layers) - 1:  # Hidden layers
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                else:  # Output layer
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.uniform_(layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        x = state
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Process through layers
        for i, layer in enumerate(self.layers[:-1]):  # All but last layer
            x = layer(x)
            
            # Apply layer normalization
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            
            # Apply activation
            x = F.relu(x)
            
            # Apply residual block if enabled
            if self.use_residual and i < len(self.residual_blocks):
                x = self.residual_blocks[i](x)
            
            # Apply dropout
            if self.dropout is not None:
                x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        action = torch.tanh(x) * self.max_action
        
        return action

class ImprovedCritic(nn.Module):
    """
    Enhanced Critic network with advanced architectural features.
    Uses separate processing paths for state and action before fusion.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [400, 300], use_attention: bool = False,
                 use_residual: bool = False, use_layer_norm: bool = True,
                 dropout: float = 0.1, use_spectral_norm: bool = False,
                 fusion_layer: int = 1):
        super(ImprovedCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fusion_layer = fusion_layer  # Which layer to fuse state and action
        
        # State processing path
        if use_attention:
            self.state_attention = AttentionLayer(state_dim)
        else:
            self.state_attention = None
        
        # Build state processing layers (before fusion)
        state_dims = [state_dim] + hidden_dims[:fusion_layer]
        self.state_layers = nn.ModuleList()
        for i in range(len(state_dims) - 1):
            layer = nn.Linear(state_dims[i], state_dims[i + 1])
            if use_spectral_norm:
                layer = SpectralNorm(layer)
            self.state_layers.append(layer)
        
        # Action processing (simple linear layer)
        self.action_layer = nn.Linear(action_dim, hidden_dims[fusion_layer - 1] if fusion_layer > 0 else hidden_dims[0])
        
        # Fused processing layers (after fusion)
        fused_input_dim = hidden_dims[fusion_layer - 1] if fusion_layer > 0 else state_dim
        if fusion_layer > 0:
            fused_input_dim += hidden_dims[fusion_layer - 1]  # Add action processing output
        else:
            fused_input_dim = state_dim + action_dim
        
        fused_dims = [fused_input_dim] + hidden_dims[fusion_layer:] + [1]
        self.fused_layers = nn.ModuleList()
        for i in range(len(fused_dims) - 1):
            layer = nn.Linear(fused_dims[i], fused_dims[i + 1])
            if use_spectral_norm and i < len(fused_dims) - 2:  # Not on output layer
                layer = SpectralNorm(layer)
            self.fused_layers.append(layer)
        
        # Twin critic (Q2 network)
        self.twin_state_layers = nn.ModuleList()
        for i in range(len(state_dims) - 1):
            layer = nn.Linear(state_dims[i], state_dims[i + 1])
            if use_spectral_norm:
                layer = SpectralNorm(layer)
            self.twin_state_layers.append(layer)
        
        self.twin_action_layer = nn.Linear(action_dim, hidden_dims[fusion_layer - 1] if fusion_layer > 0 else hidden_dims[0])
        
        self.twin_fused_layers = nn.ModuleList()
        for i in range(len(fused_dims) - 1):
            layer = nn.Linear(fused_dims[i], fused_dims[i + 1])
            if use_spectral_norm and i < len(fused_dims) - 2:
                layer = SpectralNorm(layer)
            self.twin_fused_layers.append(layer)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms_q1 = nn.ModuleList([
                LayerNorm(dim) for dim in hidden_dims[:-1]
            ])
            self.layer_norms_q2 = nn.ModuleList([
                LayerNorm(dim) for dim in hidden_dims[:-1]
            ])
        else:
            self.layer_norms_q1 = None
            self.layer_norms_q2 = None
        
        # Residual blocks
        if use_residual:
            self.residual_blocks_q1 = nn.ModuleList([
                ResidualBlock(dim, dropout) for dim in hidden_dims[:-1]
            ])
            self.residual_blocks_q2 = nn.ModuleList([
                ResidualBlock(dim, dropout) for dim in hidden_dims[:-1]
            ])
        else:
            self.residual_blocks_q1 = None
            self.residual_blocks_q2 = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        all_layers = (list(self.state_layers) + [self.action_layer] + list(self.fused_layers) +
                     list(self.twin_state_layers) + [self.twin_action_layer] + list(self.twin_fused_layers))
        
        for layer in all_layers:
            if hasattr(layer, 'weight'):
                if layer in [self.fused_layers[-1], self.twin_fused_layers[-1]]:  # Output layers
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.uniform_(layer.bias, -3e-3, 3e-3)
                else:  # Hidden layers
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def _forward_single_critic(self, state: torch.Tensor, action: torch.Tensor,
                              state_layers, action_layer, fused_layers,
                              layer_norms, residual_blocks) -> torch.Tensor:
        """Forward pass through a single critic network."""
        
        # Process state
        x_state = state
        if self.state_attention is not None:
            x_state = self.state_attention(x_state)
        
        for i, layer in enumerate(state_layers):
            x_state = F.relu(layer(x_state))
            if layer_norms is not None and i < len(layer_norms):
                x_state = layer_norms[i](x_state)
            if self.dropout is not None:
                x_state = self.dropout(x_state)
        
        # Process action
        x_action = F.relu(action_layer(action))
        
        # Fuse state and action
        if self.fusion_layer > 0:
            x = torch.cat([x_state, x_action], dim=1)
        else:
            x = torch.cat([state, action], dim=1)
        
        # Process through fused layers
        for i, layer in enumerate(fused_layers[:-1]):  # All but last layer
            x = F.relu(layer(x))
            
            if layer_norms is not None and i + self.fusion_layer < len(layer_norms):
                x = layer_norms[i + self.fusion_layer](x)
            
            if residual_blocks is not None and i + self.fusion_layer < len(residual_blocks):
                x = residual_blocks[i + self.fusion_layer](x)
            
            if self.dropout is not None:
                x = self.dropout(x)
        
        # Output layer
        q_value = fused_layers[-1](x)
        return q_value
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both critic networks."""
        
        # Q1 network
        q1 = self._forward_single_critic(
            state, action, self.state_layers, self.action_layer, self.fused_layers,
            self.layer_norms_q1, self.residual_blocks_q1
        )
        
        # Q2 network (twin)
        q2 = self._forward_single_critic(
            state, action, self.twin_state_layers, self.twin_action_layer, self.twin_fused_layers,
            self.layer_norms_q2, self.residual_blocks_q2
        )
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 network only (for actor updates)."""
        return self._forward_single_critic(
            state, action, self.state_layers, self.action_layer, self.fused_layers,
            self.layer_norms_q1, self.residual_blocks_q1
        )

class EnsembleActor(nn.Module):
    """
    Ensemble of actor networks for improved robustness and uncertainty estimation.
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0,
                 num_actors: int = 3, hidden_dims: List[int] = [400, 300],
                 **kwargs):
        super(EnsembleActor, self).__init__()
        
        self.num_actors = num_actors
        self.max_action = max_action
        
        # Create ensemble of actors
        self.actors = nn.ModuleList([
            ImprovedActor(state_dim, action_dim, max_action, hidden_dims, **kwargs)
            for _ in range(num_actors)
        ])
        
        # Mixing weights for ensemble combination
        self.mixing_weights = nn.Parameter(torch.ones(num_actors) / num_actors)
    
    def forward(self, state: torch.Tensor, return_individual: bool = False) -> torch.Tensor:
        """Forward pass through ensemble."""
        
        # Get actions from all actors
        individual_actions = []
        for actor in self.actors:
            action = actor(state)
            individual_actions.append(action)
        
        if return_individual:
            return torch.stack(individual_actions, dim=1)  # (batch, num_actors, action_dim)
        
        # Weighted combination
        actions_tensor = torch.stack(individual_actions, dim=0)  # (num_actors, batch, action_dim)
        weights = F.softmax(self.mixing_weights, dim=0)
        
        # Weighted average
        ensemble_action = torch.sum(weights.unsqueeze(1).unsqueeze(2) * actions_tensor, dim=0)
        
        return ensemble_action
    
    def get_uncertainty(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action prediction and uncertainty estimate."""
        individual_actions = self.forward(state, return_individual=True)
        
        # Mean and standard deviation across ensemble
        mean_action = torch.mean(individual_actions, dim=1)
        std_action = torch.std(individual_actions, dim=1)
        
        return mean_action, std_action

class AdaptiveNoise(nn.Module):
    """
    Learnable noise module for adaptive exploration.
    """
    
    def __init__(self, action_dim: int, initial_noise: float = 0.1):
        super(AdaptiveNoise, self).__init__()
        
        self.action_dim = action_dim
        
        # Learnable noise parameters
        self.noise_scale = nn.Parameter(torch.full((action_dim,), initial_noise))
        self.noise_bias = nn.Parameter(torch.zeros(action_dim))
        
        # Noise correlation matrix (for correlated noise)
        self.noise_correlation = nn.Parameter(torch.eye(action_dim) * 0.1)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate adaptive noise."""
        
        # Generate base noise
        base_noise = torch.randn(batch_size, self.action_dim, device=device)
        
        # Apply correlation
        correlated_noise = torch.matmul(base_noise, self.noise_correlation)
        
        # Scale and bias
        adaptive_noise = correlated_noise * self.noise_scale + self.noise_bias
        
        return adaptive_noise

class StateEncoder(nn.Module):
    """
    State encoder for learning meaningful state representations.
    Can be used to preprocess states before feeding to actor/critic.
    """
    
    def __init__(self, input_dim: int, encoded_dim: int = 64, 
                 use_variational: bool = False):
        super(StateEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        self.use_variational = use_variational
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_dim * (2 if use_variational else 1))
        )
        
        # Decoder for reconstruction (optional)
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation."""
        encoded = self.encoder(state)
        
        if self.use_variational:
            # Split into mean and log variance
            mu, log_var = torch.chunk(encoded, 2, dim=-1)
            
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            return z, mu, log_var
        else:
            return encoded
    
    def decode(self, encoded_state: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to state."""
        return self.decoder(encoded_state)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        if self.use_variational:
            z, _, _ = self.encode(state)
            return z
        else:
            return self.encode(state)

class NetworkFactory:
    """
    Factory class for creating different network architectures.
    Provides easy configuration and instantiation of networks.
    """
    
    @staticmethod
    def create_actor(config: dict) -> nn.Module:
        """Create actor network based on configuration."""
        
        architecture = config.get('architecture', 'standard')
        
        if architecture == 'standard':
            return ImprovedActor(
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                max_action=config.get('max_action', 1.0),
                hidden_dims=config.get('hidden_dims', [400, 300]),
                use_attention=config.get('use_attention', False),
                use_residual=config.get('use_residual', False),
                use_layer_norm=config.get('use_layer_norm', True),
                dropout=config.get('dropout', 0.1),
                use_spectral_norm=config.get('use_spectral_norm', False)
            )
        
        elif architecture == 'ensemble':
            return EnsembleActor(
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                max_action=config.get('max_action', 1.0),
                num_actors=config.get('num_actors', 3),
                hidden_dims=config.get('hidden_dims', [400, 300]),
                use_attention=config.get('use_attention', False),
                use_residual=config.get('use_residual', False),
                use_layer_norm=config.get('use_layer_norm', True),
                dropout=config.get('dropout', 0.1)
            )
        
        else:
            raise ValueError(f"Unknown actor architecture: {architecture}")
    
    @staticmethod
    def create_critic(config: dict) -> nn.Module:
        """Create critic network based on configuration."""
        
        return ImprovedCritic(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dims=config.get('hidden_dims', [400, 300]),
            use_attention=config.get('use_attention', False),
            use_residual=config.get('use_residual', False),
            use_layer_norm=config.get('use_layer_norm', True),
            dropout=config.get('dropout', 0.1),
            use_spectral_norm=config.get('use_spectral_norm', False),
            fusion_layer=config.get('fusion_layer', 1)
        )
    
    @staticmethod
    def create_state_encoder(config: dict) -> Optional[nn.Module]:
        """Create state encoder if specified in configuration."""
        
        if not config.get('use_state_encoder', False):
            return None
        
        return StateEncoder(
            input_dim=config['state_dim'],
            encoded_dim=config.get('encoded_dim', 64),
            use_variational=config.get('use_variational_encoder', False)
        )

# Utility functions for network analysis and debugging

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_network(model: nn.Module, input_shape: Tuple[int, ...]) -> dict:
    """Analyze network architecture and provide statistics."""
    
    total_params = count_parameters(model)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Forward pass to get output shape
    model.eval()
    with torch.no_grad():
        if isinstance(model, (ImprovedActor, EnsembleActor)):
            output = model(dummy_input)
            output_shape = output.shape[1:]
        elif isinstance(model, ImprovedCritic):
            dummy_action = torch.randn(1, 4)  # Assuming 4D action space
            output = model(dummy_input, dummy_action)
            output_shape = output[0].shape[1:]  # Q1 output shape
        else:
            output_shape = "Unknown"
    
    model.train()
    
    return {
        'total_parameters': total_params,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'model_type': type(model).__name__,
        'memory_usage_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def initialize_weights_xavier(model: nn.Module):
    """Initialize model weights using Xavier initialization."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def initialize_weights_he(model: nn.Module):
    """Initialize model weights using He initialization."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def freeze_layers(model: nn.Module, layer_names: List[str]):
    """Freeze specific layers in the model."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break

def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """Unfreeze specific layers in the model."""
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break