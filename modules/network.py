import torch.nn as nn
import torch
import numpy as np


def get_activation_function(act_type):
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'elu':
        return nn.ELU()
    elif act_type == 'sigmoid':
        return nn.Sigmoid()
    elif act_type == 'tanh':
        return nn.Tanh()
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif act_type == 'silu':
        return nn.SiLU()
    elif act_type == 'gelu':
        return nn.GELU(approximate='tanh')  # 'tanh' is the default approximation
    else:
        raise ValueError(f"Unsupported activation function: {act_type}")
    
class FiLMLinear(nn.Module):
    """Linear layer with FiLM modulation: y = γ ⊙ (Wx + b) + β."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, gamma=None, beta=None):
        h = self.lin(x)
        if gamma is not None:
            h = gamma * h + beta
        return h

class SharedFiLMBank(nn.Module):
    """
    Unified FiLM conditioner that maps the parameter vector to shared embeddings
    and produces gain/bias pairs for encoder, decoder, and auxiliary omega nets.
    """
    def __init__(self, p_dim, emb_dim,
                 enc_layer_dims, dec_layer_dims, omg_layer_dims,
                 gamma_init=1.0, beta_init=0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(p_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.ReLU()
        )
        self.gamma_init = gamma_init
        self.beta_init  = beta_init

        self.enc_heads_dims = enc_layer_dims
        self.dec_heads_dims = dec_layer_dims
        self.omg_heads_dims = omg_layer_dims

        self.enc_heads = nn.ModuleList(
            [nn.Linear(emb_dim, 2 * d) for d in enc_layer_dims]
        )
        self.dec_heads = nn.ModuleList(
            [nn.Linear(emb_dim, 2 * d) for d in dec_layer_dims]
        )
        self.omg_heads = nn.ModuleList(
            [nn.Linear(emb_dim, 2 * d) for d in omg_layer_dims]
        )

    def _split(self, out, dim):
        g, b = out.chunk(2, dim=-1)
        g = self.gamma_init + 0.1 * torch.tanh(g)
        b = self.beta_init  + 0.1 * torch.tanh(b)
        return g, b

    def forward(self, p):
        emb = self.shared(p)
        enc_gb, dec_gb, omg_gb = [], [], []
        for head, d in zip(self.enc_heads, self.enc_heads_dims):
            enc_gb.append(self._split(head(emb), d))
        for head, d in zip(self.dec_heads, self.dec_heads_dims):
            dec_gb.append(self._split(head(emb), d))
        for head, d in zip(self.omg_heads, self.omg_heads_dims):
            omg_gb.append(self._split(head(emb), d))
        return emb, enc_gb, dec_gb, omg_gb
    
class ResidualBlock(nn.Module):
    """
    Standard residual block:
      - Input dimension equals `in_dim`.
      - Hidden layers follow `block_dims` (e.g., [128, 128]).
      - A projection is inserted when the output dimension differs from `in_dim`.
    """
    def __init__(self, in_dim, block_dims, act_type='relu', norm=False):
        super(ResidualBlock, self).__init__()
        self.norm = norm
        layers = []
        current_dim = in_dim
        for hidden_dim in block_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation_function(act_type))
            current_dim = hidden_dim
        self.block = nn.Sequential(*layers)
        self.projection = None
        # Add a projection when skip-connection dimensions differ
        if current_dim != in_dim:
            proj_layers = []
            proj_layers.append(nn.Linear(in_dim, current_dim))
            if norm:
                proj_layers.append(nn.BatchNorm1d(current_dim))
            self.projection = nn.Sequential(*proj_layers)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.projection is not None:
            residual = self.projection(x)
        return out + residual

class ResidualEncoder(nn.Module):
    """
    Residual encoder defined by a list such as
    [input_dim, [128, 128], [256, 256], [128, 128], output_dim],
    where each nested list specifies a residual block.
    """
    def __init__(self, structure, act_type='relu', norm=False):
        super(ResidualEncoder, self).__init__()
        self.norm = norm
        self.act_type = act_type
        
        self.input_dim = structure[0]
        self.output_dim = structure[-1]
        
        self.blocks = nn.ModuleList()
        current_dim = self.input_dim
        for block_dims in structure[1:-1]:
            block = ResidualBlock(current_dim, block_dims, act_type=act_type, norm=norm)
            self.blocks.append(block)
            current_dim = block_dims[-1]
        
        if current_dim != self.output_dim:
            self.final_layer = nn.Linear(current_dim, self.output_dim)
        else:
            self.final_layer = nn.Identity()
    
    def forward(self, x):
        """Accepts 2D (batch, input_dim) or 3D (batch, time, input_dim) tensors."""
        if x.dim() == 2:
            out = x
            for block in self.blocks:
                out = block(out)
            out = self.final_layer(out)
            return out
        
        # Encode each time step independently for 3D input
        batch_size, time_steps, _ = x.size()
        out_list = []
        for t in range(time_steps):
            x_t = x[:, t, :]
            out_t = x_t
            for block in self.blocks:
                out_t = block(out_t)
            out_t = self.final_layer(out_t)
            out_list.append(out_t)
        return torch.stack(out_list, dim=1)
    
class ResidualDecoder(nn.Module):
    """
    Residual decoder mirroring the encoder definition, e.g.
    [latent_dim, [128, 128], [256, 256], [128, 128], original_dim].
    """
    def __init__(self, structure, act_type='relu', norm=False):
        super(ResidualDecoder, self).__init__()
        self.input_dim = structure[0]
        self.output_dim = structure[-1]
        
        self.blocks = nn.ModuleList()
        current_dim = self.input_dim
        for block_dims in structure[1:-1]:
            block = ResidualBlock(current_dim, block_dims, act_type=act_type, norm=norm)
            self.blocks.append(block)
            current_dim = block_dims[-1]
        
        if current_dim != self.output_dim:
            self.final_layer = nn.Linear(current_dim, self.output_dim)
        else:
            self.final_layer = nn.Identity()
    
    def forward(self, x):
        """Accepts 2D (batch, input_dim) or 3D (batch, time, input_dim) tensors."""
        if x.dim() == 2:
            out = x
            for block in self.blocks:
                out = block(out)
            out = self.final_layer(out)
            return out
        
        # Decode each time step independently for 3D input
        batch_size, time_steps, _ = x.size()
        out_list = []
        for t in range(time_steps):
            x_t = x[:, t, :]
            out_t = x_t
            for block in self.blocks:
                out_t = block(out_t)
            out_t = self.final_layer(out_t)
            out_list.append(out_t)
        return torch.stack(out_list, dim=1)

class Encoder(nn.Module):
    def __init__(self, encoder_layers, shifts_input, act_type='relu', norm=False):
        super(Encoder, self).__init__()
        self.act_type = act_type
        self.num_layers = len(encoder_layers) - 1
        self.shifts_input = shifts_input
        self.norm = norm
        layers = []
        for i in range(self.num_layers):
            in_dim = encoder_layers[i]
            out_dim = encoder_layers[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if norm and i < self.num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
            if i < self.num_layers - 1:
                layers.append(get_activation_function(act_type))
        self.encoder_net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Encode each snapshot independently:
        x: (batch, T, origin_dim) -> g_list: (batch, T, koopman_dim)
        """
        if x.dim() == 2:
            return self.encoder_net(x)
        batch_size, time_steps, origin_dim = x.size()
        g_list = []
        for t in range(self.shifts_input + 1):
            x_t = x[:, t, :]
            g_t = self.encoder_net(x_t)
            g_list.append(g_t)

        g_tensor = torch.stack(g_list, dim=1)
        return g_tensor

    def forward_debug(self, x):
        """Return intermediate outputs of each encoder block for inspection."""
        if x.dim() == 2:
            outputs = []
            for layer in self.encoder_net:
                x = layer(x)
                outputs.append(x)
            return outputs

        # For 3D input, not implemented yet
        raise NotImplementedError("forward_debug does not support 3D input yet.")

class Decoder(nn.Module):
    def __init__(self, decoder_layers, shifts_pred, act_type='relu', norm=False):
        super(Decoder, self).__init__()
        self.act_type = act_type
        self.num_layers = len(decoder_layers) - 1
        self.shifts_pred = shifts_pred
        self.norm = norm
        layers = []
        # Example: decoder_layers = [koopman_dim, h2, h1, origin_dim]
        for i in range(self.num_layers):
            in_dim = decoder_layers[i]
            out_dim = decoder_layers[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if norm and i < self.num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
            if i < self.num_layers - 1:
                layers.append(get_activation_function(act_type))
        self.decoder_net = nn.Sequential(*layers)

    def forward(self, g_tensor):
        """
        Decode latent representation back to the original coordinates.
        g_tensor: (batch_size, koopman_dim)
        """
        return self.decoder_net(g_tensor)
   
class KoopmanNet(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, act_type,
                 num_complex_pairs, num_real, delta_t, shifts, shifts_input, shifts_pred,
                 widths_omega_complex, widths_omega_real, norm=False, residual=False):
        super(KoopmanNet, self).__init__()
        self.num_complex_pairs = num_complex_pairs
        self.num_real = num_real
        self.delta_t = delta_t
        self.shifts = shifts
        self.shifts_input = shifts_input
        self.shifts_pred = shifts_pred
        self.norm = norm
        self.residual = residual

        self.encoder = Encoder(encoder_layer, self.shifts_input, act_type, norm) if not residual else ResidualEncoder(encoder_layer, act_type, norm)
        self.decoder = Decoder(decoder_layer, self.shifts_pred, act_type, norm) if not residual else ResidualDecoder(decoder_layer, act_type, norm)
        self.act_type = act_type

        # Build omega networks for complex conjugate blocks (radius input)
        self.omega_complex_nets = nn.ModuleList()
        for _ in range(num_complex_pairs):
            self.omega_complex_nets.append(self.build_mlp(widths_omega_complex, act_type))

        # Real eigenvalues consume the 1D coordinate directly
        self.omega_real_nets = nn.ModuleList()
        for _ in range(num_real):
            self.omega_real_nets.append(self.build_mlp(widths_omega_real, act_type))

    def build_mlp(self, layer_widths, act_type):
        # layer_widths: e.g. [1, 10, 10, 2 or 1]
        layers = []
        for i in range(len(layer_widths)-1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i+1]))
            # Skip activation on the final layer
            if i < len(layer_widths)-2:
                layers.append(get_activation_function(act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass:
        1. Encode x into Koopman latent space.
        2. Decode the initial latent state for reconstruction.
        3. Roll out in latent space and decode each predicted step.
        """
        g_list = self.encoder(x)
        decoded_list = [self.decoder(g_list[:, 0, :])]
        g_list_vary = []

        advanced_layer = g_list[:, 0, :]
        g_list_vary.append(advanced_layer)
        max_shift = self.shifts_input

        egin_list = []
        for step in range(max_shift):
            omegas = self.forward_omegas(advanced_layer)
            advanced_layer = self.varying_multiply(advanced_layer, omegas, self.delta_t)
            g_list_vary.append(advanced_layer)
            decoded_list.append(self.decoder(advanced_layer))
            egin_list.append(omegas)

        g_list_vary_tensor = torch.stack(g_list_vary, dim=1)
        decoded_tensor = torch.stack(decoded_list, dim=1)
        return g_list, g_list_vary_tensor, decoded_tensor, omegas
    
    def forward_once_for_infer(self, x, infer_times=3, delta_t=None):
        """Finite-horizon inference with optional custom Δt."""
        g = self.encoder(x)
        decoded = self.decoder(g)
        
        y_list = [g]
        x_list = [decoded]
        omega_list = []
        for t in range(infer_times):
            omegas = self.forward_omegas(g)
            omega_list.append(omegas)
            g = self.varying_multiply(g, omegas, self.delta_t if delta_t is None else delta_t)
            y_list.append(g)
            x_list.append(self.decoder(g))

        return y_list, x_list, omega_list
    
    def forward_once_for_gradient(self, z, deltat):
        """
        Execute one propagation step with a scaled Δt and return the difference quotient.
        """
        g = z  # [batch_size, koopman_dim]
        decoded = self.decoder(z)
        
        y_list = [g]
        x_list = [decoded]
        omega_list = []
        omegas = self.forward_omegas(g)
        # omega_list.append([o.detach().cpu().numpy() for o in omegas])
        omega_list.append(omegas)
        g = self.varying_multiply(g, omegas, deltat)
        y_list.append(g)
        x_list.append(self.decoder(g))
        
        gradient = (y_list[1] - y_list[0])/deltat  # [batch_size, koopman_dim]
            
        return gradient

    def forward_omegas(self, ycoords):
        """
        ycoords: latent state of shape [batch_size, koopman_dim].
        Produce omega parameters for each eigenmode.
        """
        batch_size = ycoords.size(0)
        omegas = []

        # Complex conjugate pairs consume two dimensions each
        for j in range(self.num_complex_pairs):
            ind = 2*j
            pair_of_columns = ycoords[:, ind:ind+2]
            radius_of_pair = torch.sum(pair_of_columns**2, dim=1, keepdim=True)
            omega_out = self.omega_complex_nets[j](radius_of_pair)
            omegas.append(omega_out)

        # Real eigenvalues
        for j in range(self.num_real):
            ind = 2*self.num_complex_pairs + j
            one_column = ycoords[:, ind:ind+1]
            omega_out = self.omega_real_nets[j](one_column)
            omegas.append(omega_out)

        return omegas

    def form_complex_conjugate_block(self, omegas_block, delta_t):
        """
        omegas_block: [batch_size, 2] with [omega, mu].
        Returns a [batch_size, 2, 2] block matrix:
        exp(mu * dt) * [[cos(omega*dt), -sin(omega*dt)],
                        [sin(omega*dt), cos(omega*dt)]]
        """
        omega = omegas_block[:, 0]
        mu = omegas_block[:, 1]
        scale = torch.exp(mu * delta_t)
        cos_val = torch.cos(omega * delta_t)
        sin_val = torch.sin(omega * delta_t)
        row1 = torch.stack([scale*cos_val, -scale*sin_val], dim=1)  # [batch_size, 2]
        row2 = torch.stack([scale*sin_val, scale*cos_val], dim=1)
        return torch.stack([row1, row2], dim=2)  # [batch_size, 2, 2]

    def varying_multiply(self, y, omegas_list, delta_t):
        """
        y: [batch, k], k = 2*num_complex_pairs + num_real.
        omegas_list: output from forward_omegas(), one entry per eigenmode.
        Apply the Koopman operator once and return the updated latent state.
        """
        batch_size = y.size(0)
        # Complex conjugate pairs
        complex_parts = []
        offset = 0
        for j in range(self.num_complex_pairs):
            ind = 2*j
            y_pair = y[:, ind:ind+2]
            L_block = self.form_complex_conjugate_block(omegas_list[j], delta_t)
            y_pair_expanded = y_pair.unsqueeze(1)
            next_pair = torch.matmul(y_pair_expanded, L_block).squeeze(1)
            complex_parts.append(next_pair)
        offset = self.num_complex_pairs

        # Real eigenvalues act diagonally
        real_parts = []
        for j in range(self.num_real):
            ind = 2*self.num_complex_pairs + j
            y_col = y[:, ind:ind+1]
            mu_val = omegas_list[self.num_complex_pairs + j][:, 0]
            scale = torch.exp(mu_val * delta_t).unsqueeze(1)
            next_col = y_col * scale
            real_parts.append(next_col)

        if complex_parts and real_parts:
            return torch.cat(complex_parts + real_parts, dim=1)
        elif complex_parts:
            return torch.cat(complex_parts, dim=1)
        else:
            return torch.cat(real_parts, dim=1) if real_parts else y

class FiLMMLP(nn.Module):
    """
    layer_widths: [in, h1, h2, ..., out]
    film_layers_idx: indices of Linear layers to be FiLM-modulated
    """
    def __init__(self, layer_widths, act_type, film_layers_idx=None):
        super().__init__()
        self.n_layers = len(layer_widths) - 1
        self.film_layers_idx = list(film_layers_idx or range(self.n_layers))

        self.layers = nn.ModuleList()
        self.acts   = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(FiLMLinear(layer_widths[i], layer_widths[i+1]))
            self.acts.append(get_activation_function(act_type) if i < self.n_layers-1 else nn.Identity())

    def forward(self, x, gb_sub=None):
        """
        gb_sub: list of (gamma, beta) pairs ordered like film_layers_idx.
        """
        h = x
        for i, (lin, act) in enumerate(zip(self.layers, self.acts)):
            gamma = beta = None
            if gb_sub is not None and i in self.film_layers_idx:
                # Map the linear-layer index to the gb_sub entry
                idx = self.film_layers_idx.index(i)
                gamma, beta = gb_sub[idx]
            h = lin(h, gamma, beta)
            h = act(h)
        return h
    
class FiLMEncoder(nn.Module):
    def __init__(self, encoder_layers, shifts_input, act_type,
                 norm=False, film_apply_to=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts   = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.shifts_input = shifts_input
        self.film_apply_to = sorted(film_apply_to or [])

        for i in range(len(encoder_layers) - 1):
            self.layers.append(FiLMLinear(encoder_layers[i], encoder_layers[i+1]))
            if norm and i < len(encoder_layers) - 1:
                self.norms.append(nn.BatchNorm1d(encoder_layers[i+1]))
            else:
                self.norms.append(None) # type: ignore
            self.acts.append(get_activation_function(act_type) if i < len(encoder_layers)-1 else nn.Identity())

    def _forward_single(self, x, gb_map=None):
        h = x
        for i, (lin, act, bn) in enumerate(zip(self.layers, self.acts, self.norms)):
            gamma = beta = None
            if gb_map is not None and i in gb_map:
                gamma, beta = gb_map[i]
            h = lin(h, gamma, beta)
            if bn is not None and h.dim() == 2:
                h = bn(h)
            h = act(h)
        return h

    def forward(self, x, enc_gb_list=None):
        # enc_gb_list follows self.film_apply_to
        gb_map = None
        if enc_gb_list is not None:
            gb_map = {layer_idx: gb for layer_idx, gb in zip(self.film_apply_to, enc_gb_list)}

        if x.dim() == 2:
            return self._forward_single(x, gb_map)
        B, T, _ = x.size()
        outs = []
        for t in range(self.shifts_input + 1):
            outs.append(self._forward_single(x[:, t, :], gb_map))
        return torch.stack(outs, dim=1)

class FiLMDecoder(nn.Module):
    def __init__(self, decoder_layers, act_type,
                 norm=False, film_apply_to=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts   = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.film_apply_to = sorted(film_apply_to or [])

        for i in range(len(decoder_layers) - 1):
            self.layers.append(FiLMLinear(decoder_layers[i], decoder_layers[i+1]))
            if norm and i < len(decoder_layers) - 1:
                self.norms.append(nn.BatchNorm1d(decoder_layers[i+1]))
            else:
                self.norms.append(None)  # type: ignore
            self.acts.append(get_activation_function(act_type) if i < len(decoder_layers)-1 else nn.Identity())

    def _forward_single(self, x, gb_map=None):
        h = x
        for i, (lin, act, bn) in enumerate(zip(self.layers, self.acts, self.norms)):
            gamma = beta = None
            if gb_map is not None and i in gb_map:
                gamma, beta = gb_map[i]
            h = lin(h, gamma, beta)
            if bn is not None and h.dim() == 2:
                h = bn(h)
            h = act(h)
        return h

    def forward(self, x, dec_gb_list=None):
        gb_map = None
        if dec_gb_list is not None:
            gb_map = {layer_idx: gb for layer_idx, gb in zip(self.film_apply_to, dec_gb_list)}

        if x.dim() == 2:
            return self._forward_single(x, gb_map)
        B, T, _ = x.size()
        outs = []
        for t in range(T):
            outs.append(self._forward_single(x[:, t, :], gb_map))
        return torch.stack(outs, dim=1)
    
class FiLMKoopmanNetFull(nn.Module):
    """
    Full FiLM-based Koopman network.

    This class implements:
    - Layer-wise FiLM modulation for the encoder.
    - Layer-wise FiLM modulation for all omega_complex/omega_real subnetworks.
    - A shared parameter embedding that feeds separate FiLM banks for encoder
      and Koopman (omega) branches.

    Notes
    -----
    * The code structure mirrors the original (non-FiLM) KoopmanNet while
      inserting FiLM coefficients at user-specified layer indices.
    * No functional changes have been introduced relative to the provided
      implementation; only docstrings and comments have been refined for
      academic clarity.
    """

    def __init__(self, encoder_layer, decoder_layer, act_type,
                 num_complex_pairs, num_real, delta_t, shifts, shifts_input, shifts_pred,
                 widths_omega_complex, widths_omega_real,
                 p_dim, emb_dim=64,
                 enc_film_idx=(0, 1), dec_film_idx=(), omg_film_idx=(0, 1),
                 norm=False, residual=False):

        super().__init__()
        # ----------------------- Basic attributes -----------------------
        self.num_complex_pairs = num_complex_pairs
        self.num_real = num_real
        self.delta_t = delta_t
        self.shifts = shifts
        self.shifts_input = shifts_input
        self.shifts_pred = shifts_pred
        
        # ----------------------- Omega width normalization -----------------------
        # If `widths_omega_complex` is provided as a single template list (e.g., [1,16,2]),
        # replicate it for each complex pair. Same for the real blocks.
        if isinstance(widths_omega_complex[0], (int, float)):
            template_c = widths_omega_complex[:]          # e.g., [1,16,2]
            widths_omega_complex = [template_c for _ in range(num_complex_pairs)]

        if isinstance(widths_omega_real[0], (int, float)):
            template_r = widths_omega_real[:]
            widths_omega_real = [template_r for _ in range(num_real)]

        # 1) Dimensions of encoder/decoder layers to be FiLM-modulated
        enc_dims_mod = [encoder_layer[i + 1] for i in enc_film_idx]
        dec_dims_mod = [decoder_layer[i + 1] for i in dec_film_idx]

        # --------- 2) Omega-net dimensions to be FiLM-modulated ----------
        omg_dims_mod = []
        self.omega_complex_film_splits = []
        self.omega_real_film_splits = []

        # Track which layers in each omega MLP will be FiLM'ed
        self.omega_complex_layers_idx = []
        for w in widths_omega_complex:
            # Valid linear layer indices for FiLM: 0..len(w)-2 (final layer often excluded)
            idx = [i for i in omg_film_idx if i < len(w) - 1]
            self.omega_complex_layers_idx.append(idx)
            # Accumulate the output dims of those FiLM'ed layers: w[i+1]
            omg_dims_mod.extend([w[i + 1] for i in idx])
            self.omega_complex_film_splits.append(len(idx))

        self.omega_real_layers_idx = []
        for w in widths_omega_real:
            idx = [i for i in omg_film_idx if i < len(w) - 1]
            self.omega_real_layers_idx.append(idx)
            omg_dims_mod.extend([w[i + 1] for i in idx])
            self.omega_real_film_splits.append(len(idx))

        # 3) Shared FiLM Bank
        # Generates gain/bias for both encoder and omega branches from the parameter vector p.
        self.film_bank = SharedFiLMBank(p_dim, emb_dim,
                                        enc_layer_dims=enc_dims_mod,
                                        dec_layer_dims=dec_dims_mod,
                                        omg_layer_dims=omg_dims_mod)

        # 4) Encoder / Decoder (FiLM applied to specified encoder/decoder layers)
        self.encoder = FiLMEncoder(encoder_layer, shifts_input, act_type,
                                   norm=norm,
                                   film_apply_to=enc_film_idx)
        if not residual:
            self.decoder = FiLMDecoder(decoder_layer, act_type,
                                       norm=norm,
                                       film_apply_to=dec_film_idx)
        else:
            self.decoder = ResidualDecoder(decoder_layer, act_type, norm)

        # 5) Ω-networks (FiLMMLP blocks for complex and real parts)
        self.omega_complex_nets = nn.ModuleList([
            FiLMMLP(w, act_type, film_layers_idx=self.omega_complex_layers_idx[k])
            for k, w in enumerate(widths_omega_complex)
        ])
        self.omega_real_nets = nn.ModuleList([
            FiLMMLP(w, act_type, film_layers_idx=self.omega_real_layers_idx[k])
            for k, w in enumerate(widths_omega_real)
        ])

    def decode_latent(self, z, dec_gb):
        if isinstance(self.decoder, FiLMDecoder):
            return self.decoder(z, dec_gb)
        return self.decoder(z)

    # ---------------- Helper for Ω computation ----------------
    def forward_omegas(self, ycoords, omg_gb_list):
        """
        Compute the Koopman spectral parameters (omegas) for each latent block.

        Parameters
        ----------
        ycoords : torch.Tensor
            Current latent state (B, z_dim). The first 2*num_complex_pairs dims
            correspond to complex conjugate blocks, the remaining num_real dims
            are real blocks.
        omg_gb_list : list[tuple(torch.Tensor, torch.Tensor)]
            A list of (gamma, beta) pairs for FiLM modulation of omega MLP layers.

        Returns
        -------
        omegas : list[torch.Tensor]
            Outputs of each omega subnetwork. For complex blocks, typically shape (B, 2)
            containing [omega, mu]. For real blocks, shape (B, 1) containing [mu].
        """
        omegas = []
        ptr = 0

        # Complex blocks: use radius in 2D subspace as input
        for j, mlp in enumerate(self.omega_complex_nets):
            radius = torch.sum(ycoords[:, 2*j:2*j+2]**2, dim=1, keepdim=True)
            nL = self.omega_complex_film_splits[j]
            gb_sub = omg_gb_list[ptr:ptr+nL] if nL > 0 else None
            ptr += nL
            omega_out = mlp(radius, gb_sub)
            omegas.append(omega_out)

        # Real blocks: use the 1D coordinate directly
        base = 2 * self.num_complex_pairs
        for j, mlp in enumerate(self.omega_real_nets):
            col = ycoords[:, base + j:base + j + 1]
            nL = self.omega_real_film_splits[j]
            gb_sub = omg_gb_list[ptr:ptr+nL] if nL > 0 else None
            ptr += nL
            omega_out = mlp(col, gb_sub)
            omegas.append(omega_out)

        return omegas

    @staticmethod
    def form_complex_conjugate_block(omegas_block, delta_t):
        """
        Form the 2x2 real block corresponding to a complex conjugate pair.

        Parameters
        ----------
        omegas_block : torch.Tensor
            Spectral output for the complex block, shape (B, 2) where the first
            component denotes angular frequency (omega) and the second is the
            real part (mu).
        delta_t : float
            Time step for discrete-time propagation.

        Returns
        -------
        torch.Tensor
            A tensor of shape (B, 2, 2) representing the block matrix:
            exp(mu*dt) * [[cos(omega*dt), -sin(omega*dt)],
                          [sin(omega*dt),  cos(omega*dt)]]
        """
        omega = omegas_block[:, 0]
        mu    = omegas_block[:, 1]
        scale = torch.exp(mu * delta_t)
        cos_v = torch.cos(omega * delta_t)
        sin_v = torch.sin(omega * delta_t)
        row1 = torch.stack([scale * cos_v, -scale * sin_v], dim=1)
        row2 = torch.stack([scale * sin_v,  scale * cos_v], dim=1)
        return torch.stack([row1, row2], dim=2)

    def varying_multiply(self, y, omegas_list, delta_t):
        """
        Apply the parameter-varying Koopman operator in latent space.

        Parameters
        ----------
        y : torch.Tensor
            Latent state, shape (B, z_dim).
        omegas_list : list[torch.Tensor]
            Spectral parameters for each latent block produced by `forward_omegas`.
        delta_t : float
            Time increment used for this update.

        Returns
        -------
        torch.Tensor
            Next-step latent state of shape (B, z_dim).
        """
        complex_parts, real_parts = [], []

        # Complex conjugate blocks
        for j in range(self.num_complex_pairs):
            ind = 2 * j
            y_pair = y[:, ind:ind+2]
            L_block = self.form_complex_conjugate_block(omegas_list[j], delta_t)
            y_pair_expanded = y_pair.unsqueeze(1)
            next_pair = torch.matmul(y_pair_expanded, L_block).squeeze(1)
            complex_parts.append(next_pair)

        # Real eigenvalue blocks
        for j in range(self.num_real):
            ind = 2 * self.num_complex_pairs + j
            y_col = y[:, ind:ind+1]
            mu_val = omegas_list[self.num_complex_pairs + j][:, 0]
            scale = torch.exp(mu_val * delta_t).unsqueeze(1)
            next_col = y_col * scale
            real_parts.append(next_col)

        # Concatenate updated blocks
        if complex_parts and real_parts:
            return torch.cat(complex_parts + real_parts, dim=1)
        elif complex_parts:
            return torch.cat(complex_parts, dim=1)
        else:
            return torch.cat(real_parts, dim=1) if real_parts else y
        
    def encoder_forward(self, x, p):
        # 1) Shared embedding and FiLM gain/bias sets for encoder, decoder, and omega nets
        p_emb, enc_gb, dec_gb, omg_gb = self.film_bank(p)

        # 2) Encode (FiLM applied layer-wise). `encoder` returns the latent sequence.
        g_list = self.encoder(x, enc_gb)
        
        return g_list

    # ------------------ Main forward pass ------------------
    def forward(self, x, p):
        """
        Forward pass through the FiLM-conditioned Koopman network.

        Parameters
        ----------
        x : torch.Tensor
            Either (B, T, x_dim) for a sequence or (B, x_dim) for a single snapshot.
        p : torch.Tensor
            Parameter vector (B, p_dim) that conditions both encoder/decoder and Koopman blocks.

        Returns
        -------
        g_list : torch.Tensor
            Encoded latent sequence for the input (B, T, z_dim).
        g_list_vary_tensor : torch.Tensor
            Latent rollout under the learned PC-Koopman operator
            with shape (B, shifts+1, z_dim), starting from t=0 latent.
        decoded_tensor : torch.Tensor
            Reconstructed states corresponding to each latent in g_list_vary_tensor,
            shape (B, shifts+1, x_dim).
        omegas : list[torch.Tensor]
            Spectral parameters (last step) returned by forward_omegas for the final latent.
        """
        # 1) Shared embedding and FiLM gain/bias sets for encoder, decoder, and omega nets
        p_emb, enc_gb, dec_gb, omg_gb = self.film_bank(p)

        # 2) Encode (FiLM applied layer-wise). `encoder` returns the latent sequence.
        g_list = self.encoder(x, enc_gb)

        # Initialize containers: first decoded state and latent at t=0
        decoded_list = [self.decode_latent(g_list[:, 0, :], dec_gb)]
        g_list_vary  = [g_list[:, 0, :]]
        egin_list    = []

        # Start rolling out from latent at t=0
        adv = g_list[:, 0, :]
        omegas = None  # Ensure omegas is always defined
        for _ in range(self.shifts_input):
            omegas = self.forward_omegas(adv, omg_gb)
            adv    = self.varying_multiply(adv, omegas, self.delta_t)
            g_list_vary.append(adv)
            decoded_list.append(self.decode_latent(adv, dec_gb))
            egin_list.append(omegas)

        g_list_vary_tensor = torch.stack(g_list_vary, dim=1)
        decoded_tensor     = torch.stack(decoded_list, dim=1)
        return g_list, g_list_vary_tensor, decoded_tensor, omegas
    
    def forward_once_for_infer(self, x, p, infer_times: int = 3):
        """
        Perform deterministic inference for a finite horizon using the learned
        parameter-varying Koopman operator.

        Parameters
        ----------
        x : torch.Tensor
            Input state sequence with shape (B, T, x_dim) or a single snapshot
            (B, x_dim). Only the first time step is encoded for rollout.
        p : torch.Tensor
            Parameter vector (control and operating conditions), shape (B, p_dim),
            used to generate FiLM coefficients and the PV-Koopman operator.
        infer_times : int, optional
            Number of forward propagation steps in latent space, by default 3.

        Returns
        -------
        y_list : list[torch.Tensor]
            Latent states at each step, length = infer_times + 1.
        x_list : list[torch.Tensor]
            Reconstructed states at each step, length = infer_times + 1.
        omega_list : list[list[torch.Tensor]]
            Per-step Koopman spectral components (e.g., eigenvalues or blocks),
            length = infer_times.

        Notes
        -----
        - This routine keeps the network in evaluation mode for inference-only
        rollouts. Gradients are enabled if the caller requires them.
        - The Koopman update is realized via `varying_multiply`, which applies
        the parameter-conditioned operator to the latent state.
        """
        # 1) Generate FiLM gain/bias for encoder, decoder, and Koopman blocks
        _, enc_gb, dec_gb, omg_gb = self.film_bank(p)

        # 2) Encode only the first snapshot to obtain the initial latent state
        if x.dim() == 3:
            z_full = self.encoder(x, enc_gb)   # (B, T, z_dim)
            z = z_full[:, 0, :]                # use t=0
        else:
            z = self.encoder(x, enc_gb)        # (B, z_dim)

        # 3) Initial reconstruction from the latent space
        x_rec = self.decode_latent(z, dec_gb)

        y_list = [z]
        x_list = [x_rec]
        omega_list = []

        # 4) Roll out in latent space for `infer_times` steps
        for _ in range(infer_times):
            omegas = self.forward_omegas(z, omg_gb)      # spectral terms for this step
            omega_list.append(omegas)

            z = self.varying_multiply(z, omegas, self.delta_t)  # latent update
            y_list.append(z)
            x_list.append(self.decode_latent(z, dec_gb))

        return y_list, x_list, omega_list


    def forward_once_for_gradient(self, z, deltat, p):
        """
        Execute a single latent-space propagation step for gradient-based analyses
        (e.g., sensitivity studies or Lyapunov-related constraints).

        Parameters
        ----------
        z : torch.Tensor
            Current latent state, shape (B, z_dim).
        deltat : float
            Temporal increment for this single propagation step. It may differ
            from the nominal self.delta_t used during training.
        p : torch.Tensor
            Parameter vector (B, p_dim) for FiLM conditioning of the Koopman core.

        Returns
        -------
        y_list : list[torch.Tensor]
            [z, z_next], i.e., the current and one-step-ahead latent states.
        x_list : list[torch.Tensor]
            [x_hat(z), x_hat(z_next)], reconstructions for both latent states.
        omega_list : list[list[torch.Tensor]]
            Spectral components used for this step; length = 1.

        Notes
        -----
        - Designed for use inside custom loss terms where only a single-step
        transition is required.
        - Gradients will flow through both the encoder/decoder and Koopman blocks
        unless `torch.no_grad()` is explicitly applied by the caller.
        """
        # 1) Obtain FiLM parameters for the Koopman block
        _, _, dec_gb, omg_gb = self.film_bank(p)

        # 2) Decode current latent state
        x_rec = self.decode_latent(z, dec_gb)

        # 3) Compute Koopman spectral representation and propagate
        omegas = self.forward_omegas(z, omg_gb)
        z_next = self.varying_multiply(z, omegas, deltat)

        # 4) Decode next latent state
        x_next = self.decode_latent(z_next, dec_gb)

        # return [z, z_next], [x_rec, x_next], [omegas]
        return (z_next - z) / deltat


if __name__ == "__main__":
    import torch
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Hyper-params for a quick sanity run ----------------
    B, T, dx    = 8, 51, 12          # batch, time steps, original state dim
    p_dim       = 6                  # parameter vector dim (J, D, SCR, etc.)
    num_cplx    = 2                  # number of complex conjugate eigenvalue pairs
    num_real    = 2                  # number of real eigenvalues
    koop_dim    = 2 * num_cplx + num_real
    delta_t     = 0.02
    shifts      = 10                 # short rollout just to test
    emb_dim     = 64                 # FiLM shared embedding size
    enc_film_idx = (0, 1)            # which encoder Linear layers to FiLM
    omg_film_idx = [0]           # which omega MLP layers to FiLM

    encoder_layer = [dx, 128, 128, koop_dim]
    decoder_layer = [koop_dim, 128, dx]
    widths_omega_complex = [1, 16, 2]   # radius -> (omega, mu)
    widths_omega_real    = [1,  8, 1]   # column -> (mu)

    # ---------------- Test data ----------------
    x = torch.randn(B, T, dx, device=device)
    p = torch.randn(B, p_dim, device=device)
 
    # ---------------- Build & forward ----------------
    model = FiLMKoopmanNetFull(
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer,
        act_type='relu',
        num_complex_pairs=num_cplx,
        num_real=num_real,
        delta_t=delta_t,
        shifts=shifts,
        shifts_input=T-1,
        shifts_pred=T-1,
        widths_omega_complex=widths_omega_complex,
        widths_omega_real=widths_omega_real,
        p_dim=p_dim,
        emb_dim=emb_dim,
        enc_film_idx=enc_film_idx,
        omg_film_idx=omg_film_idx,
        norm=False,
        residual=False
    ).to(device)

    g_list, g_list_vary, decoded, omegas = model(x, p)

    print("FiLM-Full forward OK")
    print("g_list:",        g_list.shape)
    print("g_list_vary:",   g_list_vary.shape)
    print("decoded:",       decoded.shape)
    print("num omega blocks:", len(omegas))
