import numpy as np
import torch
import torch.nn.functional as F
import wandb

def define_loss(g_list, g_list_vary, decoded, egins, x_true, config, epoch=0, glb_step=0, train=True):
    """
    Arguments
    ---------
    g_list : torch.Tensor
        (batch_size, 51, 2) ground-truth latent trajectory.
    g_list_vary : torch.Tensor
        (batch_size, 31, 2) latent rollout predicted by the Koopman operator.
    decoded : torch.Tensor
        (batch_size, 31, 2) decoded predictions corresponding to g_list_vary.
    x_true : torch.Tensor
        (batch_size, 31, 2) ground-truth trajectories in the original space.

    The `config` object must expose:
    - relative_loss (bool)
    - recon_lam, mid_shift_lam
    - denominator_nonzero (float)
    - Optional attributes such as num_shifts (defaults assume 31 steps here).
    """

    denominator_nonzero = config.denominator_nonzero
    relative_loss = config.relative_loss

    # ========== loss1: Reconstruction Loss ==========
    # Compare decoded[:,0,:] vs. x_true[:,0,:]
    if config.all_step_recong_loss > epoch:
        recon_mse_step1 = F.mse_loss(decoded[:, 0, :], x_true[:, 0, :])
        # recon_mse_step1 = F.mse_loss(decoded, x_true)
    else:
        recon_mse_step1 = F.mse_loss(decoded[:, 0, :], x_true[:, 0, :])
        # recon_mse_step1 = F.mse_loss(decoded, x_true)
    if relative_loss:
        loss1_denominator = torch.mean(x_true[:, 0, :]**2) + denominator_nonzero
    else:
        loss1_denominator = 1.0
    loss1 = config.recon_lam * (recon_mse_step1 / loss1_denominator)

    # ========== loss2: Future prediction Loss ==========
    # Compare decoded[:,1:,:] vs. x_true[:,1:,:] across future steps
    if decoded.size(1) > 1:
        multi_steps_mse = F.mse_loss(decoded[:, 1:config.shifts_pred+1, :], x_true[:, 1:config.shifts_pred+1, :])
        if relative_loss:
            loss2_denominator = torch.mean((x_true[:, 1:config.shifts_pred+1, :]**2)) + denominator_nonzero
        else:
            loss2_denominator = 1.0
        loss2 = config.pred_lam * (multi_steps_mse / loss2_denominator)
    else:
        # Single-step case: no future prediction loss
        loss2 = torch.zeros(1, dtype=torch.float32, device=x_true.device)

    # ========== loss3: Linearization Loss ==========
    # Align Koopman rollout with ground-truth latents
    linearity_mse = F.mse_loss(g_list_vary[:, 1:config.shifts+1, :], g_list[:, 1:config.shifts+1, :])
    if relative_loss:
        loss3_denominator = torch.mean((g_list[:, 1:config.shifts+1, :]**2)) + denominator_nonzero
    else:
        loss3_denominator = 1.0
    loss3 = config.koopman_lam * (linearity_mse / loss3_denominator)
    
    # ----- loss4: Spectral damping margin loss -----
    mu_cols = []
    for eig in egins:
        mu = eig[:, 1] if eig.shape[1] == 2 else eig[:, 0]
        mu_cols.append(mu.unsqueeze(1))
    mu_mat = torch.cat(mu_cols, dim=1)  # (B, R)

    tau = getattr(config, "sdml_tau", 0.2)
    dt = getattr(config, "delta_t", 1.0)
    mu_margin = getattr(config, "sdml_mu_margin", getattr(config, "mu_margin", 0.0))
    z = (dt * (mu_mat - mu_margin)) / tau
    z_max = z.max(dim=1, keepdim=True).values
    lse = z_max + torch.log(torch.exp(z - z_max).sum(dim=1))
    alpha_tilde = (tau / dt) * lse
    sdml = torch.relu(alpha_tilde)
    loss4 = config.egin_lam * sdml.mean()
    
    # ---------- log individual Koopman eigenvalues (8 complex pairs + 8 real roots) ----------
    if not config.disable_log and train:
        spec_log = {}
        # complex conjugate pairs: store mean mu (real part) & omega (imag part) per pair
        for i in range(min(config.num_complex_pairs, len(egins))):
            blk = egins[i]                     # shape (B,2): [imag, real]
            mu_i    = blk[:, 1].mean().item()  # real part
            omg_i   = blk[:, 0].mean().item()  # imag part
            spec_log[f"spec/complex{i}_mu"]    = mu_i
            spec_log[f"spec/complex{i}_omega"] = omg_i
        # real roots follow the complex blocks
        base = config.num_complex_pairs
        for j in range(min(config.num_real, len(egins) - base)):
            blk = egins[base + j]              # shape (B,1)
            mu_r = blk[:, 0].mean().item()
            spec_log[f"spec/real{j}_mu"] = mu_r
        wandb.log(spec_log, step=glb_step)
    
    # ---------- extra monitoring metrics (sequence-level) ----------
    # 1) Energy of true vs. predicted trajectory (mean-square over the window)
    with torch.no_grad():
        traj_len = config.shifts_pred + 1            # typically 31
        seq_true  = x_true[:, :traj_len, :]          # (B,T,dx)
        seq_pred  = decoded[:, :traj_len, :]
        energy_true = (seq_true.pow(2).mean()).item()
        energy_pred = (seq_pred.pow(2).mean()).item()
        energy_ratio = energy_pred / (energy_true + 1e-8)

        # 2) Cosine similarity between flattened trajectories
        flat_true = seq_true.reshape(seq_true.size(0), -1)
        flat_pred = seq_pred.reshape(seq_pred.size(0), -1)
        cos_sim_batch = (flat_true * flat_pred).sum(1) / (
            flat_true.norm(dim=1) * flat_pred.norm(dim=1) + 1e-8
        )
        cos_sim = cos_sim_batch.mean().item()

    if not config.disable_log and train:
        wandb.log({
            "spectrum/seq_energy_true":  energy_true,
            "spectrum/seq_energy_pred":  energy_pred,
            "spectrum/energy_ratio":     energy_ratio,
            "spectrum/cosine_sim":       cos_sim
        }, step=glb_step)
    
    # ========== loss_var & loss_cov: Latent activation balancing ==========
    # Use g_list as latent tensor proxy: shape (batch, T, dim_lat)
    # We flatten time dimension so each row is one latent sample
    latent_flat = g_list.reshape(-1, g_list.shape[-1])  # (batch*T, latent_dim)
    if latent_flat.size(0) > 1:
        # Per-dimension variance toward 1
        var = torch.var(latent_flat, dim=0, unbiased=False) + 1e-8
        loss_var = config.var_lam * F.mse_loss(var, torch.ones_like(var))
        # Covariance off-diagonal decorrelation
        latent_mean = torch.mean(latent_flat, dim=0, keepdim=True)
        latent_center = latent_flat - latent_mean
        cov = (latent_center.t() @ latent_center) / latent_flat.size(0)  # (d,d)
        cov_offdiag = cov - torch.diag(torch.diag(cov))
        loss_cov = config.cov_lam * torch.mean(cov_offdiag ** 2)
    else:
        loss_var = torch.tensor(0., device=x_true.device)
        loss_cov = torch.tensor(0., device=x_true.device)
    
    # ========== loss_Linf: Linf penalty Loss ==========
    # inf norm on autoencoder error and one prediction step
    if config.relative_loss:
        Linf1_den = torch.norm(torch.norm(x_true[:, 0, :], dim=1, p=float('inf')), p=float('inf')) + denominator_nonzero
        Linf2_den = torch.norm(torch.norm(x_true[:, 1, :], dim=1, p=float('inf')), p=float('inf')) + denominator_nonzero
    else:
        Linf1_den = torch.tensor(1.0, dtype=torch.float64)
        Linf2_den = torch.tensor(1.0, dtype=torch.float64)

    Linf1_penalty = torch.norm(torch.norm(decoded[:, 0, :] - x_true[:, 0, :], dim=1, p=float('inf')), p=float('inf')) / Linf1_den
    Linf2_penalty = torch.norm(torch.norm(decoded[:, 1, :] - x_true[:, 1, :], dim=1, p=float('inf')), p=float('inf')) / Linf2_den
    loss_Linf = config.Linf_lam * (Linf1_penalty + Linf2_penalty)

    total_loss = loss1 + loss2 + loss3 + loss4 + loss_Linf + loss_var + loss_cov

    return total_loss, loss1, loss2, loss3, loss4, loss_Linf, loss_var, loss_cov

def define_regularization(model, params, unregularized_loss, autoencoder_loss):
    """
    Define the regularization and add to the loss.

    Arguments:
        model -- PyTorch model containing the trainable parameters
        params -- dictionary of parameters for experiment
        unregularized_loss -- the unregularized loss
        autoencoder_loss -- the autoencoder component of the loss

    Returns:
        loss_L1 -- L1 regularization on weights
        loss_L2 -- L2 regularization on weights
        regularized_loss -- loss + regularization
        regularized_autoencoder_loss -- autoencoder loss + regularization
    """
    # Initialize regularization losses
    # loss_L1 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=params.device)
    # loss_L2 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=params.device)

    # Iterate over model parameters
    # for param in model.parameters():
    #     loss_L1 = loss_L1 + params.L1_lam * torch.sum(torch.abs(param))
    #     loss_L2 = loss_L2 + params.L2_lam * torch.sum(param ** 2)
    l1_reg = params.L1_lam * sum(p.abs().sum() for p in model.parameters())
    l2_reg = params.L2_lam * sum((p ** 2).sum() for p in model.parameters())
    loss_L1 = l1_reg
    loss_L2 = l2_reg

    # Combine unregularized loss with regularization
    regularized_loss = unregularized_loss + loss_L1 + loss_L2
    regularized_autoencoder_loss = autoencoder_loss + loss_L1 + loss_L2 # for starting stage if 'auto_first'

    return loss_L1, loss_L2, regularized_loss, regularized_autoencoder_loss

class LossProcess:
    
    def __init__(self, config):
        self.losses = []
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss4 = []
        self.loss_Linf = []
        self.loss_var = []
        self.loss_cov = []
        self.loss_L1 = []
        self.loss_L2 = []
        self.regularized_loss = []
        self.regularized_autoencoder_loss = []
        self.config = config
        
        self.losses_for_save = [[] for _ in range(12)]
        
    def log_loss(self, loss, loss1, loss2, loss3, loss4, loss_Linf, loss_var, loss_cov, loss_L1, loss_L2, step, prefix="train"):
        if prefix == "train":
            wandb.log({f'Loss_{prefix}/Loss': loss, f'Loss_{prefix}/loss1': loss1, f'Loss_{prefix}/loss2': loss2, f'Loss_{prefix}/loss3': loss3, f'Loss_{prefix}/loss4': loss4, f'Loss_{prefix}/loss_Linf': loss_Linf, f'Loss_{prefix}/loss_var': loss_var, f'Loss_{prefix}/loss_cov': loss_cov, f'Loss_{prefix}/loss_L1': loss_L1, f'Loss_{prefix}/loss_L2': loss_L2}, step=step)
        else:
            wandb.log({f'Loss_val/loss': loss, f'Loss_val/loss1': loss1, f'Loss_val/loss2': loss2, f'Loss_val/loss3': loss3, f'Loss_val/loss4': loss4, f'Loss_val/loss_Linf': loss_Linf, f'Loss_val/loss_var': loss_var, f'Loss_val/loss_cov': loss_cov, f'Loss_val/loss_L1': loss_L1, f'Loss_val/loss_L2': loss_L2, 'val_step': step})
        
    def __call__(self, loss, loss1, loss2, loss3, loss4, loss_Linf, loss_var, loss_cov, loss_L1, loss_L2, regularized_loss, regularized_autoencoder_loss, disable_logger=False, step=None, prefix="train", lr=None):
        self.losses.append(loss.item())
        self.loss1.append(loss1.item())
        self.loss2.append(loss2.item())
        self.loss3.append(loss3.item())
        self.loss4.append(loss4.item())
        self.loss_Linf.append(loss_Linf.item())
        self.loss_var.append(loss_var.item())
        self.loss_cov.append(loss_cov.item())
        self.loss_L1.append(loss_L1.item())
        self.loss_L2.append(loss_L2.item())
        self.regularized_loss.append(regularized_loss.item())
        self.regularized_autoencoder_loss.append(regularized_autoencoder_loss.item())
        
        if not disable_logger:
            self.log_loss(loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss_Linf.item(), loss_var.item(), loss_cov.item(), loss_L1.item(), loss_L2.item(), step=step, prefix=prefix)
            if prefix == "train" and lr is not None:
                wandb.log({"lr": lr}, step=step)
                
    def log_to_logger(self, logger, disable_logger, epoch, prefix="train"):
        losses = self.return_loss_list_per_epoch()
        if not disable_logger:
            logger.info(f"Epoch {epoch+1}: {prefix} Loss: {losses[0]:.6f}, Loss1: {losses[1]:.6f}, Loss2: {losses[2]:.6f}, Loss3: {losses[3]:.6f}, Loss4: {losses[4]:.6f}, Loss_Linf: {losses[5]:.6f}, Loss_var: {losses[6]:.6f}, Loss_cov: {losses[7]:.6f}, Loss_L1: {losses[8]:.6f}, Loss_L2: {losses[9]:.6f}, lr: {losses[10]:.6f}")

    
    def return_loss_list_per_epoch(self):
        self.losses_for_save[0].extend(self.losses)
        self.losses_for_save[1].extend(self.loss1)
        self.losses_for_save[2].extend(self.loss2)
        self.losses_for_save[3].extend(self.loss3)
        self.losses_for_save[4].extend(self.loss4)
        self.losses_for_save[5].extend(self.loss_Linf)
        self.losses_for_save[6].extend(self.loss_var)
        self.losses_for_save[7].extend(self.loss_cov)
        self.losses_for_save[8].extend(self.loss_L1)
        self.losses_for_save[9].extend(self.loss_L2)
        self.losses_for_save[10].extend(self.regularized_loss)
        self.losses_for_save[11].extend(self.regularized_autoencoder_loss)
        
        for_return = [np.mean(self.losses), np.mean(self.loss1), np.mean(self.loss2), np.mean(self.loss3), np.mean(self.loss4), np.mean(self.loss_Linf), np.mean(self.loss_var), np.mean(self.loss_cov), np.mean(self.loss_L1), np.mean(self.loss_L2), np.mean(self.regularized_loss), np.mean(self.regularized_autoencoder_loss)]
        
        self.losses.clear()
        self.loss1.clear()
        self.loss2.clear()
        self.loss3.clear()
        self.loss4.clear()
        self.loss_Linf.clear()
        self.loss_L1.clear()
        self.loss_L2.clear()
        self.regularized_loss.clear()
        self.regularized_autoencoder_loss.clear()
        self.loss_var.clear()
        self.loss_cov.clear()

        return for_return
    
    def loss_save(self, train=True):
        losses_np = np.array(self.losses_for_save)
        if train:
            np.save(self.config.log_dir + '/' + 'losses_train.npy', losses_np)
            print('Train losses saved.')
        else:
            np.save(self.config.log_dir + '/' + 'losses_val.npy', losses_np)
            print('Validation losses saved.')
        
