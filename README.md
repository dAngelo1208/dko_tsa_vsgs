# Source code of paper <*> which is currently under review.
This repository contains the reference implementation for the paper **“*”**, where we combine DeepKoopman latent linearization with a Koopman Neural Lyapunov (KNV\_\theta) network to obtain certifiable regions of attraction (ROA) for VSGs. The code base exposes the networks, datasets, and inference utilities that were used to create the figures and analyses in the manuscript.

## Status Overview

| Status | Item | Notes |
| --- | --- | --- |
| ✅ Done | DeepKoopman and FiLM networks | `modules/network.py` provides the baseline KoopmanNet as well as FiLM-conditioned variants that cover the architectures evaluated in the paper. |
| ✅ Done | Koopman Neural Lyapunov (KNV\_\theta) network | `modules/KNV_function_net.py` expose the Koopman neural Lyapunov function used for certification. |
| ✅ Done | Demo Koopman + KNV checkpoints | Pretrained Koopman and KNV weights will be shared as `logs/{NAME}_log/checkpoint_ko.pth` and `logs/{NAME}_log/LXNTrain_bnk/checkpoint_knv.pth` in the upcoming release. |
| ✅ Done | KNV vs. Koopman-only loss curves | Serialized arrays at `logs/lyap_loss_record_npy/{knv_losses.npy, koopman_only_losses.npy}` demonstrate efficiency of the proposed KNV function. |
| ⏳ TODO | Demo VSG datasets | Two released `.npz` bundles live under `data_fnl_exp/{NAME}.npz` (`NAME = VSG_demo_p1/p2`) with 100 convergent trajectories each. |
| ⏳ TODO | Publish a polished Jupyter demonstration | The public demo notebook will ship in a future drop. |
| ⏳ TODO | Release the full DeepKoopman training workflow | The internal trainer for KoopmanNet pretraining still needs to be cleaned and documented before it can be published. |

## Repository Highlights

- `modules/network.py` – KoopmanNet and FiLM-conditioned Koopman operator definitions ready for inference.
- `modules/KNV_function_net.py` – KNV\_\theta Lyapunov head plus its supporting losses.
- `utils/` – Configuration (`utils/config.py`), data loading (`utils/data_process.py`), logging, and bookkeeping helpers.
- `data_fnl_exp/` and `logs/` – Default locations where demo datasets and pretrained checkpoints should be placed.

## Released Research Assets

| Asset | Path | Description |
| --- | --- | --- |
| Demo datasets | `data_fnl_exp/{NAME}.npz` (`NAME = VSG_demo_p1/VSG_demo_p2`) | Single-machine VSG trajectories under two parameter settings, 100 convergent simulations each, resampled with a window stride for the demo notebook. |
| Koopman checkpoints | `logs/{NAME}_log/checkpoint_ko.pth` | Frozen KoopmanNet latent linearizations used in the released visualizations. |
| KNV\_\theta checkpoints | `logs/{NAME}_log/LXNTrain_bnk/checkpoint_knv.pth` | Koopman Neural Lyapunov certificates paired with each Koopman backbone. |
| Loss comparison arrays | `logs/lyap_loss_record_npy/knv_losses.npy` and `koopman_only_losses.npy` | Averaged training curves for KNV\_\theta vs. Koopman-only variants. |

<!-- Demo data archive: https://drive.google.com/file/d/1yh-o8m0xoHyYsTX_ww029TGRHEveVcmK/view?usp=share_link   -->
Checkpoints archive: https://drive.google.com/file/d/1aloZPSK0PUHtisfVy9SI0-JSPIZjpJ0j/view?usp=share_link

Demo data archive: Coming soon.

Instructions:
1. Download both archives from the links above.
2. Extract the demo dataset zip and drop the `.npz` files under `data_fnl_exp/`.
3. Extract the checkpoints zip and place the folders under `logs/` (each NAME gets its `{NAME}_log` tree).

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt  # PyTorch 2.4.0 + NumPy, SciPy, Matplotlib, WandB/SwanLab
```

GPU support follows the standard PyTorch installation matrix; adjust the `torch` wheel if you need a specific CUDA build.


## NEW: Parameter Guideline (Reproducibility Notes)

This repository exposes the reference implementation used in the manuscript. Below we summarize the practical hyperparameter choices that were used in the case studies, together with simple tuning guidance. The goal is to make the settings easy to reproduce and to clarify which knobs affect which tradeoffs.

### SDML (Spectral Damping Margin Loss)

SDML is implemented via the log-sum-exp (LSE) ``soft maximum''
\(\widetilde\alpha_{\tau}(\mathcal S_{\theta})\) followed by a one-sided hinge penalty. It encourages the learned latent generator spectrum to satisfy a damping reserve (margin) while remaining a *soft* regularizer that trades off against rollout and one-step linearity fitting.

**Key knobs**

- **Margin magnitude** `margin` (corresponds to \(\mu_m\) in the paper): we use a modest range **[-2, 0]** in p.u. units, with a default value **-1** in the case studies. Start from this interval first.
- **Temperature** `tau`: controls how sharply the LSE soft maximum focuses on the least-damped (most violating) mode. We schedule `tau` by **cosine annealing from 1.0 → 0.1**.
- **Loss weight** `delta`: multiplies \(\mathcal L_{\mathrm{SDML}}\) in the total loss. Keep `delta` **mild** so that rollout and Koopman-linearity remain dominant; overly large values can improve the margin objective but typically increase rollout error and the Koopman linear residual.

**Practical recipe**

1. Use `margin = 1.0`, `tau` cosine-annealed **1.0 → 0.1**, and a mild `delta`.
2. If the learned spectrum shows persistent margin violations (hinge frequently active), slightly increase `delta` *or* increase `margin` within **[-2, 0]**.
3. If rollout RMSE or the Koopman linear residual increases noticeably, reduce `delta` first (this is usually the most sensitive knob).

### FiLM conditioning

<!-- - **Anchor count** `N`: increasing anchors generally improves interpolation quality, but the marginal gain saturates beyond a moderate `N` (see the anchor-count curve in the manuscript). In our ablations we evaluated `N ∈ {4, 8, 16, 32, 40, 64, 80}`. -->
- **Admissible parameter domain** \(\mathcal P\): in the case studies, parameters are sampled as bounded variations around a nominal setting \(p_0\) adopted from Shuai et al. (TSG 2019). We report the variation levels in the manuscript; the code uses the same domain consistently for data generation, training, and evaluation.

### Verification (dReal tail-check)

- SMT runtime is sensitive to the **system dimension** and to the geometric complexity of the validated domain \(\Omega\) (e.g., the number of facets in a convex-hull representation). We report mean and worst-case SMT times under fixed verification settings in the manuscript.
- If verification becomes slow on larger cases, consider reducing \(\Omega\) complexity (e.g., fewer hull points / facet reduction) or verifying localized certificates.

> Note: the exact parameter names in configs may differ by experiment script. Search for `margin`, `tau`, `delta`, and `sdml` in `utils/config.py` and the corresponding run scripts to locate these settings.


## Loss-Curve Comparison

Plot the released arrays directly to recreate the KNV vs. Koopman-only panel:

```python
import numpy as np
import matplotlib.pyplot as plt

knv = np.load('logs/lyap_loss_record_npy/knv_losses.npy')
kponly = np.load('logs/lyap_loss_record_npy/koopman_only_losses.npy')
plt.plot(knv, label='KNV_θ', linewidth=2)
plt.plot(kponly, label='Koopman-only', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
```

Issues and pull requests are welcome once the remaining TODO items are released. 
