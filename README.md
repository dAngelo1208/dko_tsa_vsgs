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
