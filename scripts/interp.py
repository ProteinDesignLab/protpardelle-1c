from pathlib import Path

import numpy as np
import torch


data_dir = Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/likelihood_cc89_415/states/latents")
# data_dir = Path("/scratch/users/tianyulu/protpardelle-1c-sampling-results/likelihood_cc58_416/states/latents")
x1 = torch.load(data_dir / "pre_cyclic_1QY3_aligned_clean_encoded_latent.pt")
x2 = torch.load(data_dir / "post_cyclic_1QYF_naturalized_reatom_clean_encoded_latent.pt")

for p in torch.linspace(0.0, 1.0, 32):
    xi_linear = p * x1 + (1 - p) * x2
    torch.save(xi_linear, data_dir / f"linear_interp_{p:.2f}.pt".replace('.', '-'))

    xi_spherical = np.sqrt(p) * x1 + np.sqrt(1 - p) * x2
    torch.save(xi_spherical, data_dir / f"spherical_interp_{p:.2f}.pt".replace('.', '-'))
