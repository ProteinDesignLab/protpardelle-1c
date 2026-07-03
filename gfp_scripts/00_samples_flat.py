from pathlib import Path
import shutil
from tqdm import tqdm


sample_dir = Path(f"/scratch/users/tianyulu/protpardelle-1c-sampling-results/02052026-de_novo_gfp")
sample_flat_dir = Path(f"/scratch/users/tianyulu/protpardelle-1c-sampling-results/02052026-de_novo_gfp_samples_flat")

for fp in tqdm(sample_dir.rglob("*.pdb")):
    save_name = f"{fp.parent.parent.name}-{fp.parent.name}-{fp.stem}"
    save_dir = sample_flat_dir / fp.parent.name
    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fp, save_dir / f"{save_name}.pdb")
