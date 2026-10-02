"""Can a ligand token influence the protein prediction at all?

Distinguishes "the model has not learned to use ligands" from "ligand tokens
are not reaching the protein tokens" (an attention/plumbing bug). Translates
the ligand 50 A away and measures how much the protein prediction moves. If the
answer is exactly zero, information is not flowing.
"""
import argparse, numpy as np, torch, yaml
from protpardelle.core import models
from protpardelle.data import atomize_io as A
from protpardelle.data.dataset import make_crop_cond_mask_and_recenter_coords
from protpardelle.utils import dict_to_namespace, expand_env_in_config

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", default="examples/training/cc152.yaml")
ap.add_argument("--sigma-data", type=float, default=13.23)
ap.add_argument("--sigma", type=float, default=8.0)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()

d = expand_env_in_config(yaml.safe_load(open(a.config)))
d["data"]["sigma_data"] = a.sigma_data; d["data"]["auto_calc_sigma_data"] = False
d["train"]["ckpt_path"] = ""
m = models.Protpardelle(dict_to_namespace(d), device="cuda").eval()
c = torch.load(a.ckpt, weights_only=False, map_location="cuda")
m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c.state_dict())

ccd = A.maybe_load_default_ccd()
mirror = "/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF"
keys = sorted(set(k.strip() for k in open(
    "/scratch/users/tianyulu/datasets/pdb_atomized/val_atomized_pdb_keys.list") if k.strip()))

def fwd(feats, coords, ccm, noise, lig_atom, shift):
    n = coords.shape[1]
    am = feats["atom_mask"].unsqueeze(0).cuda()
    cc = coords.clone()
    if shift != 0.0:
        cc = torch.where(lig_atom.unsqueeze(-1), cc + shift, cc)
    noisy = torch.where(lig_atom.unsqueeze(-1), cc, (cc + noise)) * am[..., None]
    scc = cc * ccm[..., None]
    with torch.no_grad():
        return m(noisy_coords=noisy, noise_level=torch.full((1, n), a.sigma, device="cuda"),
                 seq_mask=torch.ones(1, n, device="cuda"),
                 residue_index=feats["residue_index"].unsqueeze(0).cuda().long(),
                 chain_index=feats["chain_index"].unsqueeze(0).cuda().long(),
                 struct_crop_cond=scc, crop_cond_mask=ccm,
                 atom_elem=feats["atom_elem"].unsqueeze(0).cuda(),
                 atom_ref=feats["atom_ref"].unsqueeze(0).cuda(),
                 atom_noncanon=feats["atom_noncanon"].unsqueeze(0).cuda(),
                 run_mpnn_model=False)[0]

used = 0
print(f"{'pdb':>6} {'pocket shift (A)':>17} {'far-protein shift (A)':>22}")
print("-" * 48)
tot_p = tot_f = 0.0; np_ = nf = 0
for k in keys:
    if used >= a.n: break
    import os
    p = f"{mirror}/{k[1:3]}/{k}.cif.gz"
    if not os.path.exists(p): continue
    f = A.load_feats_from_cif_atomized(p, ccd, max_tokens=384, rng=np.random.default_rng(0))
    if f is None or f["is_ligand"].sum() == 0: continue
    used += 1
    coords = f["atom_positions"].unsqueeze(0).cuda(); am = f["atom_mask"].unsqueeze(0).cuda()
    np.random.seed(used); torch.manual_seed(used)
    coords, ccm, _ = make_crop_cond_mask_and_recenter_coords(
        am, coords, aatype=f["aatype"].unsqueeze(0).cuda(),
        chain_index=f["chain_index"].unsqueeze(0).cuda(),
        contiguous_prob=0.05, discontiguous_prob=0.9, sidechain_prob=0.9,
        sidechain_only_prob=0.0, max_span_len=12, max_discontiguous_res=24,
        dist_threshold=45.0, recenter_coords=True)
    lig = ((f["is_ligand"].unsqueeze(-1).cuda() > 0) & (f["atom_mask"].cuda() > 0)).unsqueeze(0)
    g = torch.Generator().manual_seed(used)
    noise = torch.randn(coords.shape, generator=g).cuda() * a.sigma
    p0 = fwd(f, coords, ccm, noise, lig, 0.0)
    p1 = fwd(f, coords, ccm, noise, lig, 50.0)
    delta = (p0 - p1).norm(dim=-1)[0].cpu()
    prot = (f["atom_mask"] > 0) & (f["is_ligand"].unsqueeze(-1) <= 0)
    lx = f["atom_positions"][(f["is_ligand"].unsqueeze(-1) > 0) & (f["atom_mask"] > 0)]
    dmin = torch.cdist(f["atom_positions"].reshape(-1, 3), lx).min(-1).values.reshape(prot.shape)
    near = prot & (dmin <= 6.0); far = prot & (dmin > 15.0)
    dn = float(delta[near].mean()) if near.any() else float("nan")
    df = float(delta[far].mean()) if far.any() else float("nan")
    print(f"{k:>6} {dn:17.4f} {df:22.4f}")
    if near.any(): tot_p += float(delta[near].sum()); np_ += int(near.sum())
    if far.any(): tot_f += float(delta[far].sum()); nf += int(far.sum())
print("-" * 48)
print(f"{'MEAN':>6} {tot_p/max(np_,1):17.4f} {tot_f/max(nf,1):22.4f}")
print("\n(how far the protein prediction moves when the ligand is translated 50 A)")
