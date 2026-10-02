"""Is cc152 actually using the atomized channels, or ignoring them?

Zero out each new channel group in turn and see how the RMSD of the extra-block
atoms degrades. If zeroing the reference conformer hurts, the trunk is reading
the CCD topology; if zeroing the element one-hot hurts, it is reading chemistry.
"""
import argparse, numpy as np, torch, yaml
from protpardelle.core import models
from protpardelle.data import atomize_io as A
from protpardelle.data.dataset import make_crop_cond_mask_and_recenter_coords
from protpardelle.utils import dict_to_namespace, expand_env_in_config
from protpardelle.common import residue_constants as rc

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--sigma", type=float, default=8.0)
ap.add_argument("--sigma-data", type=float, default=13.61)
a = ap.parse_args()

d = expand_env_in_config(yaml.safe_load(open("examples/training/cc152.yaml")))
d["data"]["sigma_data"] = a.sigma_data; d["data"]["auto_calc_sigma_data"] = False
d["train"]["ckpt_path"] = ""
m = models.Protpardelle(dict_to_namespace(d), device="cuda").eval()
c = torch.load(a.ckpt, weights_only=False, map_location="cuda")
m.load_state_dict(c["model_state_dict"] if isinstance(c, dict) else c.state_dict())

ccd = A.maybe_load_default_ccd()
mirror = "/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF"
keys = sorted(set(k.strip() for k in open(
    "/scratch/users/tianyulu/datasets/pdb_atomized/val_atomized_pdb_keys.list") if k.strip()))

ABL = ["none", "no_ref", "no_elem", "no_flag", "all_off"]
acc = {k: {"ncaa": [0.0, 0], "het": [0.0, 0], "bb": [0.0, 0]} for k in ABL}
used = 0
for key in keys:
    if used >= a.n: break
    try:
        f = A.load_feats_from_cif_atomized(f"{mirror}/{key[1:3]}/{key}.cif.gz", ccd,
                                           max_tokens=384, rng=np.random.default_rng(0))
    except Exception:
        continue
    if f is None: continue
    am = f["atom_mask"] > 0
    nc = f["atom_noncanon"] > 0
    het = (f["aatype"] == A.UNK_AATYPE).unsqueeze(-1) & am
    bbm = torch.zeros_like(am); bbm[:, [rc.atom_order[x] for x in ("N","CA","C","O")]] = True
    cls = {"ncaa": am & nc & ~het, "het": het, "bb": am & bbm & ~nc & ~het}
    if cls["ncaa"].sum() == 0: continue
    used += 1
    coords = f["atom_positions"].unsqueeze(0).cuda(); amk = f["atom_mask"].unsqueeze(0).cuda()
    coords, ccm, _ = make_crop_cond_mask_and_recenter_coords(
        amk, coords, aatype=f["aatype"].unsqueeze(0).cuda(),
        chain_index=f["chain_index"].unsqueeze(0).cuda(),
        contiguous_prob=0.05, discontiguous_prob=0.9, sidechain_prob=0.9,
        sidechain_only_prob=0.0, max_span_len=12, max_discontiguous_res=24,
        dist_threshold=45.0, recenter_coords=True)
    n = coords.shape[1]
    g = torch.Generator().manual_seed(used)
    noisy = (coords + torch.randn(coords.shape, generator=g).cuda() * a.sigma) * amk[..., None]
    el = f["atom_elem"].unsqueeze(0).cuda(); rf = f["atom_ref"].unsqueeze(0).cuda()
    fl = f["atom_noncanon"].unsqueeze(0).cuda()
    for mode in ABL:
        e, r, fg = el, rf, fl
        if mode in ("no_ref", "all_off"): r = torch.zeros_like(rf)
        if mode in ("no_elem", "all_off"): e = torch.zeros_like(el)
        if mode in ("no_flag", "all_off"): fg = torch.zeros_like(fl)
        with torch.no_grad():
            pred = m(noisy_coords=noisy, noise_level=torch.full((1, n), a.sigma, device="cuda"),
                     seq_mask=torch.ones(1, n, device="cuda"),
                     residue_index=f["residue_index"].unsqueeze(0).cuda().long(),
                     chain_index=f["chain_index"].unsqueeze(0).cuda().long(),
                     struct_crop_cond=coords * ccm[..., None], crop_cond_mask=ccm,
                     atom_elem=e, atom_ref=r, atom_noncanon=fg, run_mpnn_model=False)[0]
        sq = ((pred[0].cpu() - coords[0].cpu()) ** 2).sum(-1)
        for cn, msk in cls.items():
            acc[mode][cn][0] += float(sq[msk].sum()); acc[mode][cn][1] += int(msk.sum())

print(f"\n{used} structures, sigma={a.sigma}, ckpt={a.ckpt.split('/')[-1]}")
print(f"{'ablation':>10} {'backbone':>10} {'NCAA extra':>11} {'ligand/metal':>13}")
print("-" * 48)
for mode in ABL:
    row = [(acc[mode][c][0] / max(acc[mode][c][1], 1)) ** 0.5 for c in ("bb", "ncaa", "het")]
    print(f"{mode:>10} {row[0]:10.3f} {row[1]:11.3f} {row[2]:13.3f}")
