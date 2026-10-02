"""Does the eval harness reproduce cc89's known-good behaviour on canonical protein?

If cc89 cannot denoise a plain single-chain protein here, the harness is wrong.
"""
import numpy as np, torch, yaml, sys
from protpardelle.core import models
from protpardelle.data import atomize_io as A
from protpardelle.data.dataset import make_crop_cond_mask_and_recenter_coords
from protpardelle.utils import dict_to_namespace, expand_env_in_config
from protpardelle.common import residue_constants as rc

def build(cfg_path, ckpt, sigma):
    d = expand_env_in_config(yaml.safe_load(open(cfg_path)))
    d["data"]["sigma_data"]=sigma; d["data"]["auto_calc_sigma_data"]=False; d["train"]["ckpt_path"]=""
    m = models.Protpardelle(dict_to_namespace(d), device="cuda").eval()
    c = torch.load(ckpt, weights_only=False, map_location="cuda")
    m.load_state_dict(c["model_state_dict"] if isinstance(c,dict) else c.state_dict())
    return m

ccd = A.maybe_load_default_ccd()
mirror="/scratch/users/tianyulu/datasets/pdb_mirror/mmCIF"
keys=[k.strip() for k in open("/scratch/users/tianyulu/datasets/pdb_atomized/val_atomized_pdb_keys.list")]
# find plain single-chain, no hetero, canonical-only examples
picks=[]
for k in sorted(set(keys)):
    f=A.load_feats_from_cif_atomized(f"{mirror}/{k[1:3]}/{k}.cif.gz", ccd, max_tokens=256,
                                     include_hetero=False, rng=np.random.default_rng(0))
    if f is None: continue
    if f["n_modified"]>0: continue
    if int(f["chain_index"].max())!=0: continue
    if f["atom_mask"][:, A.N_ATOM37:].sum()>0: continue
    picks.append((k,f))
    if len(picks)>=6: break
print("plain single-chain canonical examples:", [p[0] for p in picks])

m89 = build("examples/training/cc152.yaml",
            "/scratch/users/tianyulu/protpardelle-1c-experiments/cc152_init/cc152_from_cc89_epoch415.pth", 10.01)
BB=[rc.atom_order[a] for a in ("N","CA","C","O")]
for sigma in [0.5,1.0,4.0]:
  for mode in ["cropcond","uncond"]:
    se_m=se_n=cnt=0
    for si,(k,f) in enumerate(picks):
        coords=f["atom_positions"].unsqueeze(0).cuda(); am=f["atom_mask"].unsqueeze(0).cuda()
        n=coords.shape[1]; sm=torch.ones(1,n,device="cuda")
        ccm=None; scc=None
        if mode=="cropcond":
            coords,ccm,_=make_crop_cond_mask_and_recenter_coords(am,coords,
                aatype=f["aatype"].unsqueeze(0).cuda(), chain_index=f["chain_index"].unsqueeze(0).cuda(),
                contiguous_prob=0.05,discontiguous_prob=0.9,sidechain_prob=0.9,sidechain_only_prob=0.0,
                max_span_len=12,max_discontiguous_res=24,dist_threshold=45.0,recenter_coords=True)
            scc=coords*ccm[...,None]
        else:
            c0=(coords*am[...,None]).sum((1,2))/am.sum((1,2))[:,None]
            coords=(coords-c0[:,None,None])*am[...,None]
        g=torch.Generator().manual_seed(si)
        noisy=(coords+torch.randn(coords.shape,generator=g).cuda()*sigma)*am[...,None]
        with torch.no_grad():
            pred=m89(noisy_coords=noisy, noise_level=torch.full((1,n),sigma,device="cuda"),
                     seq_mask=sm, residue_index=f["residue_index"].unsqueeze(0).cuda().long(),
                     chain_index=f["chain_index"].unsqueeze(0).cuda().long(),
                     struct_crop_cond=scc, crop_cond_mask=ccm,
                     atom_elem=f["atom_elem"].unsqueeze(0).cuda(),
                     atom_ref=f["atom_ref"].unsqueeze(0).cuda(),
                     atom_noncanon=f["atom_noncanon"].unsqueeze(0).cuda(),
                     run_mpnn_model=False)[0]
        msk=(am[0]>0)
        bbm=torch.zeros_like(msk); bbm[:,BB]=True; msk=msk&bbm
        se_m+=float(((pred[0]-coords[0])**2).sum(-1)[msk].sum())
        se_n+=float(((noisy[0]-coords[0])**2).sum(-1)[msk].sum())
        cnt+=int(msk.sum())
    print(f"sigma={sigma:4} {mode:9} bb RMSD: cc89={ (se_m/cnt)**0.5 :6.3f}  noisy={ (se_n/cnt)**0.5 :6.3f}")
