"""Did the new atomized input channels / extra atom slots actually get trained?

Compares a checkpoint against the cc89-derived init. If the new channel columns
are still zero, they never received gradient (a wiring bug); if they moved but
the ablation shows no effect, the model has genuinely learned to ignore them.
"""
import argparse, torch

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="/scratch/users/tianyulu/protpardelle-1c-experiments/cc152_init/cc152_from_cc89_epoch415.pth")
ap.add_argument("--ckpt", required=True)
a = ap.parse_args()

def sd(p):
    c = torch.load(p, weights_only=False, map_location="cpu")
    return c["model_state_dict"] if isinstance(c, dict) else c.state_dict()

W = "struct_model.net.to_patch_embedding.1.weight"
O = "struct_model.net.from_patch.1.weight"
i, e = sd(a.init), sd(a.ckpt)
W0, W1 = i[W].view(256, 22, 61), e[W].view(256, 22, 61)

groups = [(0, 3, "noisy xyz"), (3, 6, "self-cond xyz"), (6, 9, "crop-cond xyz"),
          (9, 18, "element one-hot"), (18, 21, "ref conformer"), (21, 22, "extra-slot flag")]
print(f"{'input channel group':>20} {'|W| init':>10} {'|W| now':>10} {'|dW|':>10}")
print("-" * 54)
for lo, hi, lbl in groups:
    print(f"{lbl:>20} {W0[:,lo:hi].norm():10.4f} {W1[:,lo:hi].norm():10.4f} {(W1-W0)[:,lo:hi].norm():10.4f}")

print(f"\n{'atom slots':>20} {'|W| init':>10} {'|W| now':>10} {'|dW|':>10}")
print("-" * 54)
for lo, hi, lbl in [(0, 37, "canonical 0-36"), (37, 61, "extra 37-60")]:
    print(f"{lbl:>20} {W0[:,:,lo:hi].norm():10.4f} {W1[:,:,lo:hi].norm():10.4f} {(W1-W0)[:,:,lo:hi].norm():10.4f}")

O0, O1 = i[O].view(3, 61, 256), e[O].view(3, 61, 256)
print(f"\noutput rows  canonical |W|={O1[:,:37].norm():.4f} |dW|={(O1-O0)[:,:37].norm():.4f}"
      f"   extra |W|={O1[:,37:].norm():.4f} |dW|={(O1-O0)[:,37:].norm():.4f}")
