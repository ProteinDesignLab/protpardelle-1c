"""Print the loss curve of a cc152 run straight from the wandb API."""
import sys, wandb
run_id = sys.argv[1] if len(sys.argv) > 1 else "du7xuklf"
api = wandb.Api()
run = api.run(f"tyl/protpardelle-1c/{run_id}")
h = run.history(keys=["train/struct_loss", "train/train_loss", "train/grad_norm"], pandas=True)
if h is None or len(h) == 0:
    print("no history yet"); raise SystemExit
h = h.dropna(subset=["train/struct_loss"])
print(f"{len(h)} logged steps; first/last struct_loss = "
      f"{h['train/struct_loss'].iloc[0]:.4f} / {h['train/struct_loss'].iloc[-1]:.4f}")
n = max(1, len(h)//10)
for i in range(0, len(h), n):
    r = h.iloc[i]
    print(f"  step {int(r['_step']):6d}  struct_loss {r['train/struct_loss']:.4f}  grad_norm {r['train/grad_norm']:.3f}")
r = h.iloc[-1]
print(f"  step {int(r['_step']):6d}  struct_loss {r['train/struct_loss']:.4f}  grad_norm {r['train/grad_norm']:.3f}  <- latest")
