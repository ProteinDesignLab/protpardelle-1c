# Adapted from https://github.com/h3nnn4n/TMscore.py/tree/master/src

from functools import partial
import re
import os
import os.path
import errno
import subprocess

from multiprocessing.pool import Pool
from pathlib import Path
import pandas as pd
import typer


class TMscore():
    def __init__(self, path=None):
        self.setup_path(path)

        self.rmsd = None
        self.gdt_ts = None
        self.gdt_ts_info = None
        self.gdt_ha = None
        self.gdt_ha_info = None
        self.tm_score = None
        self.maxsub = None

    def setup_path(self, path):
        if path is None:
            path = './TMscore'

        if os.path.isfile(path):
            self.path = os.path.join(os.getcwd(), path)
        else:
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
                path
            )

    def __call__(self, prot_a, prot_b):
        if os.path.isfile(prot_a) and os.path.isfile(prot_b):
            out = subprocess.check_output([self.path, prot_a, prot_b])
            data = str(out).split("\\n")
            for d in data:
                x = re.sub(r"\s\s+", " ", d).split(' ')
                if x[0] == "TM-score" and x[1] == "=":
                    self.tm_score = float(x[2])
                elif x[0] == "GDT-TS-score=":
                    self.gdt_ts = float(x[1])
                    a = float(x[2].split('=')[1])
                    b = float(x[3].split('=')[1])
                    c = float(x[4].split('=')[1])
                    d = float(x[5].split('=')[1])
                    self.gdt_ts_info = (a, b, c, d)
                elif x[0] == "GDT-HA-score=":
                    self.gdt_ha = float(x[1])
                    a = float(x[2].split('=')[1])
                    b = float(x[3].split('=')[1])
                    c = float(x[4].split('=')[1])
                    d = float(x[5].split('=')[1])
                    self.gdt_ha_info = (a, b, c, d)
                elif x[0] == "RMSD":
                    self.rmsd = float(x[5])
                elif x[0] == "MaxSub-score=":
                    self.maxsub = float(x[1])
        else:
            raise Exception("Check that %s and %s exists" % (prot_a, prot_b))

    def print_info(self):
        print(self.rmsd)
        print(self.tm_score)
        print(self.maxsub)
        print(self.get_gdt_ts_info())
        print(self.get_gdt_ha_info())

    def get_rmsd(self):
        return self.rmsd

    def get_gdt_ts(self):
        return self.gdt_ts

    def get_gdt_ts_info(self):
        return (self.gdt_ts, self.gdt_ts_info)

    def get_gdt_ha(self):
        return self.gdt_ha

    def get_gdt_ha_info(self):
        return (self.gdt_ha, self.gdt_ha_info)

    def get_tm_score(self):
        return self.tm_score

    def get_maxsub(self):
        return self.maxsub

    def get_all(self):
        a = {}
        a['rmsd'] = self.rmsd
        a['gdt_ts'] = self.get_gdt_ts_info()
        a['gdt_ha'] = self.get_gdt_ha_info()
        a['maxsub'] = self.maxsub
        a['tm_score'] = self.tm_score

        return a
    

def pdb_to_tm(fp, pdb_keys: Path = Path("eval_split_aa"), pred_dir: Path = Path("."), gt_dir: Path = Path("pdb_store")):
    pdb_stem = fp.split("_")[0]
    gt_fp = gt_dir / pdb_keys / f"{pdb_stem}_reres.pdb"
    pred_fp = pred_dir / f"{fp}.pdb"
    tmscore = TMscore("./TMscore")
    try:
        tmscore(gt_fp, pred_fp)
    except Exception as err:
        print(err)
        print(f"Error on {fp}")
        pass
    return fp, tmscore.rmsd, tmscore.tm_score, tmscore.maxsub, tmscore.gdt_ts, tmscore.gdt_ha


def tm_runner(pdb_keys: Path, pred_dir: Path, gt_dir: Path):
    pdb_fps = []
    with open(pdb_keys, "r") as fp:
        for line in fp.readlines():
            for i in range(32):
                pdb_fps.append(f"{line.strip()}_{i}_reres")

    with Pool(8) as pool:
        ret = pool.map(partial(pdb_to_tm, pdb_keys=pdb_keys, pred_dir=pred_dir, gt_dir=gt_dir), pdb_fps)

    df = pd.DataFrame()
    df["pdb_path"] = [r[0].split("/")[-1] for r in ret]
    df["tm_rmsd"] = [r[1] for r in ret]
    df["tm_score"] = [r[2] for r in ret]
    df["tm_maxsub"] = [r[3] for r in ret]
    df["tm_gdt_ts"] = [r[4] for r in ret]
    df["tm_gdt_ha"] = [r[5] for r in ret]
    save_fp = f"{pdb_keys}_tm.csv"
    df.to_csv(save_fp, index=False)


if __name__ == "__main__":
    typer.run(tm_runner)
