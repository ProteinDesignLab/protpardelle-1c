"""
Given a folder of PDB files, possibly nested, compute their SASA with Biopython
"""

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
from pathlib import Path

from Bio.PDB import PDBParser, ShrakeRupley
import pandas as pd
from tqdm import tqdm


def compute_sasa(pdb_path: str, residue_index: list[int]) -> float:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)

    sr = ShrakeRupley()

    all_sasa = []
    
    sr.compute(structure, level="R")  # compute SASA per residue

    for ri in residue_index:
        all_sasa.append(structure[0]['A'][ri].sasa)
        
    all_sasa = [str(f"{sasa:.2f}") for sasa in all_sasa]

    return ",".join(all_sasa)


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
    

def compute_rmsd_tm(pdb_path):
    reference_pdb_path = "/home/users/tianyulu/code/allatom-design/test_data/protpardelle-1c/states/pre_cyclic_1QY3_aligned_clean_reres.pdb"
    tmscore = TMscore("./TMscore")
    tmscore(reference_pdb_path, pdb_path)
    return ",".join([f"{tmscore.rmsd:.2f}", f"{tmscore.tm_score:.2f}"])


def main(design_output_csv_path: Path):
    df = pd.read_csv(design_output_csv_path)

    results = []
    for pdb_path, residue_index in tqdm(zip(df["repack_pdb_path"], df["fix_pos"])):
        residue_index = [int(ri) for ri in residue_index.split()]
        sasa = compute_sasa(pdb_path, residue_index)
        rmsd_tm = compute_rmsd_tm(pdb_path)
        results.append(",".join([pdb_path, sasa, rmsd_tm]))

    # Save results to CSV
    out_file = design_output_csv_path.parent / "sasa_and_tm_results.csv"
    motif_header = list(range(58, 72)) + [96] + [222]
    motif_header = ','.join([str(mh) for mh in motif_header] + ['tm_rmsd', 'tm_score'])
    with open(out_file, "w") as f:
        f.write(f"pdb_path,{motif_header}\n")
        for line in results:
            f.write(f"{line}\n")
    typer.echo(f"\nResults written to {out_file}")


if __name__ == "__main__":
    typer.run(main)

