"""Stream-extract pinder pdbs.zip into a sharded directory layout.

The flat ~1.56M-file `pdbs/` layout used by genie3 hits Lustre per-directory
limits on the Sherlock /scratch filesystem ("Disk quota exceeded" while
extracting, even with plenty of overall block/inode quota free). We instead
write each PDB to:

    <out-dir>/<shard>/<id>.pdb

where `<shard>` is the first two characters of `<id>` (lowercased). Pinder
IDs start with a 4-char PDB code (e.g. `8phr__X4_..._--_...`), so two-char
prefixes give us ~hundreds of subdirs with a few thousand files each.

Only IDs present in the genie3 info.csv are extracted; everything else in the
zip is skipped so we don't waste inodes on dimers the dataset won't see.
The dataset.py pinder_2024-02 branch already knows about this layout.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from pathlib import Path


def _shard(pid: str) -> str:
    # Lowercase + first two chars. ID guaranteed >= 2 chars in practice
    # (PDB codes are 4 chars). Fallback for ultra-short IDs: padded "_".
    return (pid[:2].lower() or "__").ljust(2, "_")


def load_wanted_ids(info_csv: Path) -> set[str]:
    wanted: set[str] = set()
    with info_csv.open() as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row.get("id")
            if pid:
                wanted.add(pid)
    return wanted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-path", required=True, type=Path)
    ap.add_argument("--info-csv", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Sharded output dir. <out-dir>/<2char-prefix>/<id>.pdb",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="If > 0, stop after extracting this many files (debug).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading wanted ids from {args.info_csv} ...", flush=True)
    wanted = load_wanted_ids(args.info_csv)
    print(f"wanted ids: {len(wanted)}", flush=True)

    print(f"opening {args.zip_path} ...", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_missing_dir = 0
    with zipfile.ZipFile(args.zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Zip entries look like "pdbs/<id>.pdb". Strip the "pdbs/" prefix
            # and the ".pdb" suffix to recover the bare id.
            name = info.filename
            base = Path(name).name  # <id>.pdb
            if not base.endswith(".pdb"):
                continue
            pid = base[:-4]
            if pid not in wanted:
                n_skip += 1
                continue

            shard_dir = args.out_dir / _shard(pid)
            target = shard_dir / base
            if target.exists() and target.stat().st_size > 0:
                n_ok += 1
                continue

            try:
                shard_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                n_missing_dir += 1
                print(f"  mkdir failed for {shard_dir}: {e}", flush=True)
                continue

            # Stream contents to target. Use a tmp file + rename for atomicity.
            tmp = shard_dir / f".{base}.tmp"
            try:
                with zf.open(info) as src, tmp.open("wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        dst.write(buf)
                tmp.rename(target)
                n_ok += 1
            except OSError as e:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
                print(f"  write failed for {target}: {e}", flush=True)

            if n_ok and n_ok % 5000 == 0:
                elapsed = time.time() - t0
                rate = n_ok / elapsed if elapsed > 0 else 0
                print(
                    f"  ok={n_ok} skipped={n_skip} elapsed={elapsed:.1f}s "
                    f"rate={rate:.1f}/s",
                    flush=True,
                )
            if args.limit and n_ok >= args.limit:
                break

    elapsed = time.time() - t0
    print(
        f"done in {elapsed:.1f}s. ok={n_ok} skipped(not-in-wanted)={n_skip} "
        f"missing_dir={n_missing_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
