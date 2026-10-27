"""Delegate mini download to the official Multiface script; no archive parser copied."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument("--repository",required=True)
p.add_argument("--destination",required=True)
a=p.parse_args()
root=Path(a.repository).resolve()
script=root/"download_dataset.py"
config=root/"mini_download_config.json"
if not script.is_file() or not config.is_file():
    raise SystemExit("Clone https://github.com/facebookresearch/multiface first; install its downloader requirements separately.")
subprocess.run([sys.executable,str(script),"--dest",str(Path(a.destination).resolve()),"--download_config",str(config)],cwd=root,check=True)
