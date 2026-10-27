"""Run generated jobs sequentially. Stop on the first failure; never overlap GPU jobs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from facebench.io import read_json, resolve

p=argparse.ArgumentParser()
p.add_argument("plan")
p.add_argument("--execute",action="store_true",help="Without this flag, print commands only")
p.add_argument("--model",choices=["smirk","deca"])
p.add_argument("--python-smirk")
p.add_argument("--python-deca")
a=p.parse_args()
path=Path(a.plan).resolve()
for name in read_json(path)["training_runs"]:
    config=str(resolve(path.parent,name))
    model=read_json(config)["model"]
    if a.model and model!=a.model:continue
    custom=getattr(a,"python_"+model)
    project=Path(__file__).resolve().parents[1]
    interpreter=Path(custom) if custom else project/f".venv-{model}"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    if a.execute and not interpreter.is_file():raise SystemExit(f"Model environment missing: {interpreter}")
    command=[str(interpreter),"-m","facebench","train","--config",config]
    print(subprocess.list2cmdline(command),flush=True)
    if a.execute:subprocess.run(command,check=True)
