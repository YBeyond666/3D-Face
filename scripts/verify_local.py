"""Run the portable checks and save an inspectable report; no model assets needed."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import unittest

p=argparse.ArgumentParser()
p.add_argument("--out",required=True)
a=p.parse_args()
out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ["FACEBENCH_TEST_ROOT"]=str(out/"test-tmp")
project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project/"src"))
suite=unittest.TestLoader().discover(str(project/"tests"))
with (out/"unittest.log").open("w",encoding="utf-8") as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
report={"python":sys.version,"platform":platform.platform(),"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "skipped":[{"test":str(t),"reason":reason} for t,reason in result.skipped],"ok":result.wasSuccessful(),
        "optional_packages":{m:bool(importlib.util.find_spec(m)) for m in ("torch","cv2")},
        "scope":"core data/geometry/CLI plus optional MP4/CPU surrogate checkpoint tests; no real SMIRK/DECA GPU execution"}
(out/"verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
try:(out/"test-tmp").rmdir()
except OSError:pass
print(json.dumps(report,ensure_ascii=False,indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
