"""CPU tests of the shared runner with a tiny surrogate; not SMIRK/DECA validation."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from facebench.io import read_json, read_jsonl, write_json, write_jsonl

HAS_TORCH=importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH,"Install torch for runner checkpoint/resume tests")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        base=os.environ.get("FACEBENCH_TEST_ROOT")
        if base:Path(base).mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.temp.name)
        self.fail_step=None
        parent=self
        class TinyAdapter:
            def __init__(self,config,training=False):
                self.value=torch.nn.Parameter(torch.tensor([5.]))
                self.optimizer=torch.optim.Adam([self.value],lr=.1)
            def train_step(self,rows,step):
                if parent.fail_step==step:raise RuntimeError("simulated interruption")
                # Exercise Python/NumPy/torch RNG restoration as well as optimizer state.
                import random
                target=(random.random()+np.random.random()+torch.rand(()))/10
                self.optimizer.zero_grad();loss=(self.value-target).square().mean();loss.backward();self.optimizer.step()
                return {"loss":float(loss.detach())}
            def predict(self,rows):
                return {"landmarks2d":self.value.reshape(1,1,1).expand(1,68,2),"landmarks_gt":torch.zeros(1,68,2)}
            def state_dict(self):return {"value":self.value.detach().clone(),"optimizer":self.optimizer.state_dict()}
            def restore(self,state):
                with torch.no_grad():self.value.copy_(state["value"])
                self.optimizer.load_state_dict(state["optimizer"])
        self.adapter=TinyAdapter
        self.train_rows=[{"sample_id":f"train/{i}","dataset_id":"own","subject_id":"train","image_path":str(self.root/f"{i}.png"),"sequence_id":"seq","camera_id":"front","aux_path":str(self.root/"aux.npz")} for i in range(3)]
        self.val_rows=[{**self.train_rows[0],"subject_id":"val","sample_id":"val/0"}]
        write_jsonl(self.root/"train.jsonl",self.train_rows);write_jsonl(self.root/"val.jsonl",self.val_rows)
        (self.root/"initial.pt").write_bytes(b"surrogate initial checkpoint provenance")
    def tearDown(self):self.temp.cleanup()
    def config(self,name):
        cfg={"model":"smirk","source":"own","seed":17,"device":"cpu","upstream_root":str(self.root),"checkpoint":str(self.root/"initial.pt"),
             "train_manifest":str(self.root/"train.jsonl"),"val_manifest":str(self.root/"val.jsonl"),"output_dir":str(self.root/name),
             "max_steps":5,"batch_subjects":1,"images_per_subject":1,"save_every":2,"val_every":2,"max_val_per_subject":2}
        path=self.root/f"{name}.json";write_json(path,cfg);return path
    def run_train(self,path,resume=None):
        from facebench.experiments import train
        with patch("facebench.models.create_adapter",self.adapter),patch("facebench.experiments.check_experiment",return_value=(self.train_rows,self.val_rows,{"commit":"synthetic-test"})),contextlib.redirect_stdout(io.StringIO()):
            return train(path,resume)
    def test_update_checkpoint_and_exact_resume_after_failure(self):
        continuous=self.config("continuous");self.run_train(continuous)
        interrupted=self.config("interrupted");self.fail_step=2
        with self.assertRaises(RuntimeError):self.run_train(interrupted)
        self.assertEqual(read_json(self.root/"interrupted/run.json")["status"],"failed")
        self.fail_step=None;self.run_train(interrupted,self.root/"interrupted/last.pt")
        a=self.torch.load(self.root/"continuous/last.pt",weights_only=False)
        b=self.torch.load(self.root/"interrupted/last.pt",weights_only=False)
        self.torch.testing.assert_close(a["adapter"]["value"],b["adapter"]["value"],rtol=0,atol=0)
        self.assertLess(float(b["adapter"]["value"]),5)
        self.assertEqual(list(read_jsonl(self.root/"continuous/metrics.jsonl")),list(read_jsonl(self.root/"interrupted/metrics.jsonl")))
        self.assertEqual(read_json(self.root/"interrupted/run.json")["completed_steps"],5)
    def test_resume_rejects_changed_budget_or_data(self):
        path=self.config("run");self.run_train(path)
        cfg=read_json(path);cfg["max_steps"]=6;write_json(path,cfg)
        with self.assertRaisesRegex(ValueError,"signature mismatch"):self.run_train(path,self.root/"run/last.pt")
    def test_no_silent_overwrite_of_previous_run(self):
        path=self.config("run");self.run_train(path)
        with self.assertRaises(FileExistsError):self.run_train(path)


if __name__=="__main__":unittest.main()
