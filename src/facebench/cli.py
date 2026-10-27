from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .io import load_manifest, read_json, write_json, write_jsonl


def main(argv=None):
    p=argparse.ArgumentParser(description="SMIRK/DECA own-vs-Multiface experiment tools")
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("doctor");q.add_argument("--out",required=True);q.add_argument("--render-test",action="store_true");q.add_argument("--require-cuda",action="store_true")
    q=sub.add_parser("fetch-upstream");q.add_argument("--model",choices=["smirk","deca"],required=True);q.add_argument("--destination",required=True)
    q=sub.add_parser("index-own");q.add_argument("--spec",required=True);q.add_argument("--out",required=True)
    q=sub.add_parser("index-multiface");q.add_argument("--root",required=True);q.add_argument("--out",required=True);q.add_argument("--cameras",nargs="+");q.add_argument("--sequences",nargs="+");q.add_argument("--frame-step",type=int,default=1);q.add_argument("--require-mesh",action="store_true");q.add_argument("--geometry-unit",choices=["mm","cm","m"])
    q=sub.add_parser("audit");q.add_argument("--manifest",required=True);q.add_argument("--out",required=True);q.add_argument("--require-aux",action="store_true")
    q=sub.add_parser("split");q.add_argument("--manifest",required=True);q.add_argument("--out",required=True);q.add_argument("--counts",type=int,nargs=3,metavar=("TRAIN","VAL","TEST"),required=True);q.add_argument("--seed",type=int,default=20261027)
    q=sub.add_parser("subset");q.add_argument("--manifest",required=True);q.add_argument("--out",required=True);q.add_argument("--subjects",type=int,required=True);q.add_argument("--frames-per-subject",type=int,required=True);q.add_argument("--seed",type=int,default=17)
    q=sub.add_parser("merge");q.add_argument("--manifests",nargs="+",required=True);q.add_argument("--out",required=True)
    q=sub.add_parser("preprocess");q.add_argument("--manifest",required=True);q.add_argument("--out",required=True);q.add_argument("--mediapipe-asset",required=True);q.add_argument("--device",default="cpu")
    q=sub.add_parser("plan");q.add_argument("--local",required=True);q.add_argument("--out",required=True);q.add_argument("--seeds",nargs="+",type=int,default=[17,29,43])
    q=sub.add_parser("check-run");q.add_argument("--config",required=True)
    q=sub.add_parser("train");q.add_argument("--config",required=True);q.add_argument("--resume")
    q=sub.add_parser("infer");q.add_argument("--config",required=True);q.add_argument("--manifest",required=True);q.add_argument("--out",required=True);q.add_argument("--checkpoint")
    q=sub.add_parser("export-own-points");q.add_argument("--manifest",required=True);q.add_argument("--out",required=True);q.add_argument("--unit",choices=["mm","cm","m"],required=True);q.add_argument("--coordinate-frame",required=True)
    q=sub.add_parser("project-points");q.add_argument("--manifest",required=True);q.add_argument("--mapping",required=True);q.add_argument("--out",required=True)
    q=sub.add_parser("evaluate");q.add_argument("--reference",required=True);q.add_argument("--predictions",required=True);q.add_argument("--protocol",required=True);q.add_argument("--out",required=True)
    q=sub.add_parser("summarize");q.add_argument("--results",nargs="+",required=True);q.add_argument("--out",required=True)
    a=p.parse_args(argv)
    from . import data
    if a.command=="doctor":
        from .tools import doctor
        result=doctor(a.render_test);write_json(a.out,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if (a.require_cuda and not result["cuda_ready"]) or (a.render_test and not result.get("renderer",{}).get("ok")):return 2
        return 0
    if a.command=="fetch-upstream":
        from .experiments import fetch_upstream
        result=fetch_upstream(a.model,a.destination)
    elif a.command in {"index-own","index-multiface"}:
        rows=list(data.own_records(a.spec)) if a.command=="index-own" else list(data.multiface_records(a.root,a.cameras,a.sequences,a.frame_step,a.require_mesh,a.geometry_unit))
        result=data.audit(rows)
        if not result["ok"]:raise ValueError(result)
        write_jsonl(a.out,rows)
    elif a.command=="audit":
        result=data.audit(load_manifest(a.manifest),require_aux=a.require_aux);write_json(a.out,result)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result["ok"] else 2
    elif a.command=="split":
        groups=data.split_subjects(load_manifest(a.manifest),dict(zip(["train","val","test"],a.counts)),a.seed)
        for name,rows in groups.items():write_jsonl(Path(a.out)/f"{name}.jsonl",rows)
        result={name:{"samples":len(rows),"subjects":len({data.identity(r) for r in rows})} for name,rows in groups.items()};write_json(Path(a.out)/"split.json",result)
    elif a.command=="subset":
        rows=data.balanced_subset(load_manifest(a.manifest),a.subjects,a.frames_per_subject,a.seed);write_jsonl(a.out,rows);result={"samples":len(rows)}
    elif a.command=="merge":
        rows=[r for path in a.manifests for r in load_manifest(path)];result=data.audit(rows)
        if not result["ok"]:raise ValueError(result)
        write_jsonl(a.out,rows)
    elif a.command=="preprocess":
        from .preprocessing import Detector,preprocess
        detector=Detector(a.mediapipe_asset,a.device)
        try:result=preprocess(load_manifest(a.manifest),a.out,detector)
        finally:detector.close()
        if not result["accepted"]:raise ValueError("No valid preprocessed samples; see failures.jsonl")
    elif a.command=="plan":
        from .experiments import make_plan
        result={"configs":make_plan(a.local,a.out,a.seeds)}
    elif a.command=="check-run":
        from .experiments import check_experiment,read_config
        train,val,upstream=check_experiment(read_config(a.config));result={"train_samples":len(train),"val_samples":len(val),"upstream":upstream}
    elif a.command=="train":
        from .experiments import train
        result=train(a.config,a.resume)
    elif a.command=="infer":
        from .experiments import infer
        result=infer(a.config,a.manifest,a.out,a.checkpoint)
    elif a.command=="export-own-points":
        from .tools import export_own_points
        result=export_own_points(load_manifest(a.manifest),a.out,a.unit,a.coordinate_frame)
    elif a.command=="project-points":
        from .tools import project_predictions
        result=project_predictions(load_manifest(a.manifest),a.mapping,a.out)
    elif a.command=="evaluate":
        from .evaluation import evaluate_points
        result=evaluate_points(load_manifest(a.reference),load_manifest(a.predictions),read_json(a.protocol));write_json(a.out,result)
        result={k:v for k,v in result.items() if k not in {"per_frame","per_subject"}}
    elif a.command=="summarize":
        table=[]
        for path in a.results:
            r=read_json(path)
            table.append({"result_file":str(Path(path).resolve()),"error_mm":r["subject_macro_mean_error_mm"],"coverage":r["subject_macro_coverage"],"success_rate":r["subject_macro_success_rate"],"failed_subjects":r["subjects_without_scored_predictions"]})
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        with Path(a.out).open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        result={"rows":len(table),"csv":a.out}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
