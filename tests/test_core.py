import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
from PIL import Image
from facebench.cli import main
from facebench.data import audit, assert_disjoint, balanced_subset, identity, multiface_records, own_records, read_rgb, split_subjects
from facebench.evaluation import displacement_error, evaluate_points, trajectory_errors
from facebench.experiments import make_plan, sample_batch, validation_rows
from facebench.geometry import barycentric_points, krt_read, obj_vertices, transform_points
from facebench.io import load_manifest, read_json, read_jsonl, write_json, write_jsonl
from facebench.preprocessing import apply_crop, crop_matrix, hull_mask, normalized_landmarks, preprocess
from facebench.tools import export_own_points, project_predictions


class CoreTests(unittest.TestCase):
    def setUp(self):
        base=os.environ.get("FACEBENCH_TEST_ROOT")
        if base:Path(base).mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def rows(self,n=6,frames=4):
        rows=[]
        for person in range(n):
            for frame in range(frames):
                image=self.root/f"p{person}_{frame}.png"
                Image.new("RGB",(32,32),(100,80,70)).save(image)
                rows.append({"sample_id":f"own/p{person}/{frame}","dataset_id":"own","subject_id":f"p{person}","sequence_id":"seq","camera_id":"front","image_path":str(image)})
        return rows
    def point(self,sid,person,xyz,ids=None,valid=None,frame="rig",unit="mm"):
        path=self.root/f"points{len(list(self.root.glob('points*.npz')))}.npz"
        np.savez(path,xyz=np.array(xyz,float),point_ids=np.array(ids or ["F01","F02"]),valid=np.array(valid if valid is not None else [True]*len(xyz)))
        return {"sample_id":sid,"dataset_id":"own","subject_id":person,"points_path":str(path),"geometry_unit":unit,"geometry_frame":frame,"timestamp_s":0.}

    def test_split_is_identity_disjoint_and_reproducible(self):
        rows=self.rows();a=split_subjects(rows,{"train":3,"val":1,"test":2},42);b=split_subjects(list(reversed(rows)),{"train":3,"val":1,"test":2},42)
        self.assertEqual({identity(r) for r in a["train"]},{identity(r) for r in b["train"]})
        self.assertEqual(len(a["train"]),12);assert_disjoint(*a.values())
    def test_no_fake_split_for_one_person(self):
        with self.assertRaises(ValueError):split_subjects(self.rows(n=1),{"train":1,"val":1,"test":1})
    def test_alias_identity_leak_is_detected(self):
        a=self.rows(n=1)[:1];b=self.rows(n=2)[4:5]
        a[0]["identity_id"]=b[0]["identity_id"]="same_person"
        with self.assertRaises(ValueError):assert_disjoint(a,b)
    def test_same_image_under_different_id_is_leak(self):
        a=self.rows(n=1)[:1];b=[dict(a[0],subject_id="different",sample_id="different")]
        with self.assertRaises(ValueError):assert_disjoint(a,b)
    def test_duplicate_and_missing_file_audit(self):
        row=self.rows(n=1)[0];self.assertFalse(audit([row,row])["ok"])
        row["image_path"]=str(self.root/"absent.png");self.assertFalse(audit([row])["ok"])
    def test_relative_manifest_paths(self):
        path=self.root/"index.jsonl";row=self.rows(n=1)[0];row["image_path"]=Path(row["image_path"]).name
        write_jsonl(path,[row]);self.assertTrue(Path(load_manifest(path)[0]["image_path"]).is_file())
    def test_balanced_subset_does_not_oversample(self):
        selected=balanced_subset(self.rows(),2,3,7)
        self.assertEqual(len(selected),6);self.assertEqual(len({identity(r) for r in selected}),2)
        with self.assertRaises(ValueError):balanced_subset(self.rows(n=1),2,3,7)
    def test_deca_batches_group_same_person(self):
        rows=self.rows();groups={str(i):rows[i*4:(i+1)*4] for i in range(6)}
        random.seed(7);batch=sample_batch(groups,2,2)
        self.assertEqual(identity(batch[0]),identity(batch[1]));self.assertEqual(identity(batch[2]),identity(batch[3]));self.assertNotEqual(identity(batch[0]),identity(batch[2]))
    def test_validation_selection_is_stable(self):
        rows=self.rows();a=validation_rows(rows,2);b=validation_rows(rows[::-1],2)
        self.assertEqual([r["sample_id"] for r in a],[r["sample_id"] for r in b]);self.assertEqual(len(a),12)
    def test_multiface_index_sparse_frame_names(self):
        root=self.root/"m--person";p=root/"images/E001/400002";p.mkdir(parents=True)
        (root/"KRT").write_text("camera data")
        for frame in ["000102","000105"]:Image.new("RGB",(8,8)).save(p/f"{frame}.png")
        rows=list(multiface_records(self.root));self.assertEqual(len(rows),2);self.assertIsNone(rows[0]["timestamp_s"])
        with self.assertRaises(ValueError):list(multiface_records(self.root,require_mesh=True))
    def test_krt_block_reader(self):
        p=self.root/"KRT";p.write_text("cam\n1 0 0\n0 1 0\n0 0 1\n0 0 0 0 0\n1 0 0 0\n0 1 0 0\n0 0 1 0\n")
        self.assertEqual(krt_read(p)["cam"]["intrinsics"],np.eye(3).tolist())
    def test_obj_reader(self):
        p=self.root/"face.obj";p.write_text("v 1 2 3\nv 4 5 6\nf 1 2 1\n")
        np.testing.assert_array_equal(obj_vertices(p),[[1,2,3],[4,5,6]])
    def test_own_timestamp_and_rgb_index_mapping(self):
        (self.root/"rgb.mp4").touch()
        write_jsonl(self.root/"frames.jsonl",[{"stream":"rgb","device_ts_ms":t} for t in [1000,1033,1066]])
        write_jsonl(self.root/"points.jsonl",[{"source_indices":{"rgb":2},"markers":[]}])
        spec={"sessions":[{"subject_id":"P1","session_id":"S1","sequence_id":"Q1","camera_id":"front","video_path":"rgb.mp4","frame_count":3,"timestamps_path":"frames.jsonl","timestamp_stream":"rgb","timestamp_value_key":"device_ts_ms","timestamp_to_seconds":.001,"points_path":"points.jsonl","points_rgb_index_key":"source_indices.rgb"}]}
        write_json(self.root/"spec.json",spec);rows=list(own_records(self.root/"spec.json"))
        self.assertAlmostEqual(rows[1]["timestamp_s"],1.033);self.assertIsNone(rows[0]["points_path"]);self.assertEqual(rows[2]["points_record_index"],0)
    @unittest.skipUnless(importlib.util.find_spec("cv2"),"Install video extra to test MP4 decoding")
    def test_mp4_random_access_selects_requested_rgb_frame(self):
        import cv2
        path=self.root/"test.mp4"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),30,(32,32))
        self.assertTrue(writer.isOpened())
        for intensity in (20,80,140,220):writer.write(np.full((32,32,3),intensity,np.uint8))
        writer.release()
        for index,expected in ((3,220),(0,20),(2,140)):
            image=read_rgb({"sample_id":str(index),"video_path":str(path),"frame_index":index})
            self.assertLess(abs(float(image.mean())-expected),10)
    def test_capture_never_converts_numeric_tracks_to_F_ids(self):
        p=self.root/"raw.jsonl";write_jsonl(p,[{"markers":[{"id":8,"xyz_mm":[1,2,3]}]}])
        row={"dataset_id":"own","subject_id":"P1","sample_id":"own/P1/0","points_path":str(p),"points_record_index":0,"geometry_unit":"mm","geometry_frame":"rig"}
        with self.assertRaises(ValueError):export_own_points([row],self.root/"export","mm","rig")
        row["point_id_mapping"]={"8":"F03"};out=export_own_points([row],self.root/"export","mm","rig")
        with np.load(load_manifest(out["manifest"])[0]["points_path"]) as a:self.assertEqual(a["point_ids"][0],"F03")
    def test_fixed_transform_and_barycentric_mapping(self):
        points=barycentric_points([[0,0,0],[2,0,0],[0,2,0]],[[0,1,2]],[[.5,.25,.25]])
        matrix=np.eye(4);matrix[:3,:3]*=1000;matrix[:3,3]=[1,2,3]
        np.testing.assert_allclose(transform_points(points,matrix),[[501,502,3]])
        with self.assertRaises(ValueError):barycentric_points([[0,0,0]],[[0,0,0]],[[1,1,1]])
    def test_crop_coordinates_and_mask(self):
        points=np.array([[8,8],[24,8],[24,24],[8,24]])
        matrix=crop_matrix(points,1,224);norm=normalized_landmarks(points,matrix)
        self.assertAlmostEqual(norm[0,0],-1);self.assertAlmostEqual(norm[2,0],223/224*2-1)
        mask=hull_mask(points,32,32);self.assertEqual(mask[16,16],1);self.assertEqual(mask[0,0],0)
        self.assertEqual(apply_crop(np.zeros((32,32,3),np.uint8),matrix).shape,(224,224,3))
    def test_preprocess_failure_is_reported_not_replaced(self):
        rows=self.rows(n=1,frames=2)
        def fake(image):raise ValueError("no face")
        out=preprocess(rows,self.root/"prep",fake)
        self.assertEqual(out["accepted"],0);self.assertEqual(out["failed"],2)
    def test_evaluation_matches_marker_names_not_array_order(self):
        ref=self.point("x","p1",[[0,0,0],[10,0,0]])
        pred=self.point("x","p1",[[13,0,0],[3,0,0]],ids=["F02","F01"])
        result=evaluate_points([ref],[pred],{"coordinate_reviewed":True,"reference_description":"synthetic test","geometry_frame":"rig","success_threshold_mm":2})
        self.assertEqual(result["subject_macro_mean_error_mm"],3);self.assertEqual(result["subject_macro_success_rate"],0)
    def test_person_macro_and_failed_predictions(self):
        refs=[self.point(str(i),"long",[[0,0,0],[0,0,0]]) for i in range(5)]
        refs.append(self.point("short","short",[[0,0,0],[0,0,0]]))
        preds=[self.point(str(i),"long",[[2,0,0],[2,0,0]]) for i in range(5)]
        preds.append(self.point("short","short",[[10,0,0],[10,0,0]]))
        protocol={"coordinate_reviewed":True,"reference_description":"synthetic","geometry_frame":"rig"}
        result=evaluate_points(refs,preds,protocol);self.assertEqual(result["subject_macro_mean_error_mm"],6)
        result=evaluate_points(refs,preds[:-1],protocol);self.assertEqual(result["subjects_without_scored_predictions"],1);self.assertEqual(result["subject_macro_coverage"],.5)
    def test_evaluation_rejects_wrong_units_and_timestamps(self):
        ref=self.point("x","p1",[[0,0,0],[0,0,0]]);pred=self.point("x","p1",[[0,0,0],[0,0,0]],unit="m")
        protocol={"coordinate_reviewed":True,"reference_description":"synthetic","geometry_frame":"rig"}
        with self.assertRaises(ValueError):evaluate_points([ref],[pred],protocol)
        pred["geometry_unit"]="mm";pred["timestamp_s"]=.5
        with self.assertRaises(ValueError):evaluate_points([ref],[pred],protocol)
    def test_irregular_time_velocity_error(self):
        ref=np.zeros((3,1,3));p=ref.copy();p[:,0,0]=[0,2,8]
        result=trajectory_errors(p,ref,[0,1,3],np.ones((3,1),bool));self.assertEqual(result["mean_velocity_error_mm_s"],2.5)
        with self.assertRaises(ValueError):trajectory_errors(p,ref,[0,0,3],np.ones((3,1),bool))
    def test_local_displacement_uses_separate_neutral_positions(self):
        ref=np.array([[[10,0,0]],[[12,0,0]]],float)
        pred=np.array([[[100,0,0]],[[103,0,0]]],float)
        result=displacement_error(pred,ref,[[100,0,0]],[[10,0,0]],np.ones((2,1),bool))
        self.assertEqual(result["mean_displacement_error_mm"],.5)
    def test_plan_produces_twelve_separate_runs(self):
        local={"models":{m:{"upstream_root":m,"checkpoint":m+".pt"} for m in ["smirk","deca"]},"sources":{s:{"train_manifest":s+".jsonl"} for s in ["own","multiface"]},"val_manifest":"val.jsonl","output_root":"outputs"}
        p=self.root/"local.json";write_json(p,local);paths=make_plan(p,self.root/"runs",[17,29,43])
        self.assertEqual(len(paths),12)
        configs=[read_json(path) for path in paths];self.assertEqual(len({c["output_dir"] for c in configs}),12)
        for model in ["smirk","deca"]:
            selected=[c for c in configs if c["model"]==model]
            self.assertEqual(len({c["checkpoint"] for c in selected}),1)
    def test_cli_end_to_end_index_audit_split_subset(self):
        rows=self.rows();p=self.root/"all.jsonl";write_jsonl(p,rows)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit","--manifest",str(p),"--out",str(self.root/"audit.json")]),0)
            self.assertEqual(main(["split","--manifest",str(p),"--out",str(self.root/"split"),"--counts","3","1","2"]),0)
            self.assertEqual(main(["subset","--manifest",str(self.root/"split/train.jsonl"),"--out",str(self.root/"budget.jsonl"),"--subjects","2","--frames-per-subject","3"]),0)
        self.assertEqual(len(list(read_jsonl(self.root/"budget.jsonl"))),6)


if __name__=="__main__":unittest.main()
