"""Overnight camera refinement against TRUE mesh silhouettes.

Renders the URDF's visual meshes through Drake GL (RenderColorImage with an
arbitrary camera per call, so intrinsics/extrinsics are free parameters
without rebuilding anything) and maximizes interior-vs-ring brightness
contrast of the rendered silhouette against the real frames — the white
robot on black curtains makes that a sharp objective. Compared to
silhouette_calib.py this uses exact geometry instead of collision-sphere
blobs, frees the principal point, and scores ~80 frames spread over the
WHOLE video with a trimmed mean (operator-occluded frames land in the
trimmed tail).

Parameters: f, cx, cy, rvec(3), tvec(3), delta   (11)
Checkpoints every improvement to calib/camera_deep.json (schema-compatible
with camera_sil.json). Progress lines go to stdout for monitoring.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SCALE = int(os.environ.get("DEEP_SCALE", 4))     # work at 3840/SCALE
N_FRAMES = int(os.environ.get("DEEP_FRAMES", 80))
T_RANGE = (9.0, 384.0)
RING_PX = max(4, 32 // SCALE)
TRIM = 0.15               # drop this fraction of worst frames
CACHE = os.path.join(common.SCRATCH,
                     f"deep_calib_frames_s{SCALE}_n{N_FRAMES}.npz")
# Init/output can be overridden so a high-res round refines the low-res one.
INIT = os.environ.get("DEEP_INIT", "camera_sil.json")
OUT = os.environ.get("DEEP_OUT", "camera_deep.json")


class MeshSilhouette:
    def __init__(self):
        from pydrake.geometry import (
            MakeRenderEngineGl, RenderEngineGlParams, Rgba)
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        scene_graph.AddRenderer("deep", MakeRenderEngineGl(
            RenderEngineGlParams(default_clear_color=Rgba(0, 0, 0, 0))))
        parser = Parser(plant)
        parser.package_map().Add("fr3_marker", os.path.dirname(common.URDF))
        (model,) = parser.AddModels(common.urdf_for_rendering())
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("fr3_link0", model))
        plant.Finalize()
        diagram = builder.Build()
        self.context = diagram.CreateDefaultContext()
        self.plant = plant
        self.model = model
        self.plant_context = plant.GetMyContextFromRoot(self.context)
        self.sg = scene_graph
        self.sg_context = scene_graph.GetMyContextFromRoot(self.context)

        js = common.load_joint_states()
        self.qs = common.joint_matrix(js)
        self.t_log = js["t"]

    def q_at(self, t_epoch):
        i = np.clip(np.searchsorted(self.t_log, t_epoch), 1,
                    len(self.t_log) - 1)
        w = np.clip((t_epoch - self.t_log[i - 1])
                    / (self.t_log[i] - self.t_log[i - 1] + 1e-12), 0, 1)
        return (1 - w) * self.qs[i - 1] + w * self.qs[i]

    def mask(self, t_epoch, f, cx, cy, rvec, tvec):
        from pydrake.geometry import (
            ClippingRange, ColorRenderCamera, RenderCameraCore)
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.systems.sensors import CameraInfo

        self.plant.SetPositions(self.plant_context, self.model,
                                self.q_at(t_epoch))
        w, h = common.VIDEO_WH[0] // SCALE, common.VIDEO_WH[1] // SCALE
        info = CameraInfo(w, h, f / SCALE, f / SCALE, cx / SCALE, cy / SCALE)
        cam = ColorRenderCamera(RenderCameraCore(
            "deep", info, ClippingRange(0.05, 10.0), RigidTransform()), False)
        R_CB = cv2.Rodrigues(np.asarray(rvec, float))[0]
        X_WC = RigidTransform(RotationMatrix(R_CB.T),
                              -R_CB.T @ np.asarray(tvec, float))
        qo = self.sg.get_query_output_port().Eval(self.sg_context)
        img = qo.RenderColorImage(cam, self.sg.world_frame_id(), X_WC)
        return np.array(img.data)[:, :, 3] > 0


def load_frames(times):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if np.allclose(z["times"], times):
            return z["frames"].astype(np.float32)
    import subprocess
    w, h = common.VIDEO_WH[0] // SCALE, common.VIDEO_WH[1] // SCALE
    frames = []
    for t in times:
        cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i",
               common.VIDEO, "-frames:v", "1", "-vf", f"scale={w}:{h}",
               "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        buf = subprocess.run(cmd, capture_output=True, check=True).stdout
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w))
        if len(frames) % 20 == 0:
            print(f"loaded {len(frames)}/{len(times)} frames", flush=True)
    frames = np.array(frames)
    os.makedirs(common.SCRATCH, exist_ok=True)
    np.savez_compressed(CACHE, times=times, frames=frames)
    return frames.astype(np.float32)


def main():
    calib = common.CALIB_DIR
    cam0 = common.read_json(os.path.join(calib, INIT))
    print(f"init from {INIT} (scale {SCALE}, {N_FRAMES} frames) -> {OUT}",
          flush=True)
    v0 = cam0["video_start_epoch"]
    w, h = common.VIDEO_WH

    times = np.linspace(*T_RANGE, N_FRAMES)
    frames = load_frames(times)
    print("frames ready", flush=True)

    ms = MeshSilhouette()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
    n_keep = int(round(N_FRAMES * (1 - TRIM)))

    state = {"best": -1e9, "evals": 0, "t0": time.time()}

    def objective(x):
        f, cx, cy = x[0], x[1], x[2]
        rvec, tvec, delta = x[3:6], x[6:9], x[9]
        scores = []
        for t_vid, img in zip(times, frames):
            m = ms.mask(t_vid + v0 - delta, f, cx, cy, rvec, tvec)
            mu8 = m.astype(np.uint8)
            ring = (cv2.dilate(mu8, kernel) & ~mu8) > 0
            if m.sum() < 100 or ring.sum() < 100:
                scores.append(-50.0)
                continue
            scores.append(float(img[m].mean() - img[ring].mean()))
        val = float(np.mean(sorted(scores)[-n_keep:]))
        state["evals"] += 1
        if val > state["best"]:
            state["best"] = val
            R = cv2.Rodrigues(x[3:6])[0]
            common.write_json(os.path.join(calib, OUT), {
                "image_wh": [w, h],
                "f": float(x[0]), "cx": float(x[1]), "cy": float(x[2]),
                "k1": 0.0, "k2": 0.0,
                "rvec": x[3:6].tolist(), "tvec": x[6:9].tolist(),
                "delta_s": float(x[9]),
                "camera_in_base": (-R.T @ x[6:9]).tolist(),
                "video_start_epoch": v0, "objective": val,
                "note": "deep mesh-silhouette fit (trimmed-mean contrast); "
                        "cv2 convention, base->camera."})
            print(f"[{state['evals']:5d} evals {time.time()-state['t0']:6.0f}s]"
                  f" best {val:.3f} f={x[0]:.1f} c=({x[1]:.0f},{x[2]:.0f})"
                  f" delta={x[9]:+.3f}", flush=True)
        return -val

    x0 = np.array([cam0["f"], w / 2, h / 2, *cam0["rvec"], *cam0["tvec"],
                   cam0["delta_s"]])
    print(f"initial objective {-objective(x0):.3f}", flush=True)

    for round_i in range(3):
        res = minimize(objective, x0, method="Powell",
                       options={"maxfev": 6000, "xtol": 1e-5, "ftol": 1e-6})
        print(f"round {round_i}: {-res.fun:.3f} after {res.nfev} evals",
              flush=True)
        if np.allclose(res.x, x0, atol=1e-7):
            break
        x0 = res.x
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
