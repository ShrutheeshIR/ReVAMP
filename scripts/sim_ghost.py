"""Phase 2: render the simulated robot with the CALIBRATED camera and
superimpose it over the real footage (sim over real, low opacity).

The Drake render camera convention (+X right, +Y down, +Z forward) matches
cv2's, so the calibrated (rvec, tvec) map directly: X_WC = inverse of the
base->camera transform. Intrinsics go straight into CameraInfo (distortion is
zero in the current calibration).

--frames T [T ...]     ghost stills -> scratch/ghost_<t>.jpg
--segment T0 T1 OUT    ghost video segment
--alpha A              sim opacity (default 0.45)
--obstacles            also draw logged sphere obstacles in the sim layer
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from render_overlay import grab_bgr  # noqa: E402

RENDERER = "ghost_renderer"


class GhostRenderer:
    def __init__(self, with_obstacle_slots=0):
        from pydrake.geometry import (
            ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
            MakeRenderEngineGl, RenderCameraCore, RenderEngineGlParams,
            Rgba, Sphere)
        from pydrake.math import RigidTransform, RotationMatrix
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.multibody.tree import UnitInertia, SpatialInertia
        from pydrake.systems.framework import DiagramBuilder
        from pydrake.systems.sensors import CameraInfo, RgbdSensor

        cam = common.read_json(self._camera_path())
        self.cam = cam
        w, h = cam["image_wh"]

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
        scene_graph.AddRenderer(RENDERER, MakeRenderEngineGl(
            RenderEngineGlParams(default_clear_color=Rgba(0, 0, 0, 0))))
        parser = Parser(plant)
        parser.package_map().Add("fr3_marker", os.path.dirname(common.URDF))
        (model,) = parser.AddModels(common.urdf_for_rendering())
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("fr3_link0", model))

        # Floating spheres for logged obstacles, parked far away when unused.
        # Own model instance so the arm's SetPositions stays 7-dof.
        self.obstacle_bodies = []
        obs_model = (plant.AddModelInstance("obstacles")
                     if with_obstacle_slots else None)
        for i in range(with_obstacle_slots):
            body = plant.AddRigidBody(
                f"obs_{i}", obs_model,
                SpatialInertia(1.0, np.zeros(3), UnitInertia.SolidSphere(0.03)))
            plant.RegisterVisualGeometry(
                body, RigidTransform(), Sphere(0.03), f"obs_{i}_vis",
                np.array([0.82, 0.56, 0.0, 1.0]))  # McVAMP orange #D18F00
            self.obstacle_bodies.append(body)

        plant.Finalize()

        R_CB = cv2.Rodrigues(np.array(cam["rvec"]))[0]
        t_CB = np.array(cam["tvec"])
        X_BC = RigidTransform(RotationMatrix(R_CB.T), -R_CB.T @ t_CB)

        info = CameraInfo(w, h, cam["f"], cam["f"], cam["cx"], cam["cy"])
        core = RenderCameraCore(RENDERER, info,
                                ClippingRange(0.05, 10.0), RigidTransform())
        color_cam = ColorRenderCamera(core, False)
        depth_cam = DepthRenderCamera(core, DepthRange(0.1, 9.9))
        sensor = builder.AddSystem(RgbdSensor(
            scene_graph.world_frame_id(), X_BC, color_cam, depth_cam))
        builder.Connect(scene_graph.get_query_output_port(),
                        sensor.query_object_input_port())
        builder.ExportOutput(sensor.color_image_output_port(), "color")

        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant = plant
        self.model = model
        self.plant_context = plant.GetMyContextFromRoot(self.context)

        js = common.load_joint_states()
        self.qs = common.joint_matrix(js)
        self.t_log = js["t"]
        self.v0 = cam["video_start_epoch"]
        self.delta = cam["delta_s"]
        self.queries = common.load_queries()
        self.q_t = np.array([q["t"] for q in self.queries])

    @staticmethod
    def _camera_path():
        return common.camera_path()

    def q_at(self, t_video):
        te = t_video + self.v0 - self.delta
        i = np.searchsorted(self.t_log, te)
        i = np.clip(i, 1, len(self.t_log) - 1)
        w = ((te - self.t_log[i - 1])
             / (self.t_log[i] - self.t_log[i - 1] + 1e-12))
        return (1 - np.clip(w, 0, 1)) * self.qs[i - 1] + np.clip(w, 0, 1) * self.qs[i]

    def obstacles_at(self, t_video):
        """Sphere obstacles at a video time, interpolated between the
        bracketing query snapshots (obstacles are only logged per replan,
        median 2.3 s apart, so raw snapshots visibly lag the moving wand).
        Spheres are matched greedily by nearest neighbor across snapshots."""
        te = t_video + self.v0 - self.delta
        i = np.searchsorted(self.q_t, te) - 1
        if i < 0:
            return []
        a = self.queries[i]["spheres"]
        if i + 1 >= len(self.queries):
            return a
        b = self.queries[i + 1]["spheres"]
        w = (te - self.q_t[i]) / (self.q_t[i + 1] - self.q_t[i])
        w = float(np.clip(w, 0.0, 1.0))
        if not a or not b:
            return a if w < 0.5 else b
        pa = np.array([s["position"] for s in a])
        pb = np.array([s["position"] for s in b])
        used, out = set(), []
        for k, p in enumerate(pa):
            d = np.linalg.norm(pb - p, axis=1)
            for j in np.argsort(d):
                if j not in used:
                    break
            if d[j] < 0.35:  # same physical marker, plausibly
                used.add(j)
                out.append({"position": ((1 - w) * p + w * pb[j]).tolist(),
                            "radius": a[k]["radius"]})
            elif w < 0.5:
                out.append(a[k])
        if w >= 0.5:
            out.extend(b[j] for j in range(len(b)) if j not in used)
        return out

    def render(self, t_video, obstacles=False):
        """RGBA sim layer (uint8, h x w x 4) at a video time."""
        from pydrake.math import RigidTransform
        self.plant.SetPositions(self.plant_context, self.model,
                                self.q_at(t_video))
        spheres = self.obstacles_at(t_video) if obstacles else []
        for k, body in enumerate(self.obstacle_bodies):
            pos = (np.array(spheres[k]["position"], float)
                   if k < len(spheres) else np.array([0, 0, -5.0]))
            self.plant.SetFreeBodyPose(self.plant_context, body,
                                       RigidTransform(pos))
        img = (self.diagram.GetOutputPort("color")
               .Eval(self.context).data.copy())
        return img  # RGBA

    def composite(self, real_bgr, t_video, alpha=0.45, obstacles=False,
                  tint=None):
        """tint: optional BGR triple; pushes the sim layer toward that color
        so the ghost reads against the identical-looking real robot."""
        rgba = self.render(t_video, obstacles=obstacles)
        sim_bgr = rgba[:, :, [2, 1, 0]].astype(np.float32)
        if tint is not None:
            sim_bgr = 0.45 * sim_bgr + 0.55 * np.array(tint, np.float32)
        a = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * alpha
        out = real_bgr.astype(np.float32) * (1 - a) + sim_bgr * a
        return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=float, nargs="+")
    ap.add_argument("--segment", nargs=3, metavar=("T0", "T1", "OUT"))
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--tint", action="store_true",
                    help="tint the sim layer ReVAMP blue")
    args = ap.parse_args()
    tint = (203, 160, 141) if args.tint else None  # #8da0cb as BGR

    gr = GhostRenderer(with_obstacle_slots=9 if args.obstacles else 0)
    os.makedirs(common.SCRATCH, exist_ok=True)

    if args.frames:
        for t in args.frames:
            out = gr.composite(grab_bgr(t), t, args.alpha, args.obstacles,
                               tint=tint)
            p = os.path.join(common.SCRATCH, f"ghost_{t:.1f}.jpg")
            cv2.imwrite(p, cv2.resize(out, (1920, 1080)))
            print(p)

    if args.segment:
        t0, t1, out_path = float(args.segment[0]), float(args.segment[1]), \
            args.segment[2]
        w, h = common.VIDEO_WH
        fps = common.VIDEO_FPS
        dec = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-ss", str(t0), "-to", str(t1),
             "-i", common.VIDEO, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, bufsize=w * h * 3 * 2)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
             "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE)
        n, nbytes = 0, w * h * 3
        while True:
            buf = dec.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            enc.stdin.write(gr.composite(img, t0 + n / fps, args.alpha,
                                         args.obstacles, tint=tint).tobytes())
            n += 1
            if n % 120 == 0:
                print(f"{n} frames")
        dec.wait()
        enc.stdin.close()
        enc.wait()
        print(f"{n} frames -> {out_path}")


if __name__ == "__main__":
    main()
