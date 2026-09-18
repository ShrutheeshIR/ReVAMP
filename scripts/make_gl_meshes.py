"""Make Drake-GL-renderable visual meshes + a patched URDF variant.

The blender-pipeline converter writes OBJs without normals (fine for
Blender/meshcat); Drake's RenderEngineGl refuses them. This re-exports every
<visual> OBJ with vertex normals into meshes/visual_gl/ and writes
fr3_expo_spherized_gl.urdf referencing them. Colors are irrelevant here
(silhouettes / tinted ghosts), geometry is what matters.
"""
from __future__ import annotations

import os
import re
import sys

import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODEL_DIR = os.path.dirname(common.URDF)
VIS = os.path.join(MODEL_DIR, "meshes", "visual")
VIS_GL = os.path.join(MODEL_DIR, "meshes", "visual_gl")
URDF_GL = common.URDF.replace(".urdf", "_gl.urdf")


def main():
    os.makedirs(VIS_GL, exist_ok=True)
    with open(common.URDF) as f:
        text = f.read()
    names = sorted(set(re.findall(r"meshes/visual/([\w.]+)\.obj", text)))
    for name in names:
        src = os.path.join(VIS, f"{name}.obj")
        dst = os.path.join(VIS_GL, f"{name}.obj")
        mesh = trimesh.load(src, force="mesh")
        _ = mesh.vertex_normals  # force computation
        mesh.export(dst, include_normals=True)
        print(f"{name}.obj: {len(mesh.faces)} faces -> visual_gl/")
    text = text.replace("meshes/visual/", "meshes/visual_gl/")
    with open(URDF_GL, "w") as f:
        f.write(text)
    print(f"wrote {URDF_GL}")


if __name__ == "__main__":
    main()
