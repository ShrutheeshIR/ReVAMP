"""Make Drake-GL-renderable visual meshes + a patched URDF variant.

Drake's render engines skip .dae visual meshes silently, and RenderEngineGl
refuses OBJs without vertex normals. This converts every <visual> mesh the
URDF references (works from a pristine CRAMP checkout, whose URDF references
.dae files, and equally from a checkout where the meshes were already
converted to .obj) into OBJs WITH vertex normals under meshes/visual_gl/,
and writes fr3_expo_spherized_gl.urdf referencing them. Colors are
irrelevant here (silhouettes / tinted ghosts), geometry is what matters.

Requires trimesh + pycollada (both in requirements.txt).
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

MESH_REF = re.compile(r"meshes/visual/([\w.-]+)\.(dae|obj)")


def main():
    os.makedirs(VIS_GL, exist_ok=True)
    with open(common.URDF) as f:
        text = f.read()
    refs = sorted(set(MESH_REF.findall(text)))
    if not refs:
        sys.exit(f"no meshes/visual/ references found in {common.URDF}")
    for name, ext in refs:
        src = os.path.join(VIS, f"{name}.{ext}")
        dst = os.path.join(VIS_GL, f"{name}.obj")
        mesh = trimesh.load(src, force="mesh")
        _ = mesh.vertex_normals  # force computation
        mesh.export(dst, include_normals=True)
        print(f"{name}.{ext}: {len(mesh.faces)} faces -> visual_gl/{name}.obj")
    text = MESH_REF.sub(r"meshes/visual_gl/\1.obj", text)
    with open(URDF_GL, "w") as f:
        f.write(text)
    print(f"wrote {URDF_GL}")


if __name__ == "__main__":
    main()
