"""Generate a UV-mapped box .obj + .mtl for a real photo texture, tiled at a
real-world scale -- Drake's SDFormat parser has no support for a texture map
on a primitive <box> (confirmed empirically: `<material><pbr>...` is parsed
and then silently ignored -- "Ignoring unsupported SDFormat element in
material: pbr"), and only reads map_Kd off a mesh's own .mtl. So getting a
real wood-grain/metal photo onto the shelf/table means an actual small mesh,
not the plain SDF boxes rby1-constrained-planning's old_shelves.sdf uses.

Per-face UVs are scaled by the face's own real-world size / `tile_m`, so the
texture repeats at a fixed real-world tile size instead of being stretched
edge-to-edge across whatever the box's dimensions happen to be (that
stretch is what made the old baked table_wide.png atlas look wrong when
reused at a different scale).

    python3 make_textured_box.py OUT_DIR NAME W D H TEXTURE_PNG [--tile-m 0.3]
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def write_box(out_dir, name, w, d, h, texture_path, tile_m=0.3, mtl_name=None):
    """Each face is built from its own outward normal N plus two tangents
    U, V with U x V = N (not hand-picked corner orderings, which are easy
    to get backwards -- and were: an earlier version of this used indices
    into a shared 8-corner list and three of the six faces had normals
    pointing inward, confirmed by CalcSpatialInertia() reporting a NEGATIVE
    mesh volume when loaded standalone, and by every face silently
    rendering black in Drake/VTK, which does not shade or draw the far
    side of a backward-facing triangle). The corner order
    (-U-V, +U-V, +U+V, -U+V) is CCW as seen from along +N by construction,
    for any right-handed (N, U, V), so there is no sign to get wrong.
    """
    os.makedirs(out_dir, exist_ok=True)
    mtl_name = mtl_name or name
    half = np.array([w, d, h]) / 2.0
    tex_rel = os.path.relpath(texture_path, out_dir)

    # (outward normal, extent-index-u, extent-index-v) for the 6 faces.
    face_axes = [
        (np.array([1.0, 0, 0]), 1, 2),
        (np.array([-1.0, 0, 0]), 1, 2),
        (np.array([0, 1.0, 0]), 0, 2),
        (np.array([0, -1.0, 0]), 0, 2),
        (np.array([0, 0, 1.0]), 0, 1),
        (np.array([0, 0, -1.0]), 0, 1),
    ]

    verts, uvs, normals, faces_idx = [], [], [], []
    for n, iu, iv in face_axes:
        axis_of_n = int(np.argmax(np.abs(n)))
        u_axis = np.zeros(3); u_axis[iu] = 1.0
        # v = n x u_axis is unit length (n, u_axis orthonormal) and gives
        # u_axis x v == n exactly, so the corner order below is guaranteed
        # CCW as seen from along +n -- no sign case-work needed per face.
        v_axis = np.cross(n, u_axis)
        center = n * half[axis_of_n]
        su, sv = 2 * half[iu], 2 * half[iv]

        base = len(verts)
        for su_sign, sv_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            p = center + su_sign * (su / 2) * u_axis + sv_sign * (sv / 2) * v_axis
            verts.append(p)
        tex_u, tex_v = su / tile_m, sv / tile_m
        uvs.extend([(0.0, 0.0), (tex_u, 0.0), (tex_u, tex_v), (0.0, tex_v)])
        normals.append(n)
        faces_idx.append([base + 1, base + 2, base + 3, base + 4])

    lines_obj = [f"mtllib {name}.mtl", f"o {name}"]
    for p in verts:
        lines_obj.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    for u, v_ in uvs:
        lines_obj.append(f"vt {u:.4f} {v_:.4f}")
    for n in normals:
        lines_obj.append(f"vn {n[0]:.1f} {n[1]:.1f} {n[2]:.1f}")

    lines_obj.append(f"usemtl {mtl_name}")
    for face_i, idx in enumerate(faces_idx, start=1):
        vt0 = (face_i - 1) * 4
        lines_obj.append(
            f"f {idx[0]}/{vt0 + 1}/{face_i} {idx[1]}/{vt0 + 2}/{face_i} "
            f"{idx[2]}/{vt0 + 3}/{face_i} {idx[3]}/{vt0 + 4}/{face_i}")

    with open(os.path.join(out_dir, f"{name}.obj"), "w") as f:
        f.write("\n".join(lines_obj) + "\n")

    mtl = f"""newmtl {mtl_name}
Ns 60.000000
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.100000 0.100000 0.100000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd {tex_rel}
"""
    with open(os.path.join(out_dir, f"{name}.mtl"), "w") as f:
        f.write(mtl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("name")
    ap.add_argument("w", type=float)
    ap.add_argument("d", type=float)
    ap.add_argument("h", type=float)
    ap.add_argument("texture")
    ap.add_argument("--tile-m", type=float, default=0.3)
    args = ap.parse_args()
    write_box(args.out_dir, args.name, args.w, args.d, args.h, args.texture,
             tile_m=args.tile_m)
    print(f"wrote {args.out_dir}/{args.name}.obj (+.mtl)")


if __name__ == "__main__":
    main()
