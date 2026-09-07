"""Shared Blender-side machinery for the FR3 maze figures. Runs INSIDE Blender.

Imported by blender_maze_still.py and blender_maze_video.py, both of which are
launched as `blender --background --python <script> -- <args>`. Never importable
from a normal interpreter: everything here needs `bpy`.

Ported from ../ift/iiwa-bimanual-augmented-jacobian-test/scripts/figures/
blender_swept_volume.py, with the scene-specific parts (classification, the maze
restyle, the lid variants, the tip trace) rewritten for this project. The parts
that carry hard-won behaviour -- fit_frustum, bake_ghosts, aimed_area_light,
enable_cycles_gpu, the split world -- are kept as close to the original as the
different subject allows, including their reasoning.

The subject here differs from that repo's in a way that matters: the maze board
is 1.415 x 0.575 x 0.115 m, twelve times longer than it is tall, lying flat with
a ~1 m arm standing at its edge. Camera elevation and lighting angle both had to
move as a result; see setup_camera and setup_studio_lighting.
"""

import json
import math
import os
import sys

import addon_utils
import bpy
from mathutils import Vector

# Blender's user extensions directory. Hardcoded rather than imported from
# blender_paths.py: this module runs inside Blender, where the repo is not on
# sys.path. Must agree with blender_paths.EXTENSIONS.
EXTENSIONS = os.path.expanduser("~/.config/blender/5.0/extensions/user_default")

# Meshcat path prefixes. Classification is by path and only by path -- see
# classify() for why object names are useless here.
VISUAL_PREFIX = "/drake/visual/"
ROBOT_PREFIX = "/drake/visual/fr3/"
COLLISION_PREFIXES = ("/drake/collision/", "/drake/proximity/",
                      "/drake/inertia/", "/drake/contact_forces/")

# Dark backdrop for video, matching the sibling repos so the paper's figures and
# its supplementary video read as one set. Stills are transparent by default and
# never see this.
BG_COLOR = (0.102, 0.102, 0.18, 1.0)  # #1a1a2e

# Maze colours by structural role. The walls are the subject -- they carry the
# maze pattern -- so they are the lightest surface and catch the most light; the
# in-channel floor plates are darker so the channels read as recessed rather than
# as flat infill. Overridable per role from the CLI.
ROLE_COLORS = {
    "base":   (0.42, 0.42, 0.46, 1.0),
    "border": (0.50, 0.50, 0.55, 1.0),
    "wall":   (0.72, 0.73, 0.76, 1.0),
    "floor":  (0.30, 0.31, 0.35, 1.0),
    "lid":    (0.60, 0.61, 0.65, 1.0),
}

# The FR3's own shells. Drake hands us flat untextured colour (meshcat strips OBJ
# material libraries), so this is what the arm would otherwise be.
ROBOT_COLOR = (0.82, 0.83, 0.85, 1.0)

# The marker tip trace. #d7263d is the accent figures/robot_arm_fig.py already
# uses for its targets, so the 3D and 2D figures agree.
TRACE_COLOR = (0.843, 0.149, 0.239, 1.0)


# ── import and classification ────────────────────────────────────────────────

def import_scene(html_path, target_fps=30.0):
    """Import the meshcat page and return {meshcat_path: object}.

    Imported through the add-on's Python API rather than
    `bpy.ops.import_scene.meshcat_recording`: that operator only exists once the
    extension is *enabled* in user preferences, which under --background it
    generally is not. Calling it raises, and a script that carries on from there
    renders a perfectly clean picture of an empty scene.
    """
    if EXTENSIONS not in sys.path:
        sys.path.insert(0, EXTENSIONS)
    try:
        from meshcat_html_importer.blender_impl.scene_builder import build_scene_from_file
    except ImportError as e:
        raise SystemExit(
            f"Could not import the meshcat_html_importer add-on from {EXTENSIONS}: {e}\n"
            "Verify it with: python3 scripts/figures/check_meshcat_importer.py --probe"
        ) from e

    objs = build_scene_from_file(html_path, target_fps=target_fps, clear_scene=True,
                                 hierarchical_collections=True)
    print(f"[import] {len(objs)} objects from {html_path}")
    if not objs:
        raise SystemExit(f"[import] the add-on returned no objects from {html_path}. "
                         "The HTML is empty or not a meshcat recording.")
    return objs


def load_manifest(path):
    """Read planks.json and index it by meshcat visual path."""
    with open(path, "r") as f:
        data = json.load(f)
    return data, {p["meshcat_visual_path"]: p for p in data["planks"]}


def classify(objs, manifest_path, expect_collision=False):
    """Split the imported objects by meshcat path, deleting collision geometry.

    Returns a dict with keys: robot, planks (list), by_role (role -> list),
    by_name (plank name -> object), other.

    Classification is by meshcat path and never by object name: the importer
    derives names from paths and collapses the leaf when it is called `visual`,
    so a maze plank and a robot mesh can both arrive named plain `visual.017`.

    Collision geometry is deleted rather than hidden -- `hide_render` has proven
    unreliable across Blender/engine versions. It should not be here at all,
    because export_maze_scene.py runs `meshcat.Delete("collision")` before
    StaticHtml; finding any is a sign the HTML is stale, which is why a nonzero
    count is an error rather than a note. The FR3 URDF is spherized, so a leak
    would import as a string of collision spheres along every link.
    """
    data, by_path = load_manifest(manifest_path)

    robot, planks, other, removed = [], [], [], 0
    by_name, by_role = {}, {}

    for path, obj in objs.items():
        if path.startswith(COLLISION_PREFIXES):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
            continue
        plank = by_path.get(path)
        if plank is not None:
            planks.append(obj)
            by_name[plank["name"]] = obj
            by_role.setdefault(plank["role"], []).append(obj)
        elif path.startswith(ROBOT_PREFIX):
            robot.append(obj)
        elif path.startswith(VISUAL_PREFIX):
            other.append((path, obj))
        else:
            other.append((path, obj))

    counts = {role: len(by_role.get(role, [])) for role in
              ("base", "border", "wall", "floor", "lid")}
    print(f"[classify] robot={len(robot)} planks={len(planks)}/{data['total']} "
          + "(" + ", ".join(f"{r} {counts[r]}" for r in counts) + ") "
          + f"other={len(other)} deleted={removed}")

    for path, _ in other:
        print(f"[classify] unclassified: {path}")

    if not robot:
        raise SystemExit(
            f"[classify] no robot geometry found under {ROBOT_PREFIX}. This HTML is a "
            "different scene, or the URDF's robot name changed.")
    if len(planks) != data["total"]:
        raise SystemExit(
            f"[classify] found {len(planks)} of {data['total']} maze planks. The HTML and "
            f"the manifest disagree -- regenerate both with export_maze_scene.py.")
    if removed and not expect_collision:
        raise SystemExit(
            f"[classify] deleted {removed} collision objects, which should not have been "
            "exported at all. The HTML is stale: re-run export_maze_scene.py (without "
            "--keep-collision). Pass --expect-collision to render it anyway.")

    return {"robot": robot, "planks": planks, "by_role": by_role,
            "by_name": by_name, "other": [o for _, o in other], "manifest": data}


# ── maze appearance ──────────────────────────────────────────────────────────

def _flat_material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        # No metallic anywhere in this scene: the maze is painted board and the
        # FR3's shells are white plastic. Metallic on either throws a blown
        # highlight rather than reading as material.
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def _parse_color(value, fallback):
    """`#rrggbb`, `r,g,b`, or `r,g,b,a` -> an RGBA tuple."""
    if not value:
        return fallback
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) != 6:
            raise SystemExit(f"--*-color '{value}' is not #rrggbb")
        r, g, b = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        return (r, g, b, 1.0)
    parts = [float(v) for v in text.split(",")]
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2], 1.0)
    if len(parts) == 4:
        return tuple(parts)
    raise SystemExit(f"--*-color '{value}' should be #rrggbb, 'r,g,b' or 'r,g,b,a'")


def restyle_maze(parts, args):
    """Give each plank role its own flat material, and break the top-plane tie.

    Drake registers every maze box with SceneBox's default colour, so all 124
    planks arrive as the same flat grey and the board renders as an
    undifferentiated slab. Colouring by structural role is what makes the maze
    legible as a maze. Done here rather than in scene.py so the Drake scene the
    planner and the collision checker see stays exactly as it was.

    The z nudge is the other half of this function: 60 of the 124 planks -- the
    25 lids, the 33 tall interior walls and the 2 border walls -- have their top
    faces at exactly z=0.1025. Exactly coincident coplanar faces render as black
    patches under Cycles' CPU/Embree backend, so the lids are lifted by a
    fraction of a millimetre to break the tie before the artifact can appear.
    """
    overrides = {
        "base": _parse_color(getattr(args, "base_color", None), ROLE_COLORS["base"]),
        "border": _parse_color(getattr(args, "wall_color", None), ROLE_COLORS["border"]),
        "wall": _parse_color(getattr(args, "wall_color", None), ROLE_COLORS["wall"]),
        "floor": _parse_color(getattr(args, "floor_color", None), ROLE_COLORS["floor"]),
        "lid": _parse_color(getattr(args, "lid_color", None), ROLE_COLORS["lid"]),
    }

    materials = {role: _flat_material(f"maze_{role}", rgba)
                 for role, rgba in overrides.items()}
    for role, objs in parts["by_role"].items():
        mat = materials[role]
        for obj in objs:
            if obj.type != "MESH":
                continue
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    nudge = getattr(args, "z_fight_nudge", 1e-4)
    if nudge:
        for obj in parts["by_role"].get("lid", []):
            obj.location.z += nudge

    n_vertex_coloured = _restyle_robot(parts["robot"], args)
    print(f"[material] maze coloured by role, lid z-nudge {nudge:.1e} m, "
          f"{n_vertex_coloured}/{len(parts['robot'])} robot meshes using vertex colour")


def _vertex_colour_material(name, layer_name, roughness=0.6):
    """A Principled material whose base colour is read from a vertex colour layer.

    convert_visual_meshes.py bakes each FR3 link's per-part colours into vertex
    colours, which is what keeps the white shells distinct from the dark grey
    joint housings. Blender's OBJ importer brings the layer in but wires nothing
    to it, so without this node the whole arm renders as one flat colour.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    attr = nt.nodes.new("ShaderNodeVertexColor")
    attr.layer_name = layer_name
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _restyle_robot(robot_objects, args):
    """Give the FR3 its materials, preferring the baked vertex colours.

    Falls back to a flat shell colour for any mesh that arrived without a colour
    layer (the marker, holder and tip were always .obj and carry none).
    """
    override = getattr(args, "robot_color", None)
    flat = _flat_material("fr3_shell", _parse_color(override, ROBOT_COLOR))
    cache = {}
    used_vertex_colour = 0

    for obj in robot_objects:
        if obj.type != "MESH":
            continue
        layers = getattr(obj.data, "color_attributes", None)
        # An explicit --robot-color is a deliberate override, so it wins over
        # whatever the mesh carries.
        if layers and len(layers) and not override:
            layer_name = layers[0].name
            mat = cache.get(layer_name)
            if mat is None:
                mat = _vertex_colour_material(f"fr3_vc_{layer_name}", layer_name)
                cache[layer_name] = mat
            used_vertex_colour += 1
        else:
            mat = flat
        obj.data.materials.clear()
        obj.data.materials.append(mat)

    return used_vertex_colour


def apply_lid_variant(parts, args):
    """Handle the 25 lid plates, which otherwise hide the whole maze.

    The maze is a tray whose channels are capped from above by thin plates. A
    render that keeps them shows an opaque top surface and none of the structure
    the marker is actually tracing, so every figure has to decide what to do
    about them. The four treatments:

      keep     leave them. Honest to the physical rig; shows almost no maze.
      cull     delete all 25. Cleanest read of the channels.
      glass    low alpha, and NOT casting shadows -- a transparent lid that still
               casts leaves the channel just as dark as an opaque one did, which
               is the whole failure this variant exists to avoid.
      cutaway  delete only the lids named in --cull-planks, chosen driver-side by
               maze_manifest.select_cutaway_lids against the traced path, so the
               board still reads as a capped maze except where you need to see in.

    Deleting rather than hiding, for the same reason as the collision geometry.
    """
    lids = parts["by_role"].get("lid", [])
    variant = args.lid_variant
    doomed = []

    if variant == "cull":
        doomed = list(lids)

    elif variant == "cutaway":
        names = set(getattr(args, "cull_planks", None) or [])
        if not names:
            print("[lids] WARNING: --lid-variant cutaway with no --cull-planks; nothing "
                  "will be removed. The driver computes this list from the tip path, so "
                  "this usually means no trajectory was available.")
        for name in names:
            obj = parts["by_name"].get(name)
            if obj is None:
                print(f"[lids] WARNING: --cull-planks named {name}, which is not in "
                      "this scene")
                continue
            doomed.append(obj)

    elif variant == "glass":
        alpha = args.lid_alpha
        mat = _flat_material("maze_lid_glass",
                             _parse_color(getattr(args, "lid_color", None),
                                          ROLE_COLORS["lid"]))
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Alpha"].default_value = alpha
            bsdf.inputs["Roughness"].default_value = 0.15
        for obj in lids:
            if obj.type != "MESH":
                continue
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            # The load-bearing line: without it the channel stays dark and the
            # transparency buys nothing.
            obj.visible_shadow = False

    culled = _delete_objects(parts, doomed)
    lids = parts["by_role"].get("lid", [])
    print(f"[lids] variant={variant} kept={len(lids)} culled={culled}"
          + (f" alpha={args.lid_alpha}" if variant == "glass" else ""))
    return lids


def _delete_objects(parts, doomed):
    """Remove objects from Blender AND from every list in `parts` that holds them.

    Both halves are necessary. A Blender object that has been removed leaves its
    Python reference dangling, and merely touching `obj.type` on one afterwards
    raises `ReferenceError: StructRNA of type Object has been removed`. Since
    `parts["planks"]` is what the camera is later fitted to, culling lids without
    pruning these lists crashes the render several stages later, in code that has
    nothing to do with lids.
    """
    if not doomed:
        return 0

    condemned = set(id(obj) for obj in doomed)

    def surviving(objects):
        return [obj for obj in objects if id(obj) not in condemned]

    parts["planks"] = surviving(parts["planks"])
    parts["robot"] = surviving(parts["robot"])
    parts["other"] = surviving(parts["other"])
    for role, objects in list(parts["by_role"].items()):
        parts["by_role"][role] = surviving(objects)
    for name, obj in list(parts["by_name"].items()):
        if id(obj) in condemned:
            del parts["by_name"][name]

    for obj in doomed:
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(doomed)


# ── marker tip trace ─────────────────────────────────────────────────────────

def add_tip_trace(args):
    """Draw the marker tip's path as a Blender curve with a round bevel.

    A curve rather than geometry published from Drake: `meshcat.SetLine` does not
    survive this importer at all (the add-on records the meshcat object type and
    never reads it, then triangulates the un-indexed buffer from consecutive
    vertex triples, so a polyline arrives as garbage triangles). Building it here
    also makes radius, colour and lift render-time flags rather than properties
    baked into the exported HTML, so iterating on them costs seconds instead of a
    re-export.

    The lift is not cosmetic. The marker rides on the channel floor, so the trace
    is otherwise coincident with the floor plates -- the same coplanar-face
    problem the lid nudge addresses, and at a far more visible place.
    """
    if not getattr(args, "trace_json", None):
        return None
    if not os.path.isfile(args.trace_json):
        print(f"[trace] no tip path at {args.trace_json}; skipping the trace")
        return None

    with open(args.trace_json, "r") as f:
        data = json.load(f)
    pts = data["xyz"]
    if len(pts) < 2:
        print("[trace] fewer than 2 tip samples; skipping the trace")
        return None

    curve = bpy.data.curves.new("TipTrace", 'CURVE')
    curve.dimensions = '3D'
    spline = curve.splines.new('POLY')
    spline.points.add(len(pts) - 1)
    lift = args.trace_lift
    for i, (x, y, z) in enumerate(pts):
        spline.points[i].co = (x, y, z + lift, 1.0)

    curve.bevel_depth = args.trace_radius
    curve.bevel_resolution = 4
    curve.use_fill_caps = True

    mat = _flat_material("tip_trace",
                         _parse_color(getattr(args, "trace_color", None), TRACE_COLOR))
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # A little emission so the trace stays readable down inside a channel,
        # where very little of the studio rig reaches.
        bsdf.inputs["Emission Color"].default_value = _parse_color(
            getattr(args, "trace_color", None), TRACE_COLOR)
        bsdf.inputs["Emission Strength"].default_value = args.trace_emission
    curve.materials.append(mat)

    obj = bpy.data.objects.new("TipTrace", curve)
    bpy.context.scene.collection.objects.link(obj)

    progress = getattr(args, "trace_progress", 1.0)
    if progress is not None and progress < 1.0:
        curve.bevel_factor_end = max(0.0, min(1.0, progress))

    zs = [p[2] for p in pts]
    print(f"[trace] {len(pts)} points, radius {args.trace_radius * 1000:.1f} mm, "
          f"lift {lift * 1000:.1f} mm, source z [{min(zs):.4f}, {max(zs):.4f}]")
    return obj


def animate_trace_growth(trace, frame_start, frame_end):
    """Keyframe the trace so it draws itself across the animation.

    tip_path.json is sampled uniformly in time from the retimed trajectory and
    the spline is POLY, so bevel_factor_end tracks the marker to within one point
    spacing. Close enough to look right; check it by eye at a few frames rather
    than assuming.
    """
    if trace is None:
        return
    curve = trace.data
    curve.bevel_factor_mapping_end = 'SPLINE'
    curve.bevel_factor_end = 0.0
    curve.keyframe_insert("bevel_factor_end", frame=frame_start)
    curve.bevel_factor_end = 1.0
    curve.keyframe_insert("bevel_factor_end", frame=frame_end)

    # Linear, so the trace grows at the rate the marker actually moves. Bezier
    # (Blender's default) eases in and out, which would make the drawn length
    # lag the marker at both ends of the clip.
    n = 0
    for fcurve in _action_fcurves(curve.animation_data):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'LINEAR'
            n += 1
    print(f"[trace] growing across frames {frame_start}-{frame_end} "
          f"({n} keyframes set to linear)")


def _action_fcurves(animation_data):
    """F-curves of an action, across Blender's two action APIs.

    Blender 5.0 replaced `action.fcurves` with slotted actions, where the curves
    live under layers -> strips -> channelbags and are reached through the slot
    the animated ID is assigned to. `action.fcurves` raises AttributeError there.
    Older builds have only the flat list. Yields nothing rather than raising if
    neither shape is available, since interpolation is a refinement and not worth
    failing a render over.
    """
    action = getattr(animation_data, "action", None)
    if action is None:
        return

    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy
        return

    slot = getattr(animation_data, "action_slot", None)
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            channelbag = None
            try:
                channelbag = strip.channelbag(slot) if slot is not None else None
            except Exception:
                channelbag = None
            if channelbag is None:
                # Some builds expose the bags directly when there is only one slot.
                for bag in getattr(strip, "channelbags", ()):
                    yield from getattr(bag, "fcurves", ())
                continue
            yield from getattr(channelbag, "fcurves", ())


# ── ghosting (ported from blender_swept_volume.py) ───────────────────────────

def frame_window(t_start, t_end):
    scene = bpy.context.scene
    f0, f1 = scene.frame_start, scene.frame_end
    span = f1 - f0
    lo = int(round(f0 + t_start * span))
    hi = int(round(f0 + t_end * span))
    return f0, f1, list(range(lo, hi + 1))


def scan_motion(moving, window):
    """World-space bounding-box corners of every moving object at every frame.

    Corners rather than object origins: a link that spins about its own origin --
    the FR3's shoulder, for one -- never translates, so a test on
    matrix_world.translation alone would call it static and draw it once.
    """
    scene = bpy.context.scene
    track = {}
    for frame in window:
        scene.frame_set(frame)
        # Without this the depsgraph has not re-evaluated and every frame reads
        # back the same matrices.
        bpy.context.view_layer.update()
        for obj in moving:
            m = obj.matrix_world
            track.setdefault(obj, []).append([m @ Vector(c) for c in obj.bound_box])
    return track


def split_static(moving, track, tol):
    """Separate the links that actually move from the ones welded in place.

    fr3_link0 is welded to the world, so ghosting it stacks n_poses coincident
    copies of the same mesh. Identical surfaces at identical depths z-fight, and
    each layer of alpha multiplies into the next, so the base comes out both
    speckled and far more opaque than the rest of the arm. Drawing it once,
    opaque, is what it is.
    """
    truly_moving, welded = [], []
    for obj in moving:
        corners = track[obj]
        first = corners[0]
        travel = max((c - f).length for frame in corners for c, f in zip(frame, first))
        (welded if travel <= tol else truly_moving).append(obj)
    print(f"[static] {len(truly_moving)} links move, {len(welded)} are welded in place "
          "and drawn once")
    if not truly_moving:
        raise SystemExit("Nothing moves over the sampled window -- check --t-start/--t-end, "
                         "and check the export recorded more than one frame.")
    return truly_moving, welded


def bias_fractions(n, mid_bias):
    """`n` fractions of [0, 1], pushed together in the middle of the interval.

        s(u) = u + (b / 2*pi) * sin(2*pi*u),   ds/du = 1 + b*cos(2*pi*u)

    keeps the endpoints, stays monotone for b < 1, and stretches the steps near
    u=0 and u=1 by (1+b) while squeezing those near u=0.5 by (1-b). `--mid-bias 0`
    is plain even spacing.
    """
    if n == 1:
        return [0.0]
    return [u + (mid_bias / (2 * math.pi)) * math.sin(2 * math.pi * u)
            for u in (i / (n - 1) for i in range(n))]


def apply_nudge(fractions, args):
    """Shift individual poses along the path, for hand-correcting a stacked pair."""
    nudge = getattr(args, "pose_nudge", None)
    if not nudge:
        return fractions
    out = list(fractions)
    for i, delta in enumerate(nudge):
        if i < len(out) and delta:
            out[i] = min(1.0, max(0.0, out[i] + delta))
    return sorted(out)


def apply_drops(fractions, args):
    """Remove specific poses by index, for when one is simply in the way."""
    drops = set(getattr(args, "drop_poses", None) or ())
    if not drops:
        return fractions
    kept = [f for i, f in enumerate(fractions) if i not in drops]
    print(f"[poses] dropped {sorted(drops)}, {len(kept)} remain")
    return kept


def sample_frames(window, track, truly_moving, args):
    """Pick n_poses frames spaced along the motion, weighted toward the middle.

    Spacing by arc length rather than by time: a TOPPRA-retimed trajectory creeps
    away from the start and decelerates into the goal, so uniform-in-time samples
    pile ghosts at both ends and leave a gap through the interesting middle.
    Walking the cumulative path length of the moving geometry spaces the poses
    the way the eye reads them.
    """
    n = args.n_poses
    if n == 1:
        return [window[0]]
    fractions = apply_drops(apply_nudge(bias_fractions(n, args.mid_bias), args), args)

    if args.spacing == "time":
        frames = [window[int(round((len(window) - 1) * f))] for f in fractions]
        print(f"[poses] uniform in time, mid-bias {args.mid_bias}, frames {frames}")
        return frames

    cumulative = [0.0]
    for k in range(1, len(window)):
        step = sum((track[obj][k][j] - track[obj][k - 1][j]).length
                   for obj in truly_moving for j in range(8))
        cumulative.append(cumulative[-1] + step)
    total = cumulative[-1]
    if total <= 0.0:
        raise SystemExit("The sampled window has zero path length.")

    frames, k = [], 0
    for f in fractions:
        target = total * f
        while k + 1 < len(cumulative) and cumulative[k + 1] < target:
            k += 1
        j = k + 1 if (k + 1 < len(cumulative)
                      and abs(cumulative[k + 1] - target) < abs(cumulative[k] - target)) else k
        frames.append(window[j])
    print(f"[poses] arc length over frames {window[0]}-{window[-1]} (total {total:.2f}), "
          f"mid-bias {args.mid_bias}, frames {frames}")
    return frames


def ghost_alpha(index, n_poses, args):
    """Uniform across the sweep, so no pose reads as heavier than another.

    --endpoint-alpha opts back into solid first/last poses; off by default
    because a mixed ramp reads as an artifact rather than as a cue.
    """
    if args.endpoint_alpha is not None and index in (0, n_poses - 1):
        return args.endpoint_alpha
    return args.alpha


def alpha_material(cache, source, alpha):
    """A copy of `source` with Principled Alpha set, one per (material, alpha)."""
    key = (source.name if source else None, round(alpha, 4))
    if key in cache:
        return cache[key]
    mat = source.copy() if source else bpy.data.materials.new("ghost")
    mat.name = f"{mat.name}_a{round(alpha * 100):03d}"
    if not mat.use_nodes:
        mat.use_nodes = True
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Alpha"].default_value = alpha
            break
    cache[key] = mat
    return mat


def bake_ghosts(moving, frames, args):
    """One copy of every moving object at every sampled frame.

    The mesh datablock is shared -- the geometry is rigid, only the transform
    differs -- and the faded material is attached through an OBJECT-linked slot
    so the shared mesh data is untouched.
    """
    scene = bpy.context.scene
    cache = {}
    n = len(frames)
    ghosts = []
    for i, frame in enumerate(frames):
        scene.frame_set(frame)
        # Without this the depsgraph has not re-evaluated and every ghost bakes
        # at the same pose.
        bpy.context.view_layer.update()

        alpha = ghost_alpha(i, n, args)
        coll = bpy.data.collections.new(f"Sweep_{i:02d}")
        scene.collection.children.link(coll)

        # Stacked semi-transparent poses each casting a shadow means every pose is
        # dimmed by the ones in front of it, which reads as uneven opacity rather
        # than as depth. Only the final pose casts by default, so the figure stays
        # anchored to the board without the sweep shadowing itself.
        casts = (args.ghost_shadows == "all"
                 or (args.ghost_shadows == "last" and i == n - 1))

        for obj in moving:
            ghost = obj.copy()          # shares obj.data
            ghost.animation_data_clear()
            ghost.matrix_world = obj.matrix_world.copy()
            ghost.visible_shadow = casts
            coll.objects.link(ghost)
            for slot_index, slot in enumerate(ghost.material_slots):
                source = (obj.data.materials[slot_index]
                          if slot_index < len(obj.data.materials) else None)
                slot.link = 'OBJECT'
                slot.material = alpha_material(cache, source, alpha)
            ghosts.append(ghost)
        print(f"[ghost] pose {i} at frame {frame}, alpha {alpha:.2f}")

    # The originals are still animated; leaving them in would draw an extra,
    # unfaded pose on top of the sweep.
    for obj in list(moving):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.frame_start = scene.frame_end = scene.frame_current = 0
    print(f"[ghost] {len(ghosts)} ghost objects in {n} poses, {len(cache)} faded materials")
    return ghosts


# ── look ─────────────────────────────────────────────────────────────────────

def improve_scene_quality():
    """Smooth shading and a flat, non-metallic roughness pass on all meshes.

    Must run BEFORE restyle_maze and bake_ghosts: it walks obj.data.materials and
    would otherwise stamp over the role colours, or miss the ghost copies
    entirely.
    """
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        for poly in obj.data.polygons:
            poly.use_smooth = True
        try:
            if hasattr(obj.data, 'use_auto_smooth'):
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = math.radians(30)
        except Exception:
            pass
        for mat in (obj.data.materials or ()):
            if not (mat and mat.use_nodes):
                continue
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if not bsdf:
                continue
            color = bsdf.inputs['Base Color'].default_value
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            # No metallic anywhere. Every bright surface in this scene is painted
            # board or white plastic; at Metallic 0.3 they blow out instead of
            # reading as material.
            bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Roughness'].default_value = 0.65 if brightness > 0.6 else 0.6


def aimed_area_light(name, center, radius, azimuth, elevation, distance_scale,
                     power, size, color, focus):
    """A soft area light on an orbit around the subject, aimed at its centre.

    Placed relative to the subject rather than at fixed world coordinates, and
    with power scaling as distance squared, so the exposure does not change when
    the subject's bounding radius does.
    """
    data = bpy.data.lights.new(name, 'AREA')
    distance = distance_scale * radius
    data.energy = power * distance * distance
    data.size = size * radius
    data.color = color
    obj = bpy.data.objects.new(name, data)
    az, el = math.radians(azimuth), math.radians(elevation)
    obj.location = center + Vector((
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    ))
    con = obj.constraints.new(type='TRACK_TO')
    con.target = focus
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    bpy.context.scene.collection.objects.link(obj)
    return obj


def setup_studio_lighting(args, center, radius, camera_azimuth, focus):
    """Four large soft sources ringed around the subject, plus a sun.

    Retuned from the sibling repo's rig for a subject that is essentially a
    horizontal plane. There, the key sits at 45 degrees elevation; here that
    gives the board a broad flat wash in which the 9.5 cm walls throw almost no
    shadow, and the maze pattern disappears. Dropping the key to ~30 degrees
    makes the walls cast short shadows into the channels, so the maze reads as
    relief rather than as a printed texture.

    The wrap around the camera axis is kept: a single key leaves half the
    channels -- the ones whose walls face away from it -- completely black.
    """
    k = args.light_strength
    az = camera_azimuth
    key_el = args.key_elevation

    aimed_area_light('Key', center, radius, az + 40, key_el, 2.6, 9 * k, 2.5,
                     (1.0, 0.98, 0.95), focus)
    aimed_area_light('Fill', center, radius, az - 55, 20, 2.8, 9 * k, 3.0,
                     (0.90, 0.95, 1.0), focus)
    aimed_area_light('Back', center, radius, az + 165, 35, 3.0, 6 * k, 3.0,
                     (0.92, 0.94, 1.0), focus)
    aimed_area_light('Top', center, radius, az + 90, 80, 2.6, 6 * k, 3.0,
                     (1.0, 1.0, 1.0), focus)

    sun = bpy.data.lights.new('Sun', 'SUN')
    sun.energy = 0.35 * k
    sun.angle = math.radians(20)     # a wide disc, for soft sun shadows
    sun.color = (1.0, 0.98, 0.95)
    so = bpy.data.objects.new('Sun', sun)
    so.location = center + Vector((0, 0, 4 * radius))
    con = so.constraints.new(type='TRACK_TO')
    con.target = focus
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    bpy.context.scene.collection.objects.link(so)
    print(f"[light] 4 area sources + sun, strength x{k:.2f}, key elevation {key_el:.0f}, "
          f"ring radius {2.6 * radius:.2f} m")


def setup_world(args):
    """Dark backdrop to the camera, bright ambient to everything else.

    One Background node cannot do both: raising its strength to lift the shadows
    also raises the visible backdrop until the figure floats on a grey field.
    Splitting on `Is Camera Ray` keeps what the camera sees at --world-strength
    while the light the geometry receives comes from a brighter neutral dome at
    --ambient, which is what actually fills the shadows inside the channels.
    """
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    visible = nt.nodes.new('ShaderNodeBackground')
    visible.inputs['Color'].default_value = BG_COLOR
    visible.inputs['Strength'].default_value = args.world_strength

    ambient = nt.nodes.new('ShaderNodeBackground')
    ambient.inputs['Color'].default_value = (0.55, 0.58, 0.65, 1.0)
    ambient.inputs['Strength'].default_value = args.ambient

    path = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixShader')
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(ambient.outputs['Background'], mix.inputs[1])
    nt.links.new(visible.outputs['Background'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    print(f"[light] world backdrop {args.world_strength:.2f}, ambient {args.ambient:.2f}")


# ── camera ───────────────────────────────────────────────────────────────────

def corners_of(objects):
    """World-space bounding-box corners of `objects` as a flat list."""
    pts = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        pts.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    if not pts:
        raise SystemExit("Nothing to frame the camera on.")
    return pts


def sphere_of(pts):
    lo = Vector((min(p[i] for p in pts) for i in range(3)))
    hi = Vector((max(p[i] for p in pts) for i in range(3)))
    return (lo + hi) / 2.0, max((hi - lo).length / 2.0, 1e-6)


def fit_frustum(pts, direction, tan_w, tan_h, margin, iterations=24):
    """Smallest camera distance, and the aim point, that just contains `pts`.

    Solves the frustum directly instead of bounding the subject with a sphere.
    That matters more here than in the repo this came from: the maze board is
    twelve times longer than it is tall, and a bounding sphere reserves room for
    a ball around the whole thing, leaving most of the frame empty.

    With aim point C, camera at C - D*direction, and per-point camera-space
    coordinates u (right), v (up), w (along the view), a point is inside the
    frustum when |u| <= (D + w)*tan_w and |v| <= (D + w)*tan_h, so

        D = max over points of max(|u|/tan_w - w, |v|/tan_h - w).

    C is then slid within the view plane to balance the two sides and the
    distance re-solved, which converges in a few passes; that recentring is what
    stops one extreme corner pushing the whole subject off to the side.
    """
    up_hint = Vector((0.0, 0.0, 1.0))
    up = (up_hint - direction * up_hint.dot(direction)).normalized()
    right = direction.cross(up).normalized()

    center, _ = sphere_of(pts)
    distance = 0.0
    for _ in range(iterations):
        local = [((p - center).dot(right), (p - center).dot(up), (p - center).dot(direction))
                 for p in pts]
        distance = max(max(abs(u) / tan_w - w, abs(v) / tan_h - w) for u, v, w in local)
        distance = max(distance, 1e-3)
        sx = [u / ((distance + w) * tan_w) for u, v, w in local]
        sy = [v / ((distance + w) * tan_h) for u, v, w in local]
        du = 0.5 * (max(sx) + min(sx)) * tan_w * distance
        dv = 0.5 * (max(sy) + min(sy)) * tan_h * distance
        if abs(du) < 1e-6 and abs(dv) < 1e-6:
            break
        center = center + right * du + up * dv

    local = [((p - center).dot(right), (p - center).dot(up), (p - center).dot(direction))
             for p in pts]
    fill_x = max(abs(u) / ((distance + w) * tan_w) for u, v, w in local)
    fill_y = max(abs(v) / ((distance + w) * tan_h) for u, v, w in local)
    return center, distance * margin, fill_x, fill_y


# Defaults for THIS scene, to be pinned once the contact sheet settles them.
# Elevation is the parameter that differs most from the sibling repo's 22: a flat
# tray seen from 22 degrees foreshortens into a line and shows only its lids.
DEFAULT_AZIMUTH = 35.0
DEFAULT_ELEVATION = 40.0


def setup_camera(args, subject, light_subject):
    """Orbit the subject and solve the distance so it just fills the frame.

    Resolution must already be set: Blender's AUTO sensor fit maps the 36 mm
    sensor width onto the longer axis, so the field of view depends on the aspect
    ratio and a camera fitted before configure_output is fitted to the wrong one.
    """
    scene = bpy.context.scene
    pts = corners_of(subject)

    azimuth = math.radians(args.camera_azimuth if args.camera_azimuth is not None
                           else DEFAULT_AZIMUTH)
    elevation = math.radians(args.camera_elevation if args.camera_elevation is not None
                             else DEFAULT_ELEVATION)

    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = args.lens
    cam_data.clip_end = 1000.0

    aspect = scene.render.resolution_x / scene.render.resolution_y
    half_sensor = math.atan(cam_data.sensor_width / (2 * args.lens))
    if aspect >= 1.0:                      # AUTO fit puts the sensor on the long axis
        tan_w, tan_h = math.tan(half_sensor), math.tan(half_sensor) / aspect
    else:
        tan_h, tan_w = math.tan(half_sensor), math.tan(half_sensor) * aspect

    orbit = Vector((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    center, distance, fill_x, fill_y = fit_frustum(pts, -orbit, tan_w, tan_h,
                                                   args.frame_margin)
    if args.camera_distance is not None:
        distance = args.camera_distance
    if getattr(args, "camera_target", None):
        center = Vector([float(v) for v in args.camera_target.split(",")])

    cam = bpy.data.objects.new('Camera', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = center + orbit * distance

    target = bpy.data.objects.new('CamTarget', None)
    target.location = center
    scene.collection.objects.link(target)
    con = cam.constraints.new(type='TRACK_TO')
    con.target = target
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'

    # Printed so a good framing can be pinned as the driver's defaults. A fill
    # well under 1.0 on one axis means that much of the frame is empty, and it is
    # the resolution's aspect ratio that wants changing, not the camera.
    print(f"[camera] center=({center.x:.3f}, {center.y:.3f}, {center.z:.3f}) "
          f"azimuth={math.degrees(azimuth):+.1f} elevation={math.degrees(elevation):+.1f} "
          f"distance={distance:.3f} lens={args.lens:.1f} "
          f"fill=({fill_x:.2f}, {fill_y:.2f})")

    light_center, light_radius = sphere_of(corners_of(light_subject))
    return light_center, light_radius, math.degrees(azimuth), target, cam


def write_camera_json(cam, path):
    """Record the render camera so a later 2D overlay pass can project with it.

    Read back off the object rather than recomputed: a projection that disagrees
    with the render by even a little puts annotations in the wrong place.
    """
    scene = bpy.context.scene
    data = {
        "location": list(cam.matrix_world.translation),
        "matrix_world": [list(row) for row in cam.matrix_world],
        "lens": cam.data.lens,
        "sensor_width": cam.data.sensor_width,
        "sensor_fit": cam.data.sensor_fit,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[camera] wrote {path}")


# ── render configuration ─────────────────────────────────────────────────────

def enable_cycles_gpu(prefer=("OPTIX", "CUDA", "HIP", "ONEAPI")):
    """Enable the Cycles add-on and select a compute backend.

    Returns the backend name selected, or None if no GPU is usable.

    Cycles is not enabled under --factory-startup, so addon_utils.enable has to
    run before `scene.render.engine = 'CYCLES'` is even a valid assignment.
    get_devices() must be called AFTER setting compute_device_type or the device
    list stays empty and every device silently ends up with use = False -- a full
    CPU fallback that otherwise looks exactly like a successful GPU render.
    Filter on d.type, never on d.name: the same GPU is enumerated once per
    backend it supports.
    """
    try:
        addon_utils.enable("cycles", default_set=True, persistent=True)
    except Exception as e:
        print(f"[render] could not enable the Cycles add-on: {e}")
        return None

    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return None
    cprefs = addon.preferences

    for backend in prefer:
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue  # this build has no such backend
        try:
            cprefs.get_devices()
        except Exception:
            pass
        if not any(d.type == backend for d in cprefs.devices):
            continue
        # Enable only the accelerator. Leaving the CPU on as well makes Cycles
        # split tiles between devices, which is slower here than the GPU alone.
        for d in cprefs.devices:
            d.use = (d.type == backend)
        return backend
    return None


def configure_render(args):
    """Put the scene on Cycles, on a GPU where there is one and the CPU where not.

    Deliberately not an automatic EEVEE fallback. The ghost figure is stacked
    semi-transparent geometry, and what makes that composite correctly is
    cycles.transparent_max_bounces, which EEVEE has no equivalent for -- so a
    silent drop to EEVEE would not be a slower render of the same picture, it
    would be a different picture. Cycles on CPU is slow and right. EEVEE is still
    reachable with --engine EEVEE, explicitly.

    Always prints the engine and device actually selected: a silent fall back to
    the CPU changes only how long the render takes and is otherwise
    indistinguishable from success, which is exactly the failure the drivers grep
    for.
    """
    scene = bpy.context.scene

    if args.engine == "EEVEE":
        # Blender 5.0 folded EEVEE Next back into BLENDER_EEVEE; assigning the
        # 4.x identifier raises TypeError here, so try both.
        for identifier in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
            try:
                scene.render.engine = identifier
                break
            except (TypeError, AttributeError):
                continue
        try:
            scene.eevee.taa_render_samples = max(args.samples, 64)
        except AttributeError:
            pass
        print(f"[render] {scene.render.engine} by request "
              "-- ghost and glass transparency will not match Cycles")
        return scene.render.engine

    if args.device == "CPU":
        backend = None
        try:
            addon_utils.enable("cycles", default_set=True, persistent=True)
        except Exception as e:
            raise SystemExit(f"could not enable the Cycles add-on: {e}")
    elif args.device == "auto":
        backend = enable_cycles_gpu()
    else:
        backend = enable_cycles_gpu(prefer=(args.device,))
        if backend is None:
            raise SystemExit(
                f"--device {args.device} was requested but no {args.device} device is "
                "usable in this Blender build. Use --device auto or --device CPU.")

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU' if backend else 'CPU'
    scene.cycles.samples = args.samples
    try:
        scene.cycles.use_adaptive_sampling = True
        # Tighter than the 0.01 default: stacked transparency (ghosts, and the
        # glass lid variant) is the noisiest thing in frame, and adaptive
        # sampling will otherwise call it converged while it is visibly grainy.
        scene.cycles.adaptive_threshold = 0.004
    except AttributeError:
        pass
    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPTIX' if backend == 'OPTIX' else 'OPENIMAGEDENOISE'
    except (AttributeError, TypeError):
        pass

    # Overlapping transparent surfaces go black without a generous bounce budget
    # -- every layer of alpha costs one.
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.max_bounces = 16

    if backend:
        print(f"[render] CYCLES on GPU via {backend}, {args.samples} samples")
    else:
        print(f"[render] CYCLES on CPU, {args.samples} samples")
    return f"CYCLES/{backend or 'CPU'}"


def configure_output(args):
    """Resolution, file format and view transform. Must run BEFORE setup_camera."""
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    # Transparent by default: stills go into the paper over whatever the page
    # background is, and a baked-in navy field would sit in a box on it. The
    # world's visible colour still matters under --opaque-bg (the video path),
    # and the ambient dome lights the scene either way -- film_transparent only
    # affects camera rays that miss all geometry.
    scene.render.film_transparent = not args.opaque_bg
    # AgX rolls off highlights and desaturates; a technical figure wants the
    # colours it was given.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    print(f"[output] {args.resolution[0]}x{args.resolution[1]} RGBA, "
          f"film_transparent={scene.render.film_transparent}")


def script_args():
    """Arguments after the `--` separator Blender uses to end its own options."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def add_common_arguments(ap):
    """CLI surface shared by the still and video renderers.

    Kept in one place so the two drivers can forward flags without the two
    renderers drifting apart.
    """
    ap.add_argument("--html", required=True, help="meshcat StaticHtml file to import")
    ap.add_argument("--manifest", required=True, help="planks.json from export_maze_scene.py")
    ap.add_argument("--expect-collision", action="store_true",
                    help="tolerate collision geometry in the HTML instead of erroring")

    ap.add_argument("--lid-variant", choices=("keep", "cull", "glass", "cutaway"),
                    default="keep", help="how to treat the 25 lid plates (default: keep)")
    ap.add_argument("--lid-alpha", type=float, default=0.18,
                    help="alpha for --lid-variant glass (default: 0.18)")
    ap.add_argument("--cull-planks", nargs="*", default=None,
                    help="plank names to delete, for --lid-variant cutaway")
    ap.add_argument("--z-fight-nudge", type=float, default=1e-4,
                    help="metres to lift the lids, breaking the exactly-coplanar top "
                         "faces that otherwise render as black patches (default: 1e-4)")

    ap.add_argument("--base-color", help="#rrggbb for the base plate")
    ap.add_argument("--wall-color", help="#rrggbb for the maze walls")
    ap.add_argument("--floor-color", help="#rrggbb for the in-channel floor plates")
    ap.add_argument("--lid-color", help="#rrggbb for the lids")
    ap.add_argument("--robot-color", help="#rrggbb for the FR3's shells")

    ap.add_argument("--trace-json", help="tip_path.json; omit to draw no trace")
    ap.add_argument("--trace-radius", type=float, default=0.0025,
                    help="trace tube radius in metres (default: 0.0025)")
    ap.add_argument("--trace-lift", type=float, default=0.0015,
                    help="metres to lift the trace off the channel floor, which it is "
                         "otherwise coincident with (default: 0.0015)")
    ap.add_argument("--trace-color", default="#d7263d",
                    help="trace colour (default: #d7263d, the figures/ accent)")
    ap.add_argument("--trace-emission", type=float, default=0.35,
                    help="emission strength so the trace reads inside a channel")

    ap.add_argument("--resolution", type=int, nargs=2, default=(2800, 1400),
                    metavar=("W", "H"))
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--device", default="auto",
                    choices=("auto", "OPTIX", "CUDA", "HIP", "ONEAPI", "CPU"))
    ap.add_argument("--engine", default="CYCLES", choices=("CYCLES", "EEVEE"))

    ap.add_argument("--camera-azimuth", type=float, default=None)
    ap.add_argument("--camera-elevation", type=float, default=None)
    ap.add_argument("--camera-distance", type=float, default=None)
    ap.add_argument("--camera-target", help="x,y,z aim point, overriding the fitted one")
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--frame-margin", type=float, default=1.08)
    ap.add_argument("--camera-json", help="write the render camera's matrix here")

    ap.add_argument("--opaque-bg", action="store_true",
                    help="render the dark backdrop instead of a transparent one")
    ap.add_argument("--world-strength", type=float, default=0.35)
    ap.add_argument("--ambient", type=float, default=0.30)
    # 0.5 rather than the sibling repo's 1.0, settled by a three-value sweep: the
    # maze board is a large, pale, near-horizontal surface facing the rig, and at
    # 1.0 it clips to white and the channels stop reading as recessed at all. At
    # 0.25 the relief is there but the arm goes muddy. 0.5 holds both.
    ap.add_argument("--light-strength", type=float, default=0.5)
    ap.add_argument("--key-elevation", type=float, default=30.0,
                    help="key light elevation; low so the walls cast shadows into the "
                         "channels and the maze reads as relief (default: 30)")
    return ap
