"""Analysis of the FR3 marker maze experiment logs in this directory:

joint_states.csv -- a recorded joint trajectory (t, q1..q7, plus eef_x/y/z and z_error/
orientation_error_deg columns already computed by whatever system logged this file). We
independently re-run FK ourselves (vamp.fr3marker.eefk, same binding used in
../vamp/scripts/fr3_marker_maze_example.py and ../vamp/scripts/cpp/fr3_maze_solver.cc) on
q1..q7 and compare the resulting z to EXPECTED_EEF_Z below, rather than trusting the file's
own z_error column (which was computed against a *different* expected z -- see its docstring
note below).

planning_queries.jsonl -- one line per planning query issued during the run (rrtc_nanoseconds,
rrtc_iterations, num_waypoints, solved, etc. -- see fr3_maze_solver.cc/
fr3_marker_maze_example.py for what each field means): we report success rate and planning
time/iteration/path-size statistics over it. identify_smm_and_psi_s is read but deliberately
left out of the table/prints -- not useful for this experiment.

Besides the printed summary, this also emits (mirroring ../vamp/scripts/plot_bimanual_results.py):
  - results_table.tex: a ready-to-paste booktabs LaTeX table of the same numbers.
  - z_error_running.pdf/.svg: FK z-error over the course of the recorded trajectory --
    useful for spotting drift/outliers a single aggregate number would hide.
  - planning_time_ecdf.pdf/.svg: empirical CDF of RRTC planning time over solved queries.

Usage:
    python analyze_maze_expt.py
    python analyze_maze_expt.py --joint_states_csv other.csv --planning_queries_jsonl other.jsonl --output_dir plots
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import vamp
from fire import Fire

DEFAULT_JOINT_STATES_CSV = Path(__file__).parent / "joint_states.csv"
DEFAULT_PLANNING_QUERIES_JSONL = Path(__file__).parent / "planning_queries.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots"

# Expected eef z height for this maze task. NOT the same as joint_states.csv's own z_error
# column, which was computed against 0.14 (confirmed: eef_z - z_error == 0.14 exactly for
# every row) -- kept as a separate tunable constant here rather than a Fire arg since it's a
# property of the task setup, not something you'd want to sweep per invocation.
EXPECTED_EEF_Z = 0.15

JOINT_COLUMNS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]


def _set_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "axes.edgecolor": "0.3",
            "axes.linewidth": 0.9,
            "grid.color": "0.88",
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        },
    )


def _fk_z(df: pd.DataFrame) -> np.ndarray:
    """FR3Marker FK (vamp.fr3marker.eefk) z-translation for every row's q1..q7."""
    configs = df[JOINT_COLUMNS].to_numpy(dtype=np.float32)
    return np.array([vamp.fr3marker.eefk(q)[2, 3] for q in configs], dtype=np.float64)


def compute_joint_state_stats(df: pd.DataFrame) -> dict:
    """FK z vs. EXPECTED_EEF_Z, per-row error (m) plus summary stats (mm)."""
    fk_z = _fk_z(df)
    error_m = fk_z - EXPECTED_EEF_Z
    error_mm = error_m * 1e3
    t_relative = df["t"].to_numpy() - df["t"].to_numpy().min()
    return {
        "t_relative": t_relative,
        "error_mm": error_mm,
        "mean": error_mm.mean(),
        "median": np.median(error_mm),
        "std": error_mm.std(),
        "rmse": np.sqrt((error_mm ** 2).mean()),
        "min": error_mm.min(),
        "max": error_mm.max(),
        "mean_abs": np.abs(error_mm).mean(),
    }


def print_joint_state_stats(path: Path, stats: dict, n_rows: int) -> None:
    print(f"=== Joint states FK z-error ({path.name}, {n_rows} rows) ===")
    print(f"Expected eef z: {EXPECTED_EEF_Z} m")
    print(f"Mean error:   {stats['mean']:+.4f} mm")
    print(f"Median error: {stats['median']:+.4f} mm")
    print(f"Std error:    {stats['std']:.4f} mm")
    print(f"RMSE:         {stats['rmse']:.4f} mm")
    print(f"Min error:    {stats['min']:+.4f} mm")
    print(f"Max error:    {stats['max']:+.4f} mm")
    print(f"Mean |error|: {stats['mean_abs']:.4f} mm")
    print()


def _load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _stats(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(),
        "median": np.median(values),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
    }


def _print_stats(name: str, stats: dict, unit: str = "") -> None:
    print(f"{name}:")
    print(f"  mean:   {stats['mean']:.3f}{unit}")
    print(f"  median: {stats['median']:.3f}{unit}")
    print(f"  std:    {stats['std']:.3f}{unit}")
    print(f"  min:    {stats['min']:.3f}{unit}")
    print(f"  max:    {stats['max']:.3f}{unit}")
    print()


def compute_planning_stats(queries: list) -> dict:
    n_total = len(queries)
    solved = [q for q in queries if q["solved"]]
    n_solved = len(solved)

    result = {
        "n_total": n_total,
        "n_solved": n_solved,
        "success_rate": 100.0 * n_solved / n_total if n_total else float("nan"),
    }
    if not solved:
        return result

    result["planning_ms"] = np.array([q["rrtc_nanoseconds"] for q in solved], dtype=np.float64) / 1e6
    result["iterations"] = np.array([q["rrtc_iterations"] for q in solved], dtype=np.float64)
    result["num_waypoints"] = np.array([q["num_waypoints"] for q in solved], dtype=np.float64)
    result["external_ms"] = np.array([q["external_duration_s"] for q in solved], dtype=np.float64) * 1e3
    result["goal_resolve_ms"] = np.array([q["goal_resolve_s"] for q in solved], dtype=np.float64) * 1e3
    result["shortcut_changed_rate"] = 100.0 * sum(bool(q["shortcut_changed"]) for q in solved) / n_solved
    return result


def print_planning_stats(path: Path, stats: dict) -> None:
    print(f"=== Planning queries ({path.name}, {stats['n_total']} queries) ===")
    print(f"Solved: {stats['n_solved']} / {stats['n_total']} ({stats['success_rate']:.2f}%)")
    print()

    if stats["n_solved"] == 0:
        return

    _print_stats("RRTC planning time (ms), solved queries only", _stats(stats["planning_ms"]), " ms")
    _print_stats("RRTC iterations, solved queries only", _stats(stats["iterations"]))
    _print_stats("Shortcut path waypoints, solved queries only", _stats(stats["num_waypoints"]))
    _print_stats("External (wall-clock) duration (ms), solved queries only", _stats(stats["external_ms"]), " ms")
    _print_stats("Goal resolve time (ms), solved queries only", _stats(stats["goal_resolve_ms"]), " ms")
    print(f"Shortcut changed the path: {stats['shortcut_changed_rate']:.2f}% of solved queries")
    print()


# (key into joint/planning stats dicts, display name, value formatter, whether it's a
# median/mean/std triplet (-> "median (mean +/- std)" cell) or a single scalar)
_TABLE_ROWS = [
    ("fk_z_error_mm", "FK z-error (mm)", "{:.3f}", True),
    ("success_rate", "Success rate (\\%)", "{:.2f}", False),
    ("planning_ms", "RRTC planning time (ms)", "{:.2f}", True),
    ("iterations", "RRTC iterations", "{:.0f}", True),
    ("num_waypoints", "Shortcut waypoints", "{:.0f}", True),
    ("external_ms", "External duration (ms)", "{:.2f}", True),
    ("goal_resolve_ms", "Goal resolve time (ms)", "{:.3f}", True),
    ("shortcut_changed_rate", "Shortcut changed (\\%)", "{:.2f}", False),
]


def generate_latex_table(joint_stats: dict, planning_stats: dict, output_dir: Path) -> "tuple[Path, str]":
    """One row per metric, single "Value" column -- there's only one run here (no per-method
    comparison the way plot_bimanual_results.py's table has), so unlike that table there's
    nothing to bold. Distribution metrics (marked in _TABLE_ROWS) render as "median (mean +/-
    std)", matching that file's per-cell convention; single scalars (success rate, shortcut
    changed rate) render plain. identify_smm_and_psi_s is intentionally omitted."""
    values = {
        "fk_z_error_mm": _stats(joint_stats["error_mm"]),
        "success_rate": planning_stats["success_rate"],
        "shortcut_changed_rate": planning_stats.get("shortcut_changed_rate", float("nan")),
    }
    for key in ("planning_ms", "iterations", "num_waypoints", "external_ms", "goal_resolve_ms"):
        if key in planning_stats:
            values[key] = _stats(planning_stats[key])

    lines = []
    lines.append("% Auto-generated by analyze_maze_expt.py -- paste into your LaTeX source.")
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{FK z-error and planning statistics for the maze experiment.}")
    lines.append("\\label{tab:maze_expt_results}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Metric & Value \\\\")
    lines.append("\\midrule")

    for key, display_name, fmt, is_triplet in _TABLE_ROWS:
        if key not in values:
            continue
        v = values[key]
        if is_triplet:
            cell = f"{fmt.format(v['median'])} ({fmt.format(v['mean'])} $\\pm$ {fmt.format(v['std'])})"
        else:
            cell = fmt.format(v)
        lines.append(f"{display_name} & {cell} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_tex = "\n".join(lines) + "\n"
    path = output_dir / "results_table.tex"
    path.write_text(table_tex)
    return path, table_tex


def plot_z_error_running(joint_stats: dict, output_dir: Path) -> Path:
    """FK z-error (mm) over the course of the recorded trajectory -- a running plot catches
    drift/outliers/localized bad stretches that the aggregate mean/std in the table would
    average away."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4 * (3 / 4)))

    ax.axhline(0.0, color="0.6", linestyle="--", linewidth=1.1, zorder=1)
    ax.plot(joint_stats["t_relative"], joint_stats["error_mm"], color=sns.color_palette("deep")[0],
            linewidth=1.2, zorder=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("FK z-error (mm)")
    ax.set_title(f"FK z-error over trajectory (expected z = {EXPECTED_EEF_Z} m)")
    fig.tight_layout()

    path = output_dir / "z_error_running.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def plot_planning_time_ecdf(planning_stats: dict, output_dir: Path) -> Optional[Path]:
    """Empirical CDF of RRTC planning time (ms) over solved queries -- shows the full time
    distribution (including tail behavior) that a single mean/median can't, same rationale as
    plot_bimanual_results.py's plot_ecdf."""
    if "planning_ms" not in planning_stats or planning_stats["planning_ms"].size == 0:
        return None

    values = np.sort(planning_stats["planning_ms"])
    fractions = np.arange(1, values.size + 1) / values.size
    color = sns.color_palette("deep")[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.4 * (3 / 4)))
    ax.axhline(0.5, color="0.82", linestyle="--", linewidth=1.1, zorder=1)
    ax.fill_between(values, fractions, step="post", color=color, alpha=0.18, zorder=2)
    ax.step(values, fractions, where="post", color=color, linewidth=2.2, zorder=3)

    median = float(np.median(values))
    ax.plot(median, 0.5, marker="o", markersize=7.5, color=color, markeredgecolor="white",
            markeredgewidth=1.3, zorder=4)
    ax.annotate(f"median = {median:.2f} ms", (median, 0.5), textcoords="offset points",
                xytext=(8, 8), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("RRTC planning time (ms)")
    ax.set_ylabel("Fraction of solved queries")
    ax.set_title("Planning time ECDF")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()

    path = output_dir / "planning_time_ecdf.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def main(
    joint_states_csv: str = str(DEFAULT_JOINT_STATES_CSV),
    planning_queries_jsonl: str = str(DEFAULT_PLANNING_QUERIES_JSONL),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> None:
    _set_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(joint_states_csv)
    joint_stats = compute_joint_state_stats(df)
    print_joint_state_stats(Path(joint_states_csv), joint_stats, len(df))

    queries = _load_jsonl(Path(planning_queries_jsonl))
    planning_stats = compute_planning_stats(queries)
    print_planning_stats(Path(planning_queries_jsonl), planning_stats)

    table_path, table_tex = generate_latex_table(joint_stats, planning_stats, output_path)
    z_error_path = plot_z_error_running(joint_stats, output_path)
    ecdf_path = plot_planning_time_ecdf(planning_stats, output_path)

    print(f"Saved: {table_path}")
    print(f"Saved: {z_error_path} (+ .svg)")
    if ecdf_path is not None:
        print(f"Saved: {ecdf_path} (+ .svg)")
    print()
    print(table_tex)


if __name__ == "__main__":
    Fire(main)
