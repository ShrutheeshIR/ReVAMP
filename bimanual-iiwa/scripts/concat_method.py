"""Concatenate one method's 6 segment renders (in common.SEGMENTS order)
into out/<method>_all.mp4. Called by build_video.py as its own stage."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    method = sys.argv[1]
    seg_outs = [os.path.join(common.OUT_DIR,
                             f"{method.lower()}_{s.replace('->', '_to_')}.mp4")
               for s in common.SEGMENTS]
    for p in seg_outs:
        if not os.path.exists(p):
            sys.exit(f"missing segment render: {p}")

    os.makedirs(common.SCRATCH, exist_ok=True)
    list_path = os.path.join(common.SCRATCH, f"{method.lower()}_concat.txt")
    with open(list_path, "w") as f:
        for p in seg_outs:
            f.write(f"file '{os.path.abspath(p)}'\n")

    out = os.path.join(common.OUT_DIR, f"{method.lower()}_all.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", list_path, "-c", "copy", out], check=True)
    print(out)


if __name__ == "__main__":
    main()
