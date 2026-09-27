# Environment for scripts/figures/ -- Drake -> Meshcat StaticHtml() -> Blender
# (meshcat_html_importer) -> Cycles -> ffmpeg, per ../../FIGURES_HOWTO.md (setup) and
# ./README.md (pipeline design/defaults) at the repo root / in this directory.
#
# This image only sets up the environment (Blender 5.0.1 + the meshcat_html_importer
# add-on, ffmpeg, Python 3.12 + pydrake/numpy/matplotlib). It is deliberately NOT meant to
# be run directly -- there is no CMD/ENTRYPOINT that executes any part of the pipeline.
# Bind-mount the repo, drop into the shell it gives you, and run scripts/figures/*.py
# yourself:
#
#   docker build -f scripts/figures/Dockerfile -t pvamp-figures .
#   docker run --rm -it --gpus all \
#     -v $(pwd):/workspace \
#     -v /usr/share/nvidia:/usr/share/nvidia:ro \
#     pvamp-figures
#   # now inside the container, from /workspace (the repo root):
#   python3 scripts/figures/check_meshcat_importer.py --probe
#   python3 scripts/figures/maze_manifest.py --selftest
#   python3 scripts/figures/export_maze_scene.py --trajectory out/plans/problem_108.txt
#   python3 scripts/figures/render_still.py --mode static
#   python3 scripts/figures/render_video.py --preview-frames 24
#
# Drop --gpus all if the host has no NVIDIA GPU + nvidia-container-toolkit -- Cycles
# falls back to CPU (same picture, ~20x slower; see FIGURES_HOWTO.md Sec.0). Trajectories
# (out/plans/) are gitignored and not baked into this image -- supply them via the
# bind mount, same as running the pipeline outside Docker.
#
# The /usr/share/nvidia bind mount above is required for the OptiX *denoiser*
# specifically (separate from OptiX rendering itself): it ships nvoptix.bin, a ~60MB
# neural-net weights file installed alongside the host's NVIDIA driver, versioned to
# match it exactly. The NVIDIA container runtime mounts driver *libraries* (libcuda,
# libnvoptix.so, etc. -- see NVIDIA_DRIVER_CAPABILITIES below) automatically, but not
# this data file, so without the bind mount Cycles fails with "Unable to load denoiser
# weights" / "Failed to create OptiX denoiser" even though OptiX rendering itself works.
# Confirmed present at that path on a driver-580.142 host; adjust if yours differs.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Without this, the NVIDIA container runtime only mounts the `compute,utility` library
# subset into the container (enough for nvidia-smi and CUDA) -- not `graphics`, which is
# where libnvoptix.so.1 / libnvidia-rtcore.so live. Missing those makes Cycles fail with
# "Error: Failed to create OptiX denoiser" even though nvidia-smi and plain CUDA work fine.
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Blender's runtime deps -- it dlopens these X11/GL libs even for headless/offscreen
# Cycles renders -- plus ffmpeg (the video encode step), and wget/xz-utils/git/
# ca-certificates to fetch Blender and the add-on below.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      ffmpeg \
      wget \
      xz-utils \
      git \
      ca-certificates \
      libx11-6 \
      libxi6 \
      libxrender1 \
      libxxf86vm1 \
      libxfixes3 \
      libxkbcommon0 \
      libxrandr2 \
      libsm6 \
      libice6 \
      libgl1 \
      libegl1 \
    && rm -rf /var/lib/apt/lists/*

# --- 1. Blender 5.0.1 --------------------------------------------------------------
# Required exactly, not merely preferred: meshcat_html_importer declares
# blender_version_min = "5.0.0", and 5.0 changed two APIs the pipeline touches (actions
# became slotted, and the COLLADA importer was removed). Portable tarball, not a distro
# package (usually older) -- see FIGURES_HOWTO.md Sec.1.
ENV BLENDER_VERSION=5.0.1
ENV BLENDER=/opt/blender-5.0.1-linux-x64/blender
RUN wget -q "https://download.blender.org/release/Blender5.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz" -O /tmp/blender.tar.xz \
    && tar -xf /tmp/blender.tar.xz -C /opt \
    && rm /tmp/blender.tar.xz \
    && "${BLENDER}" --version

# --- 2. meshcat_html_importer add-on -------------------------------------------------
# Not bundled with PVAMP -- Nicholas Pfaff's, BSD-2-Clause, pinned at tag v0.1.3 (see
# FIGURES_HOWTO.md Sec.2). Blender loads user extensions from a fixed config directory,
# so the add-on tree has to be copied there rather than just cloned somewhere convenient.
# Verify with `python3 scripts/figures/check_meshcat_importer.py --probe` after entering
# the container -- a missing/broken add-on does not raise, it renders a clean picture of
# an empty scene and exits 0.
RUN git clone --depth 1 --branch v0.1.3 \
      https://github.com/nepfaff/drake-blender-recorder.git /tmp/dbr \
    && mkdir -p /root/.config/blender/5.0/extensions/user_default \
    && cp -r /tmp/dbr/blender_addons/meshcat_html_importer \
          /root/.config/blender/5.0/extensions/user_default/ \
    && rm -rf /tmp/dbr

# --- 3. Python + Drake ---------------------------------------------------------------
# Same pinned nightly as fr3_trajopt/ (bump deliberately, not lazily -- see that
# requirements.txt's own comment). Ubuntu 24.04's system python3 is 3.12, which is what
# this Drake wheel needs; --break-system-packages sidesteps PEP 668 since the whole
# container is this environment, no venv needed.
COPY fr3_trajopt/requirements.txt /tmp/fr3_trajopt_requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
      --extra-index-url https://drake-packages.csail.mit.edu/whl/nightly \
      -r /tmp/fr3_trajopt_requirements.txt \
    && rm /tmp/fr3_trajopt_requirements.txt

# The repo (this Dockerfile's build context) is bind-mounted here at `docker run` time,
# not copied in -- out/, out/plans/ and any local edits under scripts/figures/ all need
# to be visible without a rebuild.
WORKDIR /workspace

CMD ["/bin/bash"]
