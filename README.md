# ReVAMP

Paper: https://arxiv.org/abs/2609.30213
Video: https://youtu.be/A4aWkLMXusM

## Running the experiments

Experiment scripts, resources, and their CMake build live under `src/experiments/`.

```bash
# One-time: configure the build (also needed after CMakeLists.txt changes)
cmake -S src/experiments -B src/experiments/build

# Build one target
cmake --build src/experiments/build --target <target-name> -j"$(nproc)"

# Run it
./src/experiments/build/<target-name> [args...]
```

Some benchmarks are also wired up in `src/experiments/run_benchmarks.py` (`--list` to see available targets), which builds and runs a target and versions its output under `src/experiments/results/<run-id>/`.
