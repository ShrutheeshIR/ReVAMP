"""Render an ordered frame sequence across processes.

Per-frame compositing (open an image, draw a plot panel and some text on it,
hand the pixels to an encoder) is pure CPU work and is embarrassingly
parallel: every frame is an independent function of its index. Copy this file
into your project and adjust ``frame_workers`` / the self-test as needed; it
has no dependencies beyond the standard library and numpy.

Usage mirrors the serial loop it replaces::

    frames = parallel_frames(lambda: (lambda i: render_one(i)), n_frames)
    for arr in frames:
        ffmpeg_stdin.write(arr.tobytes())

``make_renderer`` is called *once per worker* and returns the per-frame
function ``render(i) -> np.ndarray``. That indirection exists because the
per-frame work usually reuses expensive state -- a matplotlib Figure, loaded
fonts, a decoded mesh -- which must not be shared across processes (most such
objects are not picklable, and some are not safe to touch from two processes
at once) but also should not be rebuilt on every single frame.

Two properties are load-bearing:

* **Order is preserved.** Blocks are drained in submission order, so the
  caller sees frame i before frame i+1. A frame sequence delivered out of
  order is not a crash -- it is a silently scrambled video, and nothing
  downstream will detect it.
* **The pool is forked eagerly, before this function returns.** If the
  caller's next step is to open a pipe to an external encoder (e.g.
  ``subprocess.Popen(["ffmpeg", ...], stdin=PIPE)``), that pipe must be
  opened *after* this call, not before. A worker process forked after the
  pipe is opened inherits the write end of that pipe as an open file
  descriptor; the encoder then never sees EOF on stdin even after the
  intended writer closes it, because the child still holds it open, and the
  encoder hangs forever waiting for more input. Do not turn this function
  into a generator that defers the fork to first iteration -- that silently
  reintroduces the same bug because the pool would then fork lazily, after
  the caller has already gone on to open its pipe.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

# Set in the parent immediately before the pool forks, and read in the child
# through fork inheritance. Passing the factory as an argument to the pool
# instead would require it to be picklable, which rules out exactly the
# closures over local state that every caller of this module wants to use.
_MAKE_RENDERER = None
_RENDER = None


def frame_workers(cap=None):
    """How many worker processes to render with.

    ``FRAME_JOBS`` overrides. Otherwise one per hardware thread less one, so
    the machine keeps a thread free for everything else -- in particular for
    whatever is consuming these frames (an encoder process), which must not
    be starved by the workers producing them.
    """
    env = os.environ.get("FRAME_JOBS", "").strip()
    if env:
        return max(1, int(env))
    n = max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(n, cap)) if cap else n


def _init():
    global _RENDER
    _RENDER = _MAKE_RENDERER()


def _render_block(bounds):
    start, stop = bounds
    return [np.ascontiguousarray(_RENDER(i), dtype=np.uint8)
            for i in range(start, stop)]


def parallel_frames(make_renderer, n_frames, *, workers=None, block=2,
                     progress_every=60, label="rendering"):
    """Iterator over frames ``0..n_frames-1``, rendered across processes, IN ORDER.

    ``make_renderer`` -- zero-arg factory, called once per worker, returning
    ``render(i) -> np.ndarray``. See the module docstring for why this
    indirection exists and why the two load-bearing properties (order,
    eager fork) must not be broken.

    ``block`` frames are rendered per task, and at most ``3 * workers`` tasks
    are kept in flight at once. That bounds how many pixels are held in
    memory at a time -- a single 1080p RGB frame is about 6 MB, so the
    default window is a few hundred MB regardless of ``n_frames`` -- while
    still keeping every worker fed with the next task before it finishes the
    current one.

    Every yielded array is coerced with ``np.ascontiguousarray(..., dtype=
    np.uint8)`` so callers can call ``.tobytes()`` on it directly (e.g. to
    write to an encoder's stdin) without worrying about dtype or a
    non-contiguous view from whatever drawing library produced it.

    Falls back to a plain serial generator when ``workers == 1`` or when
    ``n_frames`` is too small to be worth the process-pool overhead (at most
    ``block`` frames). The serial path calls ``make_renderer`` directly in
    the caller's own process -- no fork happens in that case, so the
    eager-fork guarantee is trivially satisfied (there is no pool to worry
    about inheriting anything).
    """
    global _MAKE_RENDERER

    workers = workers or frame_workers()
    if workers == 1 or n_frames <= block:
        render = make_renderer()

        def serial():
            for i in range(n_frames):
                if progress_every and i % progress_every == 0:
                    print(f"  {label} frame {i}/{n_frames}", flush=True)
                yield np.ascontiguousarray(render(i), dtype=np.uint8)

        return serial()

    bounds = [(s, min(s + block, n_frames)) for s in range(0, n_frames, block)]
    _MAKE_RENDERER = make_renderer
    # fork, not spawn: the child inherits the factory above (and any other
    # module-level state the caller has already built, e.g. a loaded font or
    # a precomputed camera) without any of it needing to be picklable. This
    # is also *why* the pool must be created here, synchronously, before this
    # function returns control to the caller -- see the module docstring.
    pool = mp.get_context("fork").Pool(workers, initializer=_init)
    print(f"  {label} {n_frames} frames across {workers} processes", flush=True)

    def drain():
        pending = []
        emitted = 0
        try:
            for b in bounds:
                pending.append(pool.apply_async(_render_block, (b,)))
                while len(pending) >= 3 * workers:
                    # Strictly FIFO: pop the OLDEST outstanding task and block
                    # on it, never whichever finishes first. That is what
                    # keeps frames leaving this generator in submission
                    # order even though the workers themselves finish their
                    # blocks in whatever order the OS schedules them.
                    for arr in pending.pop(0).get():
                        if progress_every and emitted % progress_every == 0:
                            print(f"  {label} frame {emitted}/{n_frames}",
                                  flush=True)
                        emitted += 1
                        yield arr
            while pending:
                for arr in pending.pop(0).get():
                    if progress_every and emitted % progress_every == 0:
                        print(f"  {label} frame {emitted}/{n_frames}", flush=True)
                    emitted += 1
                    yield arr
        finally:
            pool.terminate()
            pool.join()

    return drain()


if __name__ == "__main__":
    # Self-test: a trivial renderer that encodes its own frame index into
    # every pixel, run through the parallel path, and checked to come back
    # in exactly submission order. This is the property that matters most in
    # this module -- silently scrambled order does not raise, it just ships
    # a broken video -- so it is the one thing this self-test asserts.
    N = 47

    def make_renderer():
        # Deliberately cheap per-worker "expensive state" stand-in, to show
        # where a real caller would build a Figure or load a font once.
        offset = 0

        def render(i):
            return np.full((4, 4, 3), fill_value=(i + offset) % 256,
                            dtype=np.uint8)

        return render

    print(f"[self-test] parallel_frames, workers={frame_workers()}, n={N}")
    frames = list(parallel_frames(make_renderer, N, block=3,
                                   progress_every=10, label="self-test"))

    assert len(frames) == N, f"expected {N} frames, got {len(frames)}"
    for i, arr in enumerate(frames):
        assert arr.shape == (4, 4, 3), f"frame {i} has shape {arr.shape}"
        assert arr.dtype == np.uint8, f"frame {i} has dtype {arr.dtype}"
        expected = i % 256
        got = int(arr[0, 0, 0])
        assert got == expected, (
            f"frame at position {i} carries payload {got}, expected "
            f"{expected} -- frames were delivered out of order")

    # Also exercise the forced-serial path and confirm it agrees.
    serial_frames = list(parallel_frames(make_renderer, N, workers=1,
                                          progress_every=0))
    assert len(serial_frames) == N
    for i, (a, b) in enumerate(zip(frames, serial_frames)):
        assert np.array_equal(a, b), f"serial/parallel mismatch at frame {i}"

    print(f"[self-test] OK: {N} frames, parallel and serial paths agree, "
          "order preserved")
