from __future__ import annotations

import multiprocessing as mp
import os

_SINGLE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _call(item):
    i, fn, arg = item
    return i, fn(arg)


def run_tasks(fn, tasks, workers: int, tick: float = 1.0, initializer=None, initargs=()):
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        for i, t in enumerate(tasks):
            yield i, fn(t)
        return
    old = {k: os.environ.get(k) for k in _SINGLE_THREAD}
    for k in _SINGLE_THREAD:
        os.environ.setdefault(k, "1")
    try:
        with mp.get_context("spawn").Pool(min(workers, len(tasks)), initializer, initargs) as pool:
            it = pool.imap_unordered(_call, [(i, fn, t) for i, t in enumerate(tasks)])
            for _ in range(len(tasks)):
                while True:
                    try:
                        yield it.next(timeout=tick)
                        break
                    except mp.TimeoutError:
                        yield None, None
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
