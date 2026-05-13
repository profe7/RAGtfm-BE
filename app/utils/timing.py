from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed_stage(metrics: dict, name: str):
    start = perf_counter()

    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        metrics[name] = round(elapsed_ms, 2)
