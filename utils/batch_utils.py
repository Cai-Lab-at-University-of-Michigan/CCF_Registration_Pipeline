def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def step(total, workers):
    if total % workers == 0:
        s = total // workers
    else:
        s = total // workers + 1

    return s
