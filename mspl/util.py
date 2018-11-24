from collections import Iterable


def flatten(lst):
    """
    From https://stackoverflow.com/a/2158532
    """
    for el in lst:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
