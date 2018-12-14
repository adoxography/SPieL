"""
spiel.util

Utility methods for SPieL
"""
import collections
from itertools import zip_longest


def flatten(lst):
    """
    From https://stackoverflow.com/a/2158532
    """
    for elm in lst:
        if isinstance(elm, collections.Iterable) \
                and not isinstance(elm, (str, bytes)):
            yield from flatten(elm)
        else:
            yield elm


def grouper(per_slice, iterable, fillvalue=None):
    """
    Iterates over an iterable in slices, like Ruby's Enumerable#each_slice

    see https://docs.python.org/3/library/itertools.html#recipes
    """
    args = [iter(iterable)] * per_slice
    return zip_longest(*args, fillvalue=fillvalue)


def all_permutations(options):
    """
    Generates a list of all possible permutations of a list of options
    """
    solutions = [[]]

    for option_set in options:
        solutions = [item + [option]
                     for item in solutions
                     for option in option_set]

    return solutions


def pad(item, char, size):
    """
    Adds *size* *char*s to either side of *item*

    :param item: The base item to pad
    :type item: list or str
    :param char: The char to pad with
    :type char: str
    :param size: The number of *chars* to put on either end
    :type size: int
    :return: The padded item
    :rtype: Same as *item*
    """
    if isinstance(item, str):
        padding = char * size
    else:
        padding = [char] * size
    return padding + item + padding
