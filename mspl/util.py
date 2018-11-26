"""
mspl.util

Utility methods for MSPL
"""
import collections


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
