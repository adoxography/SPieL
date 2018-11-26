from collections import Iterable


def flatten(lst):
    """
    From https://stackoverflow.com/a/2158532
    """
    for elm in lst:
        if isinstance(elm, Iterable) and not isinstance(elm, (str, bytes)):
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
