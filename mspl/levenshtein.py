"""
mspl.levenshtein

Implementation of Levenshtein distance and related functionality. This module
is designed to accept any list of comparables instead of just strings, create
lists of operations that are independent of the contexts from which they were
generated, and then apply those operations using arbitrary reference points
instead of the original target.
"""
from abc import ABCMeta, abstractmethod
import re
import numpy as np
from mspl.util import flatten

INSERT_SYMBOL = 'I'
DELETE_SYMBOL = 'D'
REPLACE_SYMBOL = 'R'


def distance(origin, target):
    """
    Finds the Levenshtein distance between two iterables

    :param origin: The iterable to start from
    :type origin: str or list of str
    :param target: The iterable to end at
    :type target: str or list of str
    :return: The minimum edit distance between the two iterables
    :rtype: int
    """
    matrix = __build_matrix(origin, target)
    return matrix[-1, -1]


def operations(origin, target):
    """
    Finds the operations needed to achieve the minimum edit distance between
    two iterables

    :param origin: The iterable to start from
    :type origin: str or list of str
    :param target: The iterable to end at
    :type target: str or list of str
    :return: The operations used to get from the origin to the target
    :rtype: generator of Operation
    """
    matrix = __build_matrix(origin, target)
    i, j = [x-1 for x in matrix.shape]
    curr = matrix[i, j]

    while curr > 0:
        if origin[i-1] == target[j-1]:
            # If the origin and target characters are identical, there is no
            # operation. Just move up diagonally and carry on.
            new_i, new_j = i-1, j-1
        else:
            options = [(i-1, j), (i-1, j-1), (i, j-1)]
            new_i, new_j = min(options, key=lambda pos: matrix[pos[0], pos[1]])

            if new_i == i-1 and new_j == j-1:
                yield ReplaceOperation(i-1, j-1)
            elif new_i == i-1:
                yield DeleteOperation(i-1)
            else:
                yield InsertOperation(i-1, j-1)

        i, j = new_i, new_j
        curr = matrix[i, j]


def apply_operations(origin, reference, ops):
    """
    Applies a list of operations to a string, using a reference for where the
    operation information should come from

    :param origin: The iterable to start from
    :type origin: str or list of str
    :param reference: An iterable that is the same length as the original
                      target, from which the operations will reference for new
                      characters
    :type reference: str or list of str
    :return: The origin iterable, modified by the operations
    :rtype: Same as origin
    """
    string_mode = isinstance(origin, str)
    origin = list(origin)

    for oper in ops:
        oper.apply(origin, reference)

    origin = [x for x in flatten(origin) if x]
    if string_mode:
        origin = ''.join(origin)

    return origin


def annotate(origin, reference, ops, annotation=None):
    """
    Uses a list of operations to create an annotation for how the operations
    should be applied, using an origin and a reference target

    :param origin: The original iterable
    :type origin: str or list
    :param reference: The original target
    :type reference: str or list
    :ops: The operations to apply
    :type ops: list of Operation
    :param annotation: An optional initialization for the annotation. If none
                       is provided, the annotation will be based on the
                       origin. If one is provided, it will be modified by this
                       method.
    :return: An annotation based on the operations
    :rtype: list of str
    """
    annotation = annotation or list(origin)

    for oper in ops:
        oper.annotate(annotation, origin, reference)

    for i, label in reversed(list(enumerate(annotation))):
        if label[0] == '+' and i > 0:
            annotation[i-1] += label
            del annotation[i]

    return annotation


def __build_matrix(origin, target):
    """
    Builds a Levenshtein matrix based on two iterables

    :param origin: The iterable to start from
    :type origin: str or list of str
    :param target: The iterable to end at
    :type target: str or list of str
    :return: The corresponding Levenshtein matrix
    :rtype: np.array
    """
    width = len(origin) + 1
    height = len(target) + 1

    matrix = np.zeros((width, height), dtype=int)

    # Fill in the empty row and column
    for i in range(width):
        matrix[i][0] = i
    for i in range(1, height):
        matrix[0][i] = i

    # Fill in the rest of the Levenshtein matrix
    for i in range(1, width):
        for j in range(1, height):
            sub_cost = 0 if origin[i-1] == target[j-1] else 1
            options = [
                matrix[i-1, j] + 1,          # deletion
                matrix[i, j-1] + 1,          # insertion
                matrix[i-1, j-1] + sub_cost  # substitution
            ]
            best_option = min(options)

            matrix[i, j] = best_option

    return matrix


class Operation(metaclass=ABCMeta):
    """
    Abstract base class for all Operations

    Defines the interface, that all Operations need an apply() method
    """
    def __init__(self, name, origin_pos, target_pos):
        """
        Initializes the Operation

        :param name: The name of the operation
        :type name: str
        """
        self.type = name
        self.origin_pos = origin_pos
        self.target_pos = target_pos
        super().__init__()

    @abstractmethod
    def apply(self, origin, reference=None):
        """
        Applies the operation to a list, using another iterable as a reference

        :param origin: The list to apply the operation to
        :type origin: list
        :param reference: The reference iterable for insertions and
                          replacements
        :type reference: list or str
        """

    @abstractmethod
    def annotate(self, annotation, origin, reference):
        """
        Marks *annotation* as being modified by this operation

        :param annotation: The annotation to mark
        :type annotation: list of str
        :param origin: The original iterable
        :type origin: list or str
        :param reference: The reference that this operation should modify
                          toward
        :type reference: list or str
        """

    def __eq__(self, other):
        return self.type == other.type \
            and self.origin_pos == other.origin_pos \
            and self.target_pos == other.target_pos


class ReplaceOperation(Operation):
    """
    Defines an operation where one character is replaced with another
    """
    def __init__(self, origin_pos, target_pos):
        """
        Initializes the Operation

        :param origin_pos: The position of the character in the origin string
                           that should be replaced
        :type origin_pos: int
        :param target_pos: The position of the character of the target string
                           that the character should be replaced with
        :type target_pos: int
        """
        super().__init__('REPLACE', origin_pos, target_pos)

    def apply(self, origin, reference):
        """
        Replaces the character at *origin_pos* in *origin* with the character
        at *target_pos* from *reference*

        :param origin: The list to alter
        :type origin: list
        :param reference: The list to reference in the operation
        :type reference: str or list
        """
        index = self.origin_pos
        while isinstance(origin[index], list):
            origin = origin[index]
            index = 0
        origin[index] = reference[self.target_pos]

    def annotate(self, annotation, origin, reference):
        """
        Marks *annotation* as being modified by this operation

        :param annotation: The annotation to mark
        :type annotation: list of str
        :param origin: The original iterable
        :type origin: list or str
        :param reference: The reference that this operation should modify
                          toward
        :type reference: list or str
        """
        origin_str = origin[self.origin_pos]
        target_str = reference[self.target_pos]
        annotation[self.origin_pos] += f"+{REPLACE_SYMBOL}({origin_str},{target_str})"

    def __repr__(self):
        return f"Replace {self.origin_pos} with {self.target_pos}"


class InsertOperation(Operation):
    """
    Defines an operation where a character is inserted after a certain position
    """
    def __init__(self, origin_pos, target_pos):
        """
        Initializes the Operation

        :param origin_pos: The position of the character in the origin string
                           that should be inserted after
        :type origin_pos: int
        :param target_pos: The position of the character of the target string
                           that the character should be replaced with
        :type target_pos: int
        """
        super().__init__('INSERT', origin_pos, target_pos)

    def apply(self, origin, reference):
        """
        Inserts the character at *target_pos* in *reference* after *origin_pos*
        in *origin*

        :param origin: The list to alter
        :type origin: list
        :param reference: The list to reference in the operation
        :type reference: str or list
        """
        index = self.origin_pos
        while isinstance(origin[index], list):
            origin = origin[index]
            index = 0
        origin[index] = [origin[index],
                         reference[self.target_pos]]

    def annotate(self, annotation, origin, reference):
        """
        Marks *annotation* as being modified by this operation

        :param annotation: The annotation to mark
        :type annotation: list of str
        :param origin: The original iterable
        :type origin: list or str
        :param reference: The reference that this operation should modify
                          toward
        :type reference: list or str
        """
        target_str = reference[self.target_pos]
        annotation[self.origin_pos] += f"+{INSERT_SYMBOL}({target_str})"

    def __repr__(self):
        return f"Insert {self.target_pos} at position {self.origin_pos}"


class DeleteOperation(Operation):
    """
    Defines an operation where a character is deleted at a certain position
    """
    def __init__(self, origin_pos):
        """
        Initializes the Operation

        :param origin_pos: The position of the character in the origin string
                           that should be deleted
        :type origin_pos: int
        """
        super().__init__('DELETE', origin_pos, None)

    def apply(self, origin, reference=None):
        """
        Deletes the character at *origin_pos* in *origin*

        :param origin: The list to alter
        :type origin: list
        :param reference: Dummy parameter to conform to the Operation interface
        :type reference: any
        """
        index = self.origin_pos
        while isinstance(origin[index], list):
            origin = origin[index]
            index = 0
        origin[index] = None

    def annotate(self, annotation, origin, reference):
        """
        Marks *annotation* as being modified by this operation

        :param annotation: The annotation to mark
        :type annotation: list of str
        :param origin: The original iterable
        :type origin: list or str
        :param reference: The reference that this operation should modify
                          toward
        :type reference: list or str
        """
        origin_str = origin[self.origin_pos]
        match = re.search(r'\+.*$', annotation[self.origin_pos])
        if match:
            tags = match[0]
        else:
            tags = ''
        annotation[self.origin_pos] = f"+{DELETE_SYMBOL}({origin_str}){tags}"

    def __repr__(self):
        return f"Delete {self.origin_pos}"
