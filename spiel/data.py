"""
spiel.data

Module for organizing input data to the SPieL system
"""
from spiel.util import grouper


class Instance:
    """
    Represents a single end-to-end training instance
    """
    def __init__(self, shape, segments, labels):
        """
        Initializes the instance

        :param shape: The written shape of the instance
        :type shape: str
        :param segments: The segments that make up the shape
        :type segments: list of str
        :param labels: The labels for each segment
        :type labels: list of str
        """
        if len(shape) == 0:
            raise ValueError(f"Shape cannot be empty. (Segments: {segments})")

        if len(segments) == 0:
            raise ValueError(f"Segments cannot be empty. (Shape: {shape})")

        if not len(segments) == len(labels):
            raise ValueError(f"Number of segments must match number of \
labels; got {len(segments)} segments, but {len(labels)} labels.")

        self.shape = shape
        self.segments = segments
        self.labels = labels

    @property
    def annotations(self):
        """
        Returns the combination of the instance's segments and labels

        :rtype: list of (str, str)
        """
        return list(zip(self.segments, self.labels))

    def __eq__(self, other):
        return self.shape == other.shape and \
               self.segments == other.segments and \
               self.labels == other.labels


def load(file_name):
    """
    Loads a list of instances from a file

    Instances must be ordered as follows:

    Line 1*n: Shape
    Line 2*n: Segments
    Line 3*n: Labels
    Line 4*n: ignored

    :return: A list of training instances
    :rtype: list of Instance
    """
    instances = []

    with open(file_name) as f:
        for shape, segments, labels, _ in grouper(4, f):
            shape = shape.strip()
            segments = segments.split()
            labels = labels.split()
            instances.append(Instance(shape, segments, labels))

    return instances
