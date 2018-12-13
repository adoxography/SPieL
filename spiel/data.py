"""
spiel.data
"""
from spiel.util import grouper


class Instance:
    def __init__(self, shape, segments, labels):
        self.shape = shape
        self.segments = segments
        self.labels = labels

    @property
    def annotations(self):
        return list(zip(self.segments, self.labels))

    def __eq__(self, other):
        return self.shape == other.shape and \
               self.segments == other.segments and \
               self.labels == other.labels


def load(file_name):
    instances = []

    with open(file_name) as f:
        for shape, segments, labels, _ in grouper(4, f):
            shape = shape.strip()
            segments = segments.split()
            labels = labels.split()
            instances.append(Instance(shape, segments, labels))

    return instances
