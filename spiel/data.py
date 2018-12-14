"""
spiel.data

Module for organizing input data to the SPieL system
"""


class ParseError(ValueError):
    """ Raised by bad values for a new instance """


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
        self.shape = shape
        self.segments = segments
        self.labels = labels

    @staticmethod
    def fit(lines, strict):
        try:
            shape, segments, labels = lines
        except ValueError:
            if len(lines) < 3:
                if strict:
                    raise ParseError("Not enough fields provided")
                else:
                    shape, segments, labels = lines[0], None, None
            else:
                raise ParseError("Too many fields provided")

        if (segments or labels) and not len(segments) == len(labels):
            raise ParseError(f"Number of segments must match number of \
    labels; got segments '{segments}' segments, but labels '{labels}'.")

        return Instance(''.join(shape), segments, labels)

    @property
    def annotations(self):
        """
        Returns the combination of the instance's segments and labels

        :rtype: list of (str, str)
        """
        return list(zip(self.segments, self.labels))

    def annotation_string(self):
        """
        Returns a string representation of the instance's segments and labels
        """
        return '-'.join([f"{segment}/{label}"
                         for segment, label in self.annotations])

    def __eq__(self, other):
        return self.shape == other.shape and \
               self.segments == other.segments and \
               self.labels == other.labels


def load_file(file_name, strict=True):
    """
    Loads a list of instances from a file
    """
    with open(file_name) as f:
        return load(f, strict)


def load(lines, strict=True):
    """
    Loads a list of instances from a list of lines

    Lines must be ordered as follows:

    Line 1*n: Shape
    Line 2*n: Segments
    Line 3*n: Labels
    Line 4*n: blank

    :return: A list of training instances
    :rtype: list of Instance
    """
    instances = []

    data = []

    for line in lines:
        line = line.strip()
        if len(line) == 0:
            if data:
                instances.append(Instance.fit(data, strict))
                data = []
        else:
            data.append(line.split())

    if data:
        instances.append(Instance.fit(data, strict))

    return instances
