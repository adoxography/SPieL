"""
segmentation.py

Contains logic for segmenting shapes into morphemes
"""
import re
from mspl import levenshtein
from mspl.levenshtein import INSERT_SYMBOL


class Featurizer:
    """
    Used to convert basic instances into training instances for a classifier
    """
    def __init__(self, inside_label='I', tokenize=None):
        """
        Initializes the featurizer

        :param inside_label: The label to use for a character that does not
                             begin a label
        :type inside_label: str
        :param tokenize: A function to tokenize incoming strings. Defaults to
                         list()
        :type tokenize: callable
        """
        self.inside_label = inside_label
        self.tokenize = tokenize or list

    def convert(self, shape, labels):
        """
        Converts a shape and corresponding labels into training instances

        :param shape: The shape to convert
        :type shape: list or str
        :param labels: The labels for each token in the shape
        :type labels: list
        :return: A list of training instances, where the first element is a
                 dict of features, and the second is a trigram label for the
                 instance
        :rtype: (dict, str)
        """
        if isinstance(shape, str):
            shape = self.tokenize(shape)

        if not len(shape) == len(labels):
            raise RuntimeError

        padded_shape = pad(shape, '_', 4)
        padded_labels = pad(labels, '_', 4)
        instances = []

        for i in range(3, len(padded_shape) - 3):
            features = {
                'prefix': ''.join(padded_shape[i-3:i]),
                'focus': padded_shape[i],
                'suffix': ''.join(padded_shape[i+1:i+4])
            }

            labels = '+'.join(padded_labels[i-1:i+2])

            instances.append((features, labels))

        return instances

    def label(self, shape, annotations):
        """
        Generates a list of annotations corresponding to each element in
        *shape*

        :param shape: The sequence to annotate
        :type shape: list or str
        :param annotations: A list of tuples, where the first is the shape of
                            the annotations, and the second is its label
        :type annotations: list of (str, str)
        :return: A list of labels
        :rtype: list
        """
        if isinstance(shape, str):
            shape = self.tokenize(shape)
        annotation_string = self.tokenize(concat_annotations(annotations))
        labels = label_annotations(annotations, self.inside_label,
                                   self.tokenize)

        ops = levenshtein.operations(annotation_string, shape)
        labels = levenshtein.annotate(annotation_string, shape, ops, labels)

        for i, label in enumerate(labels):
            matches = re.findall(rf'\+{INSERT_SYMBOL}(.*?)', label)
            for _ in matches:
                labels.insert(i+1, self.inside_label)

        return labels


def concat_annotations(annotations):
    """
    Glues the shapes of a list of annotations together

    :param annotations: A list of tuples, where the first is the shape of the
                        annotations, and the second is its label
    :type annotations: list of (str, str)
    :return: The combined shape of the annotations
    :rtype: str
    """
    return ''.join([shape for shape, _ in annotations])


def label_annotations(annotations, inside_label, tokenize=None):
    """
    Creates a list of labels for each character in a list of annotations

    :param annotations: A list of tuples, where the first is the shape of the
                        annotations, and the second is its label
    :type annotations: list of (str, str)
    :param inside_label: The label to use for characters that do not begin an
                         annotation
    :param tokenize: The function that should be used to tokenize the
                     annotation shapes. Defaults to calling list()
    :type tokenize: callable
    :return: A list of labels for the combined annotations
    :rtype: list of str
    """
    labels = []
    tokenize = tokenize or list

    for shape, label in annotations:
        for i, _ in enumerate(tokenize(shape)):
            if i == 0:
                labels.append(label)
            else:
                labels.append(inside_label)

    return labels


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
