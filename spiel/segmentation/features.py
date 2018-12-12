"""
spiel.segmentation.features

Handles featurization of strings for segmenters
"""
import re
from spiel import levenshtein
from spiel.levenshtein import INSERT_SYMBOL
from spiel.util import pad


class FeaturizationException(Exception):
    """Raises for an error in segmentation"""


class Featurizer:
    """
    Used to convert basic instances into training instances for a classifier
    """
    def __init__(self, mode='normal', inside_label='I', pad_token='_',
                 tokenize=None):
        """
        Initializes the featurizer

        :param mode: The labelling mode to use; one of 'basic', 'normal', or
                     'full'
        :type mode: str
        :param inside_label: The label to use for a character that does not
                             begin a label; default 'I'
        :type inside_label: str
        :param pad_token: The token to pad strings with; default '_'
        :type pade_token: str
        :param tokenize: A function to tokenize incoming strings. Defaults to
                         list()
        :type tokenize: callable
        """
        self.inside_label = inside_label
        self.pad_token = pad_token
        self.tokenize = tokenize or list
        self.mode = mode

    def convert_pairs(self, shape, labels):
        """
        Converts a shape and corresponding labels into training instances

        :param shape: The shape to convert
        :type shape: list or str
        :param labels: The labels for each token in the shape
        :type labels: list
        :return: A list of training instances, where the first element is a
                 dict of features, and the second is a trigram label for the
                 instance
        :rtype: list of (dict, str)
        """
        if isinstance(shape, str):
            shape = self.tokenize(shape)

        if not len(shape) == len(labels):
            raise FeaturizationException(f"{len(shape)} tokens in *shape*,\
but {len(labels)} labels provided")

        features = self.convert_features(shape)

        padded_labels = pad(labels, self.pad_token, 4)
        instances = ['-'.join(padded_labels[i-1:i+2])
                     for i in range(3, len(padded_labels) - 3)]

        return list(zip(features, instances))

    def convert_features(self, shape):
        """
        Converts a shape into training instances

        :param shape: The shape to convert
        :type shape: list or str
        :return: A list of training instances, where the first element is a
                 dict of features, and the second is a trigram label for the
                 instance
        :rtype: list of dict
        """
        if not shape:
            return []

        if isinstance(shape, str):
            shape = self.tokenize(shape)

        padded_shape = pad(shape, self.pad_token, 4)
        instances = []

        for i in range(3, len(padded_shape) - 3):
            features = {
                'prefix': ''.join(padded_shape[i-3:i]),
                'focus': padded_shape[i],
                'suffix': ''.join(padded_shape[i+1:i+4])
            }

            instances.append(features)

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
            # Perform any insertion directives
            matches = re.findall(rf'\+{INSERT_SYMBOL}(.*?)', label)
            for _ in matches:
                labels.insert(i+1, self.inside_label)

        return [self.__format_label(label) for label in labels]

    def __format_label(self, label):
        if self.mode == 'basic' or self.mode == 'normal':
            # Get rid of any directives
            label = re.match(r'^[^+]*', label)[0]

        if self.mode == 'basic':
            label = self.__simplify_label(label)

        return label

    def __simplify_label(self, label):
        if label == self.pad_token:
            return 'O'
        if label == self.inside_label:
            return 'I'
        return 'B'


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
