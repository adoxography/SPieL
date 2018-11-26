"""
segmentation.py

Contains logic for segmenting shapes into morphemes
"""
import re
from collections import defaultdict
from mspl import levenshtein
from mspl.levenshtein import INSERT_SYMBOL


class SegmentationException(Exception):
    """Raises for an error in segmentation"""


class FeaturizationException(Exception):
    """Raises for an error in segmentation"""


class Constraint:
    def __init__(self, span, label):
        self.span = span
        self.label = label

    def __repr__(self):
        return f"{self.span}->{self.label}"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self.span == other.span and self.label == other.label


class Segmenter:
    def __init__(self, Classifier):
        self.ClassifierType = Classifier
        self.featurizer = Featurizer()
        self.classifier = None

    def train(self, instances):
        self.classifier = self.ClassifierType.train(instances)

    def segment(self, string):
        if self.classifier is None:
            raise SegmentationException("The segmenter has not been trained")

        constraints = {}
        features = self.featurizer.convert_features(string)

        for i, feature in enumerate(features):
            distribution = self.classifier.prob_classify(feature)
            constraints.update(self.generate_constraints(distribution, i))

        return []

    def generate_constraints(self, distribution, index):
        """
        Generates constraints for *distribution*

        :param distribution: A probability distribution of label/probability
                             pairs
        :type distribution: dict of str => float
        :param index: The index that the distribution came from
        :type index: int
        :return: The constraints that were extracted from the distribution
        :rtype: dict of Constraint => float
        """
        constraints = defaultdict(float)

        tg_label, tg_weight = max(distribution, key=lambda x: x[1])
        tg_constraint = Constraint((index-1, index+2), tg_label)
        constraints[tg_constraint] = tg_weight

        pre_label, foc_label, suf_label = tg_label.split('-')
        pre_bg_label = pre_label + '-' + foc_label
        suf_bg_label = foc_label + '-' + suf_label

        pre_ug_constraint = Constraint((index-1, index), pre_label)
        foc_ug_constraint = Constraint((index, index+1), foc_label)
        suf_ug_constraint = Constraint((index+1, index+2), suf_label)
        pre_bg_constraint = Constraint((index-1, index+1), pre_bg_label)
        suf_bg_constraint = Constraint((index, index+2), suf_bg_label)

        for label, weight in distribution:
            if re.search('^%s-' % re.escape(pre_label), label):
                constraints[pre_ug_constraint] += weight
            if re.search('-%s-' % re.escape(foc_label), label):
                constraints[foc_ug_constraint] += weight
            if re.search('-%s$' % re.escape(suf_label), label):
                constraints[suf_ug_constraint] += weight
            if re.search('^%s-' % re.escape(pre_bg_label), label):
                constraints[pre_bg_constraint] += weight
            if re.search('-%s$' % re.escape(suf_bg_label), label):
                constraints[suf_bg_constraint] += weight

        return constraints


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
        instances = ['-'.join(padded_labels[i-1:i+2]) for i in range(3, len(padded_labels) - 3)]

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
            matches = re.findall(rf'\+{INSERT_SYMBOL}(.*?)', label)
            for _ in matches:
                labels.insert(i+1, self.inside_label)

        if self.mode in ['basic', 'normal']:
            # Get rid of any operator directives
            labels = [re.match(r'^[^+]*', label)[0] for label in labels]

        if self.mode == 'basic':
            # Convert all labels to B/I/O scheme
            labels = [self.__simplify_label(label) for label in labels]

        return labels

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
