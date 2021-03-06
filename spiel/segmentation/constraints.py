"""
spiel.segmentation.constraints

Module for segmenting strings based on constraints. Implementation based on
van den Bosch and Canisius (2006). (http://aclweb.org/anthology/W06-3206)
"""
import re
from collections import defaultdict
from spiel.segmentation.features import Featurizer
from spiel.segmentation.classification import SKLearnNaiveBayesClassifier
from spiel.util import all_permutations


class SegmentationException(Exception):
    """Raises for an error in segmentation"""


class Constraint:
    """
    Represents a constraint on a sequence
    """
    def __init__(self, span, label, position=None):
        """
        Initializes the constraint

        :param span: The start and one past the end of the span constrained
        :type span: (int, int)
        :param label: The label of this constraint
        :type label: str
        :param position: Where in the trigram sequence this constraint falls
        """
        self.span = span
        self.label = label
        self.position = position

    def is_satisfied(self, sequence):
        """
        Determines if a sequence is satisfied by this constraint

        :param sequence: The sequence to check
        :type sequence: list of str
        :rtype: bool
        """
        segment = '-'.join(sequence[self.span[0]:self.span[1]])
        return self.label == segment

    @property
    def pattern(self):
        """
        Generates an escaped pattern for searching for this constraint, based
        on its position

        :rtype: str
        """
        if self.position == 'prefix':
            base = '^%s-'
        elif self.position == 'focus':
            base = '-%s-'
        elif self.position == 'suffix':
            base = '-%s$'
        else:
            base = '%s'

        return base % re.escape(self.label)

    def __repr__(self):
        return f"{self.span}->{self.label}"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self.span == other.span and self.label == other.label


class ConstraintSegmenter:
    """
    Segments strings based on constraint satisfaction
    """
    def __init__(self, Classifier=None, featurizer=None):
        self.classifier_type = Classifier or SKLearnNaiveBayesClassifier
        self.featurizer = featurizer or Featurizer()
        self.classifier = None

    def train(self, shapes, annotations):
        """
        Trains the underlying classifier

        :param shapes: The observable strings to train on
        :type shapes: list of str or list of list of str
        :param annotations: Annotations for each shape
        :type annotations: list of list of str
        """
        if not len(shapes) == len(annotations):
            raise SegmentationException(f"There are {len(shapes)} shapes but \
{len(annotations)} annotations.")
        instances = []
        for shape, annotation in zip(shapes, annotations):
            labels = self.featurizer.label(shape, annotation)
            instances += self.featurizer.convert_pairs(shape, labels)
        self.classifier = self.classifier_type.train(instances)

    def annotate(self, sequence):
        """
        Generates an annotated version of a sequence

        :param sequence: The sequence to segment
        :type sequence: list or str
        :return: A list of morpheme/label pairs
        :rtype: list of (str, str)
        """
        if self.classifier is None:
            raise SegmentationException("The segmenter has not been trained")

        if isinstance(sequence, str):
            sequence = self.featurizer.tokenize(sequence)

        constraints = {}
        features = self.featurizer.convert_features(sequence)

        for i, feature in enumerate(features):
            distribution = self.classifier.prob_classify(feature)
            constraints.update(generate_constraints(distribution, i+2))

        options = generate_options(sequence, constraints)
        solutions = all_permutations(options)

        optimal_solution = find_optimal_solution(solutions, constraints)
        return self.__merge_labels(sequence, optimal_solution[3:-3])

    def segment(self, sequence):
        """
        Segments a sequence into morphemes

        :param sequence: The sequence to segment
        :type sequence: list or str
        :return: A list of morpheme/label pairs
        :rtype: list of (str, str)
        """
        return [segment for segment, _ in self.annotate(sequence)]

    def label(self, sequence):
        """
        Generates the labels for a sequence

        :param sequence: The sequence to segment
        :type sequence: list or str
        :return: A list of morpheme/label pairs
        :rtype: list of (str, str)
        """
        return [label for _, label in self.annotate(sequence)]

    def __merge_labels(self, sequence, labels):
        segments = []
        curr_segment = ''
        curr_label = None

        for token, label in zip(sequence, labels):
            if not label == self.featurizer.inside_label:
                if curr_label is not None:
                    segments.append((curr_segment, curr_label))
                    curr_segment = ''
                curr_label = label
            curr_segment += token

        if curr_label:
            segments.append((curr_segment, curr_label))

        return segments


def generate_constraints(distribution, index):
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
    weights = defaultdict(float)

    tg_label, _ = max(distribution, key=lambda x: x[1])

    constraints = __initialize_constraints(tg_label, index)

    for label, weight in distribution:
        for constraint in constraints:
            if re.search(constraint.pattern, label):
                weights[constraint] += weight

    return weights


def __initialize_constraints(tg_label, index):
    pre_label, foc_label, suf_label = tg_label.split('-')
    pre_bg_label = pre_label + '-' + foc_label
    suf_bg_label = foc_label + '-' + suf_label

    pre_ug_constraint = Constraint((index-1, index), pre_label, 'prefix')
    foc_ug_constraint = Constraint((index, index+1), foc_label, 'focus')
    suf_ug_constraint = Constraint((index+1, index+2), suf_label, 'suffix')
    pre_bg_constraint = Constraint((index-1, index+1), pre_bg_label, 'prefix')
    suf_bg_constraint = Constraint((index, index+2), suf_bg_label, 'suffix')
    tg_constraint = Constraint((index-1, index+2), tg_label)

    return [pre_ug_constraint,
            foc_ug_constraint,
            suf_ug_constraint,
            pre_bg_constraint,
            suf_bg_constraint,
            tg_constraint]


def generate_options(sequence, constraints):
    """
    Generates options for each element of *sequence*, based on *constraints*

    :param seqence: The sequence to generate options for
    :type sequence: list or str
    :param constraints: The constraints to base the options on
    :type constraints: list of Constraint
    :return: A list of options
    :rtype: list of set of str
    """
    options = [set() for _ in range(len(sequence) + 6)]

    for constraint in constraints:
        labels = constraint.label.split('-')
        for i, label in enumerate(labels):
            index = constraint.span[0] + i
            options[index].add(label)

    for option in options:
        if not option:
            option.add('_')

    return options


def find_optimal_solution(solutions, constraints):
    """
    Finds the optimal solution given a list of solutions and a list of
    constraints
    """
    weighted_solutions = []

    for solution in solutions:
        value = 0
        for constraint, weight in constraints.items():
            if constraint.is_satisfied(solution):
                value += weight
        weighted_solutions.append((solution, value))

    best_solution, _ = max(weighted_solutions, key=lambda x: x[1])
    return best_solution
