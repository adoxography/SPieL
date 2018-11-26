"""
mspl.segmentation.constraints

Module for segmenting strings based on constraints
"""
import re
from collections import defaultdict
from mspl.segmentation.features import Featurizer
from mspl.util import all_permutations


class SegmentationException(Exception):
    """Raises for an error in segmentation"""


class Constraint:
    """
    Represents a constraint on a sequence
    """
    def __init__(self, span, label):
        """
        Initializes the constraint

        :param span: The start and one past the end of the span constrained
        :type span: (int, int)
        :param label: The label of this constraint
        :type label: str
        """
        self.span = span
        self.label = label

    def is_satisfied(self, sequence):
        """
        Determines if a sequence is satisfied by this constraint

        :param sequence: The sequence to check
        :type sequence: list of str
        :rtype: bool
        """
        segment = '-'.join(sequence[self.span[0]:self.span[1]])
        return self.label == segment

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
    def __init__(self, Classifier):
        self.classifier_type = Classifier
        self.featurizer = Featurizer()
        self.classifier = None

    def train(self, instances):
        """
        Trains the underlying classifier

        :param instances: The instances to use to train the classifier
        :type instances: list of (dict, str)
        """
        self.classifier = self.classifier_type.train(instances)

    def segment(self, string):
        """
        Segments a string into morphemes

        :param string: The string to segment
        :type string: str
        :return: A list of morphemes
        :rtype: list of str
        """
        if self.classifier is None:
            raise SegmentationException("The segmenter has not been trained")

        constraints = {}
        features = self.featurizer.convert_features(string)

        for i, feature in enumerate(features):
            distribution = self.classifier.prob_classify(feature)
            constraints.update(generate_constraints(distribution, i+2))

        options = generate_options(string, constraints)
        solutions = all_permutations(options)

        optimal_solution = find_optimal_solution(solutions, constraints)
        return optimal_solution[3:-3]


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
