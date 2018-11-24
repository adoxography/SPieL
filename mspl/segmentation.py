import re
from mspl import levenshtein


class Featurizer:
    def __init__(self):
        self.inside_label = 'I'

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
        annotation_string = concat_annotations(annotations)
        labels = label_annotations(annotations, self.inside_label)

        ops = levenshtein.operations(annotation_string, shape)
        labels = levenshtein.annotate(annotation_string, shape, ops, labels)

        for i, label in enumerate(labels):
            matches = re.findall(r'\+I(.*?)', label)
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


def label_annotations(annotations, inside_label):
    """
    Creates a list of labels for each character in a list of annotations

    :param annotations: A list of tuples, where the first is the shape of the
                        annotations, and the second is its label
    :type annotations: list of (str, str)
    :param inside_label: The label to use for characters that do not begin an
                         annotation
    :return: A list of labels for the combined annotations
    :rtype: list of str
    """
    labels = []

    for shape, label in annotations:
        for i, _ in enumerate(shape):
            if i == 0:
                labels.append(label)
            else:
                labels.append(inside_label)

    return labels
