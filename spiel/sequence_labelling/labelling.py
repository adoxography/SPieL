"""
spiel.sequence_labelling.labelling
"""
from spiel.sequence_labelling.features import Featurizer
from spiel.sequence_labelling.crf import SequenceClassifier


class LabellingException(RuntimeError):
    """Thrown if an exception occurs in labelling"""


class SequenceLabeller:
    """
    Labels sequences
    """
    def __init__(self, Classifier=None):
        self.featurizer = Featurizer(ngrams=3)
        self.classifier_type = Classifier or SequenceClassifier
        self.model = None

    def train(self, sequences, labels, grid_search=True):
        """
        Trains the underlying classifier

        :param sequences: The sequences to train on
        :type sequences: list of list
        :param labels: The labels for each sequence
        :type labels: list of list
        :param grid_search: Whether or not to use grid search to optimize the
                            model
        :type grid_search: bool
        """
        features = self.featurizer.convert_many(sequences)
        if grid_search:
            self.model = self.classifier_type.grid_search(features, labels)
        else:
            self.model = self.classifier_type.build(features, labels)

    def label(self, sequence):
        """
        Labels a sequence

        :param sequence: The sequence to label
        :type sequence: list
        :return: The labels for each segment of the sequence
        :rtype: list of str
        """
        if not self.model:
            raise LabellingException("The model has not been trained.")
        features = self.featurizer.convert(sequence)
        return self.model.predict(features)
