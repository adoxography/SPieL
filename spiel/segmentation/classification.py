"""
spiel.segmentation.classification

Defines classes that can be used by segmenters
"""
from abc import ABCMeta, abstractmethod
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class ClassifierAdaptor(metaclass=ABCMeta):
    """
    Adaptor interface to ensure a classifier can be used by a segmenter
    """
    @abstractmethod
    def prob_classify(self, features):
        """
        Gives back all of the labels and their corresponding probabilities
        based on the provided features

        :param features: The features to classify
        :type features: dict
        :return: The available labels and their corresponding probabilities
        :rtype: dict of str => float
        """

    @staticmethod
    @abstractmethod
    def train(data):
        """
        Initializes a new instance of the classifier, using the data to train
        on

        :param data: The data to train the classifier with
        :type data: list of (dict, str)
        :return: An instance of this class
        """


class SKLearnNaiveBayesClassifier(ClassifierAdaptor):
    """
    Wrapper around sklearn's NaiveBayesClassifier to ensure it conforms to the
    interface spiel is expecting
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def prob_classify(self, features):
        probabilities = self.pipeline.predict_proba(features)[0]
        labels = self.pipeline.classes_
        return list(zip(labels, probabilities))

    @staticmethod
    def train(data):
        pipeline = Pipeline([
            ('vect', DictVectorizer()),
            ('clf', MultinomialNB())
        ])
        pipeline.fit(*zip(*data))
        return SKLearnNaiveBayesClassifier(pipeline)
