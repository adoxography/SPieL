# coding: spec

from unittest import TestCase
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from spiel.segmentation.classification import SKLearnNaiveBayesClassifier

describe TestCase 'SKLearnNaiveBayesClassifier':
    describe 'train':
        it 'creates a classifier containing a dict vectorizer and a NB classifier':
            data = [({'foo': 'bar'}, 'FOO'), ({'foo': 'y'}, 'BAR')]
            classifier = SKLearnNaiveBayesClassifier.train(data)
            steps = classifier.pipeline.steps

            self.assertIsInstance(classifier.pipeline, Pipeline)
            self.assertIsInstance(steps[0][1], DictVectorizer)
            self.assertIsInstance(steps[1][1], MultinomialNB)

    describe 'prob_classify':
        it 'returns a list of label/probability pairs':
            data = [({'foo': 'bar'}, 'FOO'), ({'foo': 'y'}, 'BAR')]
            classifier = SKLearnNaiveBayesClassifier.train(data)
            results = classifier.prob_classify({'foo': 'bar'})
            
            for result in results:
                self.assertIsInstance(result[0], str)
                self.assertIsInstance(result[1], float)
