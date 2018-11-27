# coding: spec

from unittest import TestCase
from spiel.sequence_labelling import SequenceLabeller
from spiel.sequence_labelling.labelling import LabellingException


class DummyClassifier:
    """
    Mock classifier to ensure labelling functions are working
    """
    def __init__(self, data, mode):
        self.data = data 
        self.mode = mode

    @staticmethod
    def build(features, labels):
        data = {'features': features, 'labels': labels}
        return DummyClassifier(data, 'build')

    @staticmethod
    def grid_search(features, labels):
        data = {'features': features, 'labels': labels}
        return DummyClassifier(data, 'grid')

    def predict(self, features):
        return ['FOO', 'BAR']


describe TestCase 'SequenceLabeller':
    describe 'train':
        it 'passes the supplied data on to the classifier':
            labeller = SequenceLabeller(DummyClassifier)
            labeller.train([['foo']], [['FOO']])

            self.assertEqual(labeller.model.data, {
                'features': [[{
                    'bias': 1.0,
                    'shape': 'foo',
                    'prefix1': 'f',
                    'prefix2': 'fo',
                    'prefix3': 'foo',
                    'suffix1': 'o',
                    'suffix2': 'oo',
                    'suffix3': 'foo',
                    'BOS': True,
                    'EOS': True
                }]],
                'labels': [['FOO']]
            })

        it 'uses grid search by default':
            labeller = SequenceLabeller(DummyClassifier)
            labeller.train([['foo']], [['FOO']])
            self.assertEqual(labeller.model.mode, 'grid')

        it 'does not use grid search if told not to':
            labeller = SequenceLabeller(DummyClassifier)
            labeller.train([['foo']], [['FOO']], grid_search=False)
            self.assertEqual(labeller.model.mode, 'build')

    describe 'label':
        it 'raises an error if the model has not been trained':
            labeller = SequenceLabeller()
            with self.assertRaises(LabellingException):
                labeller.label('foo')

        it 'predicts sequences':
            labeller = SequenceLabeller(DummyClassifier)
            labeller.train([['foo']], [['FOO']])
            labels = labeller.label(['foo', 'bar'])
            self.assertEqual(labels, ['FOO', 'BAR'])
