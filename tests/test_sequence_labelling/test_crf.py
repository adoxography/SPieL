# coding: spec

from unittest import TestCase
from pathlib import Path

from sklearn_crfsuite import CRF

from spiel.sequence_labelling import SequenceClassifier

describe TestCase 'SequenceClassifier':
    before_each:
        self.features = [
            [{'x': 'a'}, {'x': 'b'}],
            [{'x': 'b'}, {'x': 'c'}],
            [{'x': 'c'}, {'x': 'a'}]
        ]
        # Setting these to the same category so that warnings don't have to be
        # dealt with
        self.labels = [
            ['FOO', 'BAR'],
            ['BAR', 'BAZ'],
            ['BAZ', 'FOO']
        ]

    describe 'build':
        it 'generates a trained crf classifier':
            labeller = SequenceClassifier.build(self.features, self.labels)
            self.assertIsInstance(labeller.model, CRF)

        it 'passes additional options through to the crf':
            options = {
                'c1': 0.05,
                'max_iterations': 4
            }
            labeller = SequenceClassifier.build(self.features, self.labels, **options)
            crf = labeller.model
            self.assertEqual(crf.c1, 0.05)
            self.assertEqual(crf.max_iterations, 4)

    describe 'grid_search':
        it 'generates an optimal crf classifier':
            # Setting these to the same category so that warnings don't have to be
            # dealt with
            labels = [
                ['FOO', 'FOO'],
                ['FOO', 'FOO'],
                ['FOO', 'FOO']
            ]
            labeller = SequenceClassifier.grid_search(self.features, labels)
            self.assertIsInstance(labeller.model, CRF)

    describe 'save':
        before_each:
            self.path = Path('TEST_SEQUENCE_LABEL_MODEL.pickle')

        after_each:
            delete_file(self.path)

        it 'saves itself to disk':
            labeller = SequenceClassifier.build(self.features, self.labels)
            labeller.save(self.path)
            self.assertTrue(self.path.exists())

    describe 'load':
        before_each:
            self.path = Path('TEST_SEQUENCE_LABEL_MODEL.pickle')
            labeller = SequenceClassifier.build(self.features, self.labels, c1=0.001)
            labeller.save(self.path)

        after_each:
            delete_file(self.path)

        it 'loads a file from disk':
            loaded = SequenceClassifier.load(self.path)
            self.assertEqual(loaded.model.c1, 0.001)

    describe 'predict':
        it 'predicts labels for features':
            labeller = SequenceClassifier.build(self.features, self.labels)
            prediction = labeller.predict([{'x': 'a'}, {'x': 'b'}])
            self.assertEqual(prediction, ['FOO', 'BAR'])

    describe 'predict_many':
        it 'predicts multiple labels for multiple features':
            features = [
                [{'x': 'a'}, {'x': 'b'}],
                [{'x': 'b'}, {'x': 'c'}]
            ]
            labeller = SequenceClassifier.build(self.features, self.labels)
            prediction = labeller.predict_many(features)
            self.assertEqual(prediction, [['FOO', 'BAR'], ['BAR', 'BAZ']])

    describe 'evaluate':
        it 'evaluates its own performance':
            features = [
                [{'x': 'a'}, {'x': 'b'}],
                [{'x': 'b'}, {'x': 'c'}]
            ]
            labels = [['FOO', 'BAR'], ['BAZ', 'BAZ']]
            labeller = SequenceClassifier.build(self.features, self.labels)
            f1 = labeller.evaluate(features, labels)
            self.assertEqual(f1, 0.75)


def delete_file(path):
    if path.exists():
        path.unlink()
