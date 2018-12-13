# coding: spec
from pathlib import Path

from sklearn_crfsuite import CRF

from spiel.sequence_labelling import SequenceClassifier

describe 'SequenceClassifier':
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
            classifier = SequenceClassifier.build(self.features, self.labels)
            self.assertIsInstance(classifier.model, CRF)

        it 'passes additional options through to the crf':
            options = {
                'c1': 0.05,
                'max_iterations': 4
            }
            classifier = SequenceClassifier.build(self.features, self.labels, **options)
            crf = classifier.model
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
            classifier = SequenceClassifier.grid_search(self.features, labels)
            self.assertIsInstance(classifier.model, CRF)

    describe 'save':
        before_each:
            self.path = Path('TEST_SEQUENCE_LABEL_MODEL.pickle')

        after_each:
            delete_file(self.path)

        it 'saves itself to disk':
            classifier = SequenceClassifier.build(self.features, self.labels)
            classifier.save(self.path)
            self.assertTrue(self.path.exists())

    describe 'load':
        before_each:
            self.path = Path('TEST_SEQUENCE_LABEL_MODEL.pickle')
            classifier = SequenceClassifier.build(self.features, self.labels, c1=0.001)
            classifier.save(self.path)

        after_each:
            delete_file(self.path)

        it 'loads a file from disk':
            loaded = SequenceClassifier.load(self.path)
            self.assertEqual(loaded.model.c1, 0.001)

    describe 'predict':
        it 'predicts labels for features':
            classifier = SequenceClassifier.build(self.features, self.labels)
            prediction = classifier.predict([{'x': 'a'}, {'x': 'b'}])
            self.assertEqual(prediction, ['FOO', 'BAR'])

    describe 'predict_many':
        it 'predicts multiple labels for multiple features':
            features = [
                [{'x': 'a'}, {'x': 'b'}],
                [{'x': 'b'}, {'x': 'c'}]
            ]
            classifier = SequenceClassifier.build(self.features, self.labels)
            prediction = classifier.predict_many(features)
            self.assertEqual(prediction, [['FOO', 'BAR'], ['BAR', 'BAZ']])

    describe 'evaluate':
        it 'evaluates its own performance':
            features = [
                [{'x': 'a'}, {'x': 'b'}],
                [{'x': 'b'}, {'x': 'c'}]
            ]
            labels = [['FOO', 'BAR'], ['BAZ', 'BAZ']]
            classifier = SequenceClassifier.build(self.features, self.labels)
            f1 = classifier.evaluate(features, labels)
            self.assertEqual(f1, 0.75)


def delete_file(path):
    if path.exists():
        path.unlink()
