# coding: spec

from unittest import TestCase
import re
from spiel.segmentation import ConstraintSegmenter, Featurizer
from spiel.segmentation.constraints import (
    Constraint,
    SegmentationException,
    generate_constraints,
    generate_options
)
from spiel.segmentation.classification import ClassifierAdaptor


class DummyClassifier(ClassifierAdaptor):
    def __init__(self, instances):
        self.instances = instances

    @staticmethod
    def train(data):
        return DummyClassifier(data)

    def prob_classify(self, features):
        table = {
            '____fo_': [
                ('_-_-FOO', .9),
                ('_-FOO-BAR', .02),
                ('FOO-BAR-_', .03),
                ('BAR-_-_', 0.5)
            ],
            '___fo__': [
                ('_-_-FOO', .03),
                ('_-FOO-BAR', .92),
                ('FOO-BAR-_', .04),
                ('BAR-_-_', 0.01)
            ],
            '__fo___': [
                ('_-_-FOO', .03),
                ('_-FOO-BAR', .015),
                ('FOO-BAR-_', .95),
                ('BAR-_-_', 0.005)
            ],
            '_fo____': [
                ('_-_-FOO', .001),
                ('_-FOO-BAR', .009),
                ('FOO-BAR-_', .08),
                ('BAR-_-_', 0.91)
            ]
        }
        table['____f&o_'] = table['____fo_']
        table['___f&o__'] = table['___fo__']
        table['__f&o___'] = table['__fo___']
        table['_f&o____'] = table['_fo____']

        lookup = features['prefix']+features['focus']+features['suffix']
        if lookup in table:
            return table[lookup]
        return  [
            ('_-_-FOO', .23),
            ('_-FOO-BAR', .24),
            ('FOO-BAR-_', .26),
            ('BAR-_-_', 0.27)
        ]


describe TestCase 'ConstraintSegmenter':
    before_each:
        train_shapes = ['fo']
        train_annotations = [[('f', 'FOO'), ('o', 'BAR')]]
        self.segmenter = ConstraintSegmenter(DummyClassifier)
        self.segmenter.train(train_shapes, train_annotations)

    it 'accepts a custom featurizer':
        featurizer = Featurizer()
        segmenter = ConstraintSegmenter(DummyClassifier, featurizer=featurizer)
        self.assertIs(segmenter.featurizer, featurizer)

    describe 'train':
        it 'converts items into training instances and passes them into an internal classifier':
            instances = [
                ({'prefix': '___', 'focus': '_', 'suffix': 'fo_'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': 'o__'}, '_-FOO-BAR'),
                ({'prefix': '__f', 'focus': 'o', 'suffix': '___'}, 'FOO-BAR-_'),
                ({'prefix': '_fo', 'focus': '_', 'suffix': '___'}, 'BAR-_-_')
            ]
            self.assertEqual(self.segmenter.classifier.instances, instances)

        it 'raises an error if the number of shapes do not match the number of annotations':
            train_shapes = ['fo', 'bar']
            train_annotations = [[('f', 'FOO'), ('o', 'BAR')]]
            segmenter = ConstraintSegmenter(DummyClassifier)
            with self.assertRaises(SegmentationException):
                segmenter.train(train_shapes, train_annotations)

        it 'uses a custom featurizer':
            shapes = ['f&o']
            annotations = [[('f&', 'FOO'), ('o', 'BAR')]]
            featurizer = Featurizer(tokenize=lambda x: re.findall('.&?', x))
            segmenter = ConstraintSegmenter(DummyClassifier, featurizer=featurizer)
            segmenter.train(shapes, annotations)
            instances = [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f&o_'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f&', 'suffix': 'o__'}, '_-FOO-BAR'),
                ({'prefix': '__f&', 'focus': 'o', 'suffix': '___'}, 'FOO-BAR-_'),
                ({'prefix': '_f&o', 'focus': '_', 'suffix': '___'}, 'BAR-_-_')
            ]
            self.assertEqual(segmenter.classifier.instances, instances)


    describe 'segment':
        it 'raises an error if the segmenter has not already been trained':
            segmenter = ConstraintSegmenter(DummyClassifier)
            with self.assertRaises(SegmentationException):
                segmenter.segment('foo')

        it 'returns an empty list if an empty string is provided':
            self.assertEqual(self.segmenter.segment(''), [])

        it 'finds the most likely sequence of labels':
            labels = self.segmenter.segment('fo')
            self.assertEqual(labels, [('f', 'FOO'), ('o', 'BAR')])

        it 'uses a custom featurizer':
            shapes = ['f&o']
            annotations = [[('f&', 'FOO'), ('o', 'BAR')]]
            featurizer = Featurizer(tokenize=lambda x: re.findall('.&?', x))
            segmenter = ConstraintSegmenter(DummyClassifier, featurizer=featurizer)
            segmenter.train(shapes, annotations)
            labels = segmenter.segment('f&o')
            self.assertEqual(labels, [('f&', 'FOO'), ('o', 'BAR')])


describe TestCase 'generate_constraints':
    it 'generates constraints for a feature set':
        distribution = [
            ('_-_-FOO', .9),
            ('_-FOO-BAR', .02),
            ('FOO-BAR-_', .03),
            ('BAR-_-_', 0.5)
        ]
        constraints = generate_constraints(distribution, 3)

        target = {
            Constraint((2, 5), '_-_-FOO'): 0.9,
            Constraint((2, 3), '_'): 0.92,
            Constraint((3, 4), '_'): 1.4,
            Constraint((4, 5), 'FOO'): 0.9,
            Constraint((2, 4), '_-_'): 0.9,
            Constraint((3, 5), '_-FOO'): 0.9
        }

        self.assertEqual(constraints, target)


describe TestCase 'generate_options':
    it 'returns a list with six more elements than the string passed in':
        constraints = {
            Constraint((0, 3), '_-_-FOO'): 0.9,
            Constraint((0, 1), '_'): 0.92,
            Constraint((1, 2), '_'): 1.4,
            Constraint((2, 3), 'FOO'): 0.9,
            Constraint((0, 2), '_-_'): 0.9,
            Constraint((1, 3), '_-FOO'): 0.9
        }
        options = generate_options('foo', constraints)
        self.assertEqual(len(options), 9)

    it 'inserts an underscore if no constraint matches an index':
        constraints = {
            Constraint((1, 2), '_'): 1.4,
            Constraint((2, 3), 'FOO'): 0.9,
            Constraint((1, 3), '_-FOO'): 0.9
        }
        options = generate_options('foo', constraints)
        self.assertEqual(options[0], set('_'))

    it 'returns a list of sets of options based on the constraints':
        constraints = {
            Constraint((0, 3), '_-_-FOO'): 0.9,
            Constraint((0, 1), '_'): 0.92,
            Constraint((1, 2), '_'): 1.4,
            Constraint((2, 3), 'FOO'): 0.9,
            Constraint((0, 2), '_-_'): 0.9,
            Constraint((1, 3), '_-FOO'): 0.9,
            Constraint((1, 3), '_-BAR'): 0.9
        }
        options = generate_options('foo', constraints)
        self.assertEqual(options, [
            set('_'),
            set('_'),
            set(['FOO', 'BAR']),
            set('_'),
            set('_'),
            set('_'),
            set('_'),
            set('_'),
            set('_')
        ])
