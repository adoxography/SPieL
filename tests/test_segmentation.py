# coding: spec

from unittest import TestCase
import re
from mspl.segmentation import (
    Featurizer,
    Segmenter,
    Constraint,
    SegmentationException,
    FeaturizationException,
    concat_annotations,
    label_annotations,
    pad
)

class DummyClassifier:
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
        lookup = features['prefix']+features['focus']+features['suffix']
        if lookup in table:
            return table[lookup]
        return  [
            ('_-_-FOO', .23),
            ('_-FOO-BAR', .24),
            ('FOO-BAR-_', .26),
            ('BAR-_-_', 0.27)
        ]


describe TestCase 'Segmenter':
    before_each:
        instances = [
            ({'prefix': '___', 'focus': '_', 'suffix': 'fo_'}, '_-_-FOO'),
            ({'prefix': '___', 'focus': 'f', 'suffix': 'o__'}, '_-FOO-BAR'),
            ({'prefix': '__f', 'focus': 'o', 'suffix': '___'}, 'FOO-BAR-_'),
            ({'prefix': '_fo', 'focus': '_', 'suffix': '___'}, 'BAR-_-_')
        ]
        self.segmenter = Segmenter(DummyClassifier)
        self.segmenter.train(instances)

    describe 'train':
        it 'passes instances into an internal classifier':
            self.assertEqual(len(self.segmenter.classifier.instances),  4)

    describe 'segment':
        it 'raises an error if the segmenter has not already been trained':
            segmenter = Segmenter(DummyClassifier)
            with self.assertRaises(SegmentationException):
                segmenter.segment('foo')

        it 'returns an empty list if an empty string is provided':
            self.assertEqual(self.segmenter.segment(''), [])

    describe 'generate_constraints':
        it 'generates constraints for a feature set':
            distribution = [
                ('_-_-FOO', .9),
                ('_-FOO-BAR', .02),
                ('FOO-BAR-_', .03),
                ('BAR-_-_', 0.5)
            ]
            constraints = self.segmenter.generate_constraints(distribution, 3)

            target = {
                Constraint((2, 5), '_-_-FOO'): 0.9,
                Constraint((2, 3), '_'): 0.92,
                Constraint((3, 4), '_'): 1.4,
                Constraint((4, 5), 'FOO'): 0.9,
                Constraint((2, 4), '_-_'): 0.9,
                Constraint((3, 5), '_-FOO'): 0.9
            }

            self.assertEqual(constraints, target)

    describe 'generate_options':
        it 'returns a list with two more elements than the string passed in':
            constraints = {
                Constraint((0, 3), '_-_-FOO'): 0.9,
                Constraint((0, 1), '_'): 0.92,
                Constraint((1, 2), '_'): 1.4,
                Constraint((2, 3), 'FOO'): 0.9,
                Constraint((0, 2), '_-_'): 0.9,
                Constraint((1, 3), '_-FOO'): 0.9
            }
            options = self.segmenter.generate_options('foo', constraints)
            self.assertEqual(len(options), 5)

        it 'inserts an underscore if no constraint matches an index':
            constraints = {
                Constraint((1, 2), '_'): 1.4,
                Constraint((2, 3), 'FOO'): 0.9,
                Constraint((1, 3), '_-FOO'): 0.9
            }
            options = self.segmenter.generate_options('foo', constraints)
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
            options = self.segmenter.generate_options('foo', constraints)
            self.assertEqual(options, [
                set('_'),
                set('_'),
                set(['FOO', 'BAR']),
                set('_'),
                set('_')
            ])


describe TestCase 'Featurizer':
    describe 'convert_pairs':
        it 'returns three training instances when given a single character':
            featurizer = Featurizer()
            shape = 'f'
            labels = ['FOO']
            instances = featurizer.convert_pairs(shape, labels)
            self.assertEqual(instances, [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f__'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': '___'}, '_-FOO-_'),
                ({'prefix': '__f', 'focus': '_', 'suffix': '___'}, 'FOO-_-_')
            ])

        it 'returns four training instances when given two characters':
            featurizer = Featurizer()
            shape = 'fo'
            labels = ['FOO', 'BAR']
            instances = featurizer.convert_pairs(shape, labels)
            self.assertEqual(instances, [
                ({'prefix': '___', 'focus': '_', 'suffix': 'fo_'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': 'o__'}, '_-FOO-BAR'),
                ({'prefix': '__f', 'focus': 'o', 'suffix': '___'}, 'FOO-BAR-_'),
                ({'prefix': '_fo', 'focus': '_', 'suffix': '___'}, 'BAR-_-_')
            ])

        it 'handles shapes when they are lists':
            featurizer = Featurizer()
            shape = ['f', 'oo']
            labels = ['FOO', 'BAR']
            instances = featurizer.convert_pairs(shape, labels)
            self.assertEqual(instances, [
                ({'prefix': '___', 'focus': '_', 'suffix': 'foo_'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': 'oo__'}, '_-FOO-BAR'),
                ({'prefix': '__f', 'focus': 'oo', 'suffix': '___'}, 'FOO-BAR-_'),
                ({'prefix': '_foo', 'focus': '_', 'suffix': '___'}, 'BAR-_-_')
            ])

        it 'uses a tokenize function to separate shape tokens':
            tokenize = lambda x: list(re.findall(r'.&?', x))
            featurizer = Featurizer(tokenize=tokenize)
            shape = 'f&'
            labels = ['FOO']
            instances = featurizer.convert_pairs(shape, labels)
            self.assertEqual(instances, [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f&__'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f&', 'suffix': '___'}, '_-FOO-_'),
                ({'prefix': '__f&', 'focus': '_', 'suffix': '___'}, 'FOO-_-_')
            ])

        it 'does not use the tokenize function on lists':
            tokenize = lambda x: list(re.findall(r'.&?', x))
            featurizer = Featurizer(tokenize=tokenize)
            shape = ['f&']
            labels = ['FOO']
            instances = featurizer.convert_pairs(shape, labels)
            self.assertEqual(instances, [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f&__'}, '_-_-FOO'),
                ({'prefix': '___', 'focus': 'f&', 'suffix': '___'}, '_-FOO-_'),
                ({'prefix': '__f&', 'focus': '_', 'suffix': '___'}, 'FOO-_-_')
            ])

        it 'raises an error if the size of the shape and labels do not match':
            featurizer = Featurizer()
            shape = 'f&'
            labels = ['FOO']
            with self.assertRaises(FeaturizationException):
                featurizer.convert_pairs(shape, labels)

    describe 'convert_features':
        it 'returns three training instances when given a single character':
            featurizer = Featurizer()
            shape = 'f'
            labels = ['FOO']
            instances = featurizer.convert_features(shape)
            self.assertEqual(instances, [
                {'prefix': '___', 'focus': '_', 'suffix': 'f__'},
                {'prefix': '___', 'focus': 'f', 'suffix': '___'},
                {'prefix': '__f', 'focus': '_', 'suffix': '___'}
            ])

        it 'returns four training instances when given two characters':
            featurizer = Featurizer()
            shape = 'fo'
            labels = ['FOO', 'BAR']
            instances = featurizer.convert_features(shape)
            self.assertEqual(instances, [
                {'prefix': '___', 'focus': '_', 'suffix': 'fo_'},
                {'prefix': '___', 'focus': 'f', 'suffix': 'o__'},
                {'prefix': '__f', 'focus': 'o', 'suffix': '___'},
                {'prefix': '_fo', 'focus': '_', 'suffix': '___'}
            ])

    describe 'label':
        it 'returns an empty list if no tokens are provided':
            featurizer = Featurizer()
            labels = featurizer.label('', [])
            self.assertEqual(labels, [])

        it 'returns a list of one element when there is only one character':
            featurizer = Featurizer()
            annotations = [('f', 'bar')]
            labels = featurizer.label('f', annotations)
            self.assertEqual(labels, ['bar'])

        it 'pads with inside labels if there are more characters than morphemes':
            featurizer = Featurizer()
            annotations = [('foo', 'bar')]
            labels = featurizer.label('foo', annotations)
            self.assertEqual(labels, ['bar', 'I', 'I'])

        it 'uses the character provided as the inside label to the featurizer':
            featurizer = Featurizer(inside_label='FOO')
            annotations = [('foo', 'bar')]
            labels = featurizer.label('foo', annotations)
            self.assertEqual(labels, ['bar', 'FOO', 'FOO'])

        it 'uses the function provided for tokenization':
            tokenize = lambda x: list(re.findall(r'._?', x))
            featurizer = Featurizer(mode='full', tokenize=tokenize)
            annotations = [('foo', 'bar')]
            labels = featurizer.label('fo_', annotations)
            self.assertEqual(labels, ['bar', 'I+R(o,o_)+D(o)'])

        it 'does not try to use the tokenization function when the input is a str':
            tokenize = lambda x: list(re.findall(r'._?', x))
            featurizer = Featurizer(mode='full', tokenize=tokenize)
            annotations = [('foo', 'bar')]
            labels = featurizer.label(['f', 'o_'], annotations)
            self.assertEqual(labels, ['bar', 'I+R(o,o_)+D(o)'])

        it 'handles multiple labels':
            featurizer = Featurizer()
            annotations = [('foo', 'bar'), ('baz', 'boo')]
            labels = featurizer.label('foobaz', annotations)
            self.assertEqual(labels, ['bar', 'I', 'I', 'boo', 'I', 'I'])

        it 'handles single morphemes when the shape has a deleted character':
            featurizer = Featurizer(mode='full')
            annotations = [('baz', 'bar')]
            labels = featurizer.label('ba', annotations)
            self.assertEqual(labels, ['bar', 'I+D(z)'])

        it 'handles single morphemes when the shape has an inserted character':
            featurizer = Featurizer(mode='full')
            annotations = [('baz', 'bar')]
            labels = featurizer.label('baza', annotations)
            self.assertEqual(labels, ['bar', 'I', 'I+I(a)', 'I'])

        it 'handles single morphemes when the shape has multiple inserted characters':
            featurizer = Featurizer(mode='full')
            annotations = [('baz', 'bar')]
            labels = featurizer.label('bazaa', annotations)
            self.assertEqual(labels, ['bar', 'I', 'I+I(a)+I(a)', 'I', 'I'])

        it 'can remove operation directions':
            featurizer = Featurizer(mode='normal')
            annotations = [('foo', 'bar'), ('baz', 'boo')]
            labels = featurizer.label('fooba', annotations)
            self.assertEqual(labels, ['bar', 'I', 'I', 'boo', 'I'])

        it 'can reduce labels down to B/I/O':
            featurizer = Featurizer(mode='basic')
            annotations = [('foo', 'bar'), ('baz', 'boo')]
            labels = featurizer.label('fooba', annotations)
            self.assertEqual(labels, ['B', 'I', 'I', 'B', 'I'])


describe TestCase 'concat_annotations':
    it 'returns an empty string if no annotations are provided':
        concat = concat_annotations([])
        self.assertEqual(concat, '')

    it 'returns a concatenated string of the first elements in a series of tuples':
        concat = concat_annotations([('foo', ''), ('bar', '')])
        self.assertEqual(concat, 'foobar')

describe TestCase 'label_annotations':
    it 'returns an empty list if no annotations are provided':
        labels = label_annotations([], 'I')
        self.assertEqual(labels, [])

    it 'returns the label of each annotation as the first label of each annotation':
        annotations = [('f', 'A'), ('b', 'B')]
        labels = label_annotations(annotations, 'I')
        self.assertEqual(labels, ['A', 'B'])

    it 'uses the inside label for subsequent characters in an annotation':
        annotations = [('foo', 'A'), ('bar', 'B')]
        labels = label_annotations(annotations, 'I')
        self.assertEqual(labels, ['A', 'I', 'I', 'B', 'I', 'I'])

    it 'uses a provided function for tokenization':
        annotations = [('fo.', 'A'), ('b.r', 'B')]
        tokenize = lambda x: list(re.findall(r'.\.?', x))
        labels = label_annotations(annotations, 'I', tokenize)
        self.assertEqual(labels, ['A', 'I', 'B', 'I'])


describe TestCase 'pad':
    it 'adds a character to either side of a string':
        padded = pad('foo', '_', 1)
        self.assertEqual(padded, '_foo_')

    it 'adds multiple characters to either side of a string':
        padded = pad('foo', '_', 3)
        self.assertEqual(padded, '___foo___')

    it 'adds elements to either side of a list':
        padded = pad(['f', 'oo'], '_', 3)
        self.assertEqual(padded, ['_', '_', '_', 'f', 'oo', '_', '_', '_'])
