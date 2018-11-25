# coding: spec

import re
from mspl.segmentation import (
    Featurizer,
    concat_annotations,
    label_annotations,
    pad
)

describe 'Featurizer':
    describe 'convert':
        it 'returns three training instances when given a single character':
            featurizer = Featurizer()
            shape = 'f'
            labels = ['FOO']
            instances = featurizer.convert(shape, labels)
            assert instances == [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f__'}, '_+_+FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': '___'}, '_+FOO+_'),
                ({'prefix': '__f', 'focus': '_', 'suffix': '___'}, 'FOO+_+_')
            ]

        it 'returns four training instances when given two characters':
            featurizer = Featurizer()
            shape = 'fo'
            labels = ['FOO', 'BAR']
            instances = featurizer.convert(shape, labels)
            assert instances == [
                ({'prefix': '___', 'focus': '_', 'suffix': 'fo_'}, '_+_+FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': 'o__'}, '_+FOO+BAR'),
                ({'prefix': '__f', 'focus': 'o', 'suffix': '___'}, 'FOO+BAR+_'),
                ({'prefix': '_fo', 'focus': '_', 'suffix': '___'}, 'BAR+_+_')
            ]

        it 'handles shapes when they are lists':
            featurizer = Featurizer()
            shape = ['f', 'oo']
            labels = ['FOO', 'BAR']
            instances = featurizer.convert(shape, labels)
            assert instances == [
                ({'prefix': '___', 'focus': '_', 'suffix': 'foo_'}, '_+_+FOO'),
                ({'prefix': '___', 'focus': 'f', 'suffix': 'oo__'}, '_+FOO+BAR'),
                ({'prefix': '__f', 'focus': 'oo', 'suffix': '___'}, 'FOO+BAR+_'),
                ({'prefix': '_foo', 'focus': '_', 'suffix': '___'}, 'BAR+_+_')
            ]

        it 'uses a tokenize function to separate shape tokens':
            tokenize = lambda x: list(re.findall(r'.&?', x))
            featurizer = Featurizer(tokenize=tokenize)
            shape = 'f&'
            labels = ['FOO']
            instances = featurizer.convert(shape, labels)
            assert instances == [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f&__'}, '_+_+FOO'),
                ({'prefix': '___', 'focus': 'f&', 'suffix': '___'}, '_+FOO+_'),
                ({'prefix': '__f&', 'focus': '_', 'suffix': '___'}, 'FOO+_+_')
            ]

        it 'does not use the tokenize function on lists':
            tokenize = lambda x: list(re.findall(r'.&?', x))
            featurizer = Featurizer(tokenize=tokenize)
            shape = ['f&']
            labels = ['FOO']
            instances = featurizer.convert(shape, labels)
            assert instances == [
                ({'prefix': '___', 'focus': '_', 'suffix': 'f&__'}, '_+_+FOO'),
                ({'prefix': '___', 'focus': 'f&', 'suffix': '___'}, '_+FOO+_'),
                ({'prefix': '__f&', 'focus': '_', 'suffix': '___'}, 'FOO+_+_')
            ]

    describe 'label':
        it 'returns an empty list if no tokens are provided':
            featurizer = Featurizer()
            labels = featurizer.label('', [])
            assert labels == []

        it 'returns a list of one element when there is only one character':
            featurizer = Featurizer()
            annotations = [('f', 'bar')]
            labels = featurizer.label('f', annotations)
            assert labels == ['bar']

        it 'pads with inside labels if there are more characters than morphemes':
            featurizer = Featurizer()
            annotations = [('foo', 'bar')]
            labels = featurizer.label('foo', annotations)
            assert labels == ['bar', 'I', 'I']

        it 'uses the character provided as the inside label to the featurizer':
            featurizer = Featurizer(inside_label='FOO')
            annotations = [('foo', 'bar')]
            labels = featurizer.label('foo', annotations)
            assert labels == ['bar', 'FOO', 'FOO']

        it 'uses the function provided for tokenization':
            tokenize = lambda x: list(re.findall(r'._?', x))
            featurizer = Featurizer(tokenize=tokenize)
            annotations = [('foo', 'bar')]
            labels = featurizer.label('fo_', annotations)
            assert labels == ['bar', 'I+R(o,o_)+D(o)']

        it 'does not try to use the tokenization function when the input is a str':
            tokenize = lambda x: list(re.findall(r'._?', x))
            featurizer = Featurizer(tokenize=tokenize)
            annotations = [('foo', 'bar')]
            labels = featurizer.label(['f', 'o_'], annotations)
            assert labels == ['bar', 'I+R(o,o_)+D(o)']

        it 'handles multiple labels':
            featurizer = Featurizer()
            annotations = [('foo', 'bar'), ('baz', 'boo')]
            labels = featurizer.label('foobaz', annotations)
            assert labels == ['bar', 'I', 'I', 'boo', 'I', 'I']

        it 'handles single morphemes when the shape has a deleted character':
            featurizer = Featurizer()
            annotations = [('baz', 'bar')]
            labels = featurizer.label('ba', annotations)
            assert labels == ['bar', 'I+D(z)']

        it 'handles single morphemes when the shape has an inserted character':
            featurizer = Featurizer()
            annotations = [('baz', 'bar')]
            labels = featurizer.label('baza', annotations)
            assert labels == ['bar', 'I', 'I+I(a)', 'I']

        it 'handles single morphemes when the shape has multiple inserted characters':
            featurizer = Featurizer()
            annotations = [('baz', 'bar')]
            labels = featurizer.label('bazaa', annotations)
            assert labels == ['bar', 'I', 'I+I(a)+I(a)', 'I', 'I']

describe 'concat_annotations':
    it 'returns an empty string if no annotations are provided':
        concat = concat_annotations([])
        assert concat == ''

    it 'returns a concatenated string of the first elements in a series of tuples':
        concat = concat_annotations([('foo', ''), ('bar', '')])
        assert concat == 'foobar'

describe 'label_annotations':
    it 'returns an empty list if no annotations are provided':
        labels = label_annotations([], 'I')
        assert labels == []

    it 'returns the label of each annotation as the first label of each annotation':
        annotations = [('f', 'A'), ('b', 'B')]
        labels = label_annotations(annotations, 'I')
        assert labels == ['A', 'B']

    it 'uses the inside label for subsequent characters in an annotation':
        annotations = [('foo', 'A'), ('bar', 'B')]
        labels = label_annotations(annotations, 'I')
        assert labels == ['A', 'I', 'I', 'B', 'I', 'I']

    it 'uses a provided function for tokenization':
        annotations = [('fo.', 'A'), ('b.r', 'B')]
        tokenize = lambda x: list(re.findall(r'.\.?', x))
        labels = label_annotations(annotations, 'I', tokenize)
        assert labels == ['A', 'I', 'B', 'I']

describe 'pad':
    it 'adds a character to either side of a string':
        padded = pad('foo', '_', 1)
        assert padded == '_foo_'

    it 'adds multiple characters to either side of a string':
        padded = pad('foo', '_', 3)
        assert padded == '___foo___'

    it 'adds elements to either side of a list':
        padded = pad(['f', 'oo'], '_', 3)
        assert padded == ['_', '_', '_', 'f', 'oo', '_', '_', '_']
