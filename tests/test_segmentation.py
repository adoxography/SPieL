# coding: spec

import re
from mspl.segmentation import Featurizer, concat_annotations, label_annotations

describe 'Featurizer':
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
