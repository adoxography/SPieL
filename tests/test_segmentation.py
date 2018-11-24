# coding: spec

from mspl.segmentation import Featurizer

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
