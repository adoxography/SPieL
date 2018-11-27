# coding: spec

from unittest import TestCase
from spiel.sequence_labelling import Featurizer


describe TestCase 'Featurizer':
    before_each:
        self.featurizer = Featurizer()

    describe 'convert':
        it 'converts sequences to features':
            sequence = ['foot', 'ball']
            features = self.featurizer.convert(sequence)
            target = [
                {
                    'bias': 1.0,
                    'shape': 'foot',
                    'suffix3': 'oot',
                    'suffix2': 'ot',
                    'suffix1': 't',
                    'prefix3': 'foo',
                    'prefix2': 'fo',
                    'prefix1': 'f',
                    'next_shape': 'ball',
                    'BOS': True
                },
                {
                    'bias': 1.0,
                    'shape': 'ball',
                    'suffix3': 'all',
                    'suffix2': 'll',
                    'suffix1': 'l',
                    'prefix3': 'bal',
                    'prefix2': 'ba',
                    'prefix1': 'b',
                    'prev_shape': 'foot',
                    'EOS': True
                }
            ]

            self.assertEqual(features, target)

    describe 'convert_many':
        it 'converts multiple sequences to features':
            sequences = [['foot'], ['ball']]
            features = self.featurizer.convert_many(sequences)
            target = [
                [{
                    'bias': 1.0,
                    'shape': 'foot',
                    'suffix3': 'oot',
                    'suffix2': 'ot',
                    'suffix1': 't',
                    'prefix3': 'foo',
                    'prefix2': 'fo',
                    'prefix1': 'f',
                    'BOS': True,
                    'EOS': True
                }],
                [{
                    'bias': 1.0,
                    'shape': 'ball',
                    'suffix3': 'all',
                    'suffix2': 'll',
                    'suffix1': 'l',
                    'prefix3': 'bal',
                    'prefix2': 'ba',
                    'prefix1': 'b',
                    'BOS': True,
                    'EOS': True
                }]
            ]
            self.assertEqual(features, target)
