# coding: spec
from unittest import TestCase
from mspl.util import all_permutations, pad


describe TestCase 'all_permutations':
    it 'combines all possible options':
        options = [set(['a', 'b']), set(['c', 'd']), set(['e', 'f'])]
        solutions = all_permutations(options)
        self.assertCountEqual(solutions, [
            ['a', 'c', 'e'],
            ['a', 'd', 'e'],
            ['a', 'c', 'f'],
            ['a', 'd', 'f'],
            ['b', 'c', 'e'],
            ['b', 'd', 'e'],
            ['b', 'c', 'f'],
            ['b', 'd', 'f']
        ])


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
