# coding: spec
from spiel.util import all_permutations, pad, grouper


describe 'all_permutations':
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


describe 'pad':
    it 'adds a character to either side of a string':
        padded = pad('foo', '_', 1)
        self.assertEqual(padded, '_foo_')

    it 'adds multiple characters to either side of a string':
        padded = pad('foo', '_', 3)
        self.assertEqual(padded, '___foo___')

    it 'adds elements to either side of a list':
        padded = pad(['f', 'oo'], '_', 3)
        self.assertEqual(padded, ['_', '_', '_', 'f', 'oo', '_', '_', '_'])


describe 'grouper':
    it 'fills missing values with fillvalue':
        iterable = 'abcdefg'
        iterations = list(grouper(4, iterable, fillvalue='foo'))
        self.assertEqual(iterations[-1][-1], 'foo')
