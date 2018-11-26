# coding: spec
from unittest import TestCase
from mspl.util import all_permutations


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
