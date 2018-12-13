# coding: spec
from unittest import TestCase
from spiel.data import Instance, load


describe TestCase 'Instance':
    it 'raises an error if the number of segments do not match the number of labels':
        with self.assertRaises(ValueError):
            Instance('foo', ['f', 'o', 'o'], ['b', 'a'])

    it 'ensures that its shape is not empty':
        with self.assertRaises(ValueError):
            Instance('', ['f', 'o', 'o'], ['b', 'a', 'r'])

    it 'ensures that its segments are not empty':
        with self.assertRaises(ValueError):
            Instance('foo', [], [])

    describe 'annotations':
        it 'combines its segments and labels into annotations':
            segments = ['f', 'o', 'o']
            labels = ['B', 'A', 'R']
            instance = Instance('foo', segments, labels)
            self.assertEqual(instance.annotations,
                             [('f', 'B'), ('o', 'A'), ('o', 'R')])

    describe '__eq__':
        it 'equals another instance when its properties match':
            instance_1 = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            instance_2 = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            self.assertEqual(instance_1, instance_2)

        it 'does not equal another instance when its properties do not match':
            instance_1 = Instance('foo', ['f', 'o', ], ['B', 'A'])
            instance_2 = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            self.assertNotEqual(instance_1, instance_2)


describe TestCase 'load':
    it 'reads an instance from a file':
        instances = load('tests/test_data/resources/instance.txt')
        self.assertEqual(instances,
                         [Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])])

    it 'reads multiple instances from a file':
        instances = load('tests/test_data/resources/instances.txt')
        self.assertEqual(instances,
                         [Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R']),
                          Instance('ba', ['b', 'a',], ['A', 'B'])])
