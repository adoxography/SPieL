# coding: spec
from spiel.data import Instance, load, load_file, ParseError


describe 'Instance':
    describe 'annotations':
        it 'combines its segments and labels into annotations':
            segments = ['f', 'o', 'o']
            labels = ['B', 'A', 'R']
            instance = Instance('foo', segments, labels)
            self.assertEqual(instance.annotations,
                             [('f', 'B'), ('o', 'A'), ('o', 'R')])

    describe 'annotation_string':
        it "presents the instance's annotation as a string":
            instance = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            self.assertEqual(instance.annotation_string(),
                             'f/B-o/A-o/R')

    describe '__eq__':
        it 'equals another instance when its properties match':
            instance_1 = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            instance_2 = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            self.assertEqual(instance_1, instance_2)

        it 'does not equal another instance when its properties do not match':
            instance_1 = Instance('foo', ['f', 'o', ], ['B', 'A'])
            instance_2 = Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])
            self.assertNotEqual(instance_1, instance_2)


describe 'load_file':
    it 'loads instances from a file':
        instances = load_file('tests/test_data/resources/instances.txt')
        self.assertEqual(instances,
                         [Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R']),
                          Instance('ba', ['b', 'a',], ['A', 'B'])])


describe 'load':
    it 'reads an instance from a list':
        instances = load(['foo', 'f o o', 'B A R'])
        self.assertEqual(instances,
                         [Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R'])])

    it 'reads multiple instances from a list':
        instances = load(['foo', 'f o o', 'B A R', '', 'ba', 'b a', 'A B'])
        self.assertEqual(instances,
                         [Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R']),
                          Instance('ba', ['b', 'a',], ['A', 'B'])])

    it 'raises an error if more than three lines are present':
        with self.assertRaises(ParseError) as e:
            load(['foo', 'f o o', 'B A R', 'baz'])
        self.assertEqual(str(e.exception), 'Too many fields provided')

    it 'raises an error if the number of segments do not match the number of labels':
        with self.assertRaises(ParseError):
            load(['foo', 'f o o', 'b a'])

    it 'ensures that its shape is not empty':
        with self.assertRaises(ParseError):
            load(['', 'f o o', 'b a r'])

    it 'handles multiple blank lines':
        instances = load(['foo', 'f o o', 'B A R', '', '', 'ba', 'b a', 'A B'])
        self.assertEqual(instances,
                         [Instance('foo', ['f', 'o', 'o'], ['B', 'A', 'R']),
                          Instance('ba', ['b', 'a',], ['A', 'B'])])

    context 'strict mode':
        it 'raises an error if fewer than three lines are present':
            with self.assertRaises(ParseError) as e:
                load(['foo'], strict=True)
            self.assertEqual(str(e.exception), 'Not enough fields provided')

        it 'ensures that its segments are not empty':
            with self.assertRaises(ParseError):
                load(['foo', '', ''], strict=True)

    context 'not strict mode':
        it 'reads an instance with just a shape':
            instances = load(['foo'], strict=False)
            self.assertEqual(instances, [Instance('foo', None, None)])

        it 'reads multiple instances of just shapes':
            instances = load(['foo', '', 'bar'], strict=False)
            self.assertEqual(instances,
                             [Instance('foo', None, None),
                              Instance('bar', None, None)])

        it 'reads mixed shapes and full instances':
            instances = load(['foo', '', 'ba', 'b a', 'A B', '', 'baz'],
                             strict=False)
            self.assertEqual(instances,
                             [Instance('foo', None, None),
                              Instance('ba', ['b', 'a'], ['A', 'B']),
                              Instance('baz', None, None)])

        it 'raises an error if only two lines are present':
            with self.assertRaises(ParseError):
                load(['foo', 'f o o'], strict=True)
