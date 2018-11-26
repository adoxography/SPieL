# coding: spec

from unittest import TestCase
from mspl import levenshtein
from mspl.levenshtein import DeleteOperation, InsertOperation, ReplaceOperation


describe TestCase 'distance':
    it 'returns 0 for identical strings':
        distance = levenshtein.distance('foo', 'foo')
        self.assertEqual(distance, 0)

    it 'returns 1 for strings with one substitution':
        distance = levenshtein.distance('bar', 'baz')
        self.assertEqual(distance, 1)

    it 'returns 1 for strings with one deletion':
        distance = levenshtein.distance('foo', 'fo')
        self.assertEqual(distance, 1)

    it 'returns 1 for strings with one insertion':
        distance = levenshtein.distance('fo', 'foo')
        self.assertEqual(distance, 1)

    it 'returns 3 for strings with exactly one substitution, deletion, and insertion':
        distance = levenshtein.distance('banana', 'faanaa')
        self.assertEqual(distance, 3)


describe TestCase 'operations':
    it 'returns an empty generator for identical strings':
        operations = levenshtein.operations('foo', 'foo')
        self.assertEqual(list(operations), [])

    it 'returns a ReplaceOperation for strings with one substitution':
        operations = levenshtein.operations('bar', 'baz')
        self.assertEqual(list(operations), [ReplaceOperation(2, 2)])

    it 'returns a DeleteOperation for strings with one deletion':
        operations = levenshtein.operations('bar', 'ba')
        self.assertEqual(list(operations), [DeleteOperation(2)])

    it 'returns an InsertOperation for strings with one deletion':
        operations = levenshtein.operations('ba', 'bar')
        self.assertEqual(list(operations), [InsertOperation(1, 2)])

    it 'returns operations for each needed operation':
        operations = levenshtein.operations('abcde', 'fcdeg')
        self.assertEqual(list(operations), [
            InsertOperation(4, 4),
            DeleteOperation(1),
            ReplaceOperation(0, 0)
        ])


describe TestCase 'apply_operations':
    it 'does nothing if there are no operations':
        operations = []
        output = levenshtein.apply_operations('foo', 'bar', operations)
        self.assertEqual(output, 'foo')

    it 'applies a list of operations':
        operations = [
            InsertOperation(2, 1),
            ReplaceOperation(0, 0)
        ]
        output = levenshtein.apply_operations('foo', 'bar', operations)
        self.assertEqual(output, 'booa')

    it 'excludes Nones from the final output':
        operations = [DeleteOperation(1)]
        output = levenshtein.apply_operations(['f', 'o', 'o'], 'bar', operations)
        self.assertEqual(output, ['f', 'o'])

    it 'returns a list if a list was passed in':
        operations = [
            InsertOperation(2, 1),
            ReplaceOperation(0, 0)
        ]
        output = levenshtein.apply_operations(['f', 'o', 'o'], 'bar', operations)
        self.assertEqual(output, ['b', 'o', 'o', 'a'])

    it 'applies multiple operations on the same index':
        operations = [
            InsertOperation(1, 0),
            ReplaceOperation(1, 2),
            DeleteOperation(1)
        ]

        output = levenshtein.apply_operations('foo', 'bar', operations)
        self.assertEqual(output, 'fbo')


describe TestCase 'annotate':
    it 'does nothing if no operations are provided':
        annotation = levenshtein.annotate(['f', 'o', 'o'], '', [])
        self.assertEqual(annotation, ['f', 'o', 'o'])

    it 'adds multiple annotations':
        operations = [
            ReplaceOperation(0, 0),
            InsertOperation(1, 2),
            DeleteOperation(2)
        ]
        annotation = levenshtein.annotate(['f', 'o', 'o'], 'bar', operations)
        self.assertEqual(annotation, ['f+R(f,b)', 'o+I(r)+D(o)'])

    it 'can annotate strings':
        operations = [
            ReplaceOperation(0, 0),
            InsertOperation(1, 2),
            DeleteOperation(2)
        ]
        annotation = levenshtein.annotate('foo', 'bar', operations)
        self.assertEqual(annotation, ['f+R(f,b)', 'o+I(r)+D(o)'])

    it 'can use an alternate list as a starting point':
        operations = [
            ReplaceOperation(0, 0),
            InsertOperation(1, 2),
            DeleteOperation(2)
        ]
        annotation = ['b', 'a', 'z']
        annotation = levenshtein.annotate('foo', 'bar', operations, annotation)
        self.assertEqual(annotation, ['b+R(f,b)', 'a+I(r)+D(o)'])


describe TestCase 'ReplaceOperation':
    describe 'apply':
        it 'replaces a character in an origin list with a character in a reference list':
            operation = ReplaceOperation(1, 3)
            origin = ['f', 'o', 'o']
            operation.apply(origin, ['h', 'e', 'l', 'l', 'o'])
            self.assertEqual(origin, ['f', 'l', 'o'])

        it 'handles nested origin lists':
            operation = ReplaceOperation(1, 3)
            origin = ['f', ['o', 'b'], 'a']
            operation.apply(origin, 'hello')
            self.assertEqual(origin, ['f', ['l', 'b'], 'a'])

        it 'annotates annotations':
            operation = ReplaceOperation(0, 0)
            annotation = ['f', 'o', 'o']
            operation.annotate(annotation, 'foo', 'bar')
            self.assertEqual(annotation, ['f+R(f,b)', 'o', 'o'])

    it 'can be represented':
        operation = ReplaceOperation(1, 5)
        self.assertEqual(str(operation), 'Replace 1 with 5')


describe TestCase 'InsertOperation':
    describe 'apply':
        it 'inserts a character in the origin list using a reference':
            operation = InsertOperation(2, 1)
            origin = ['f', 'o', 'o']
            operation.apply(origin, ['h', 'e', 'l', 'l', 'o'])
            self.assertEqual(origin, ['f', 'o', ['o', 'e']])

        it 'handles nested origin lists':
            operation = InsertOperation(1, 2)
            origin = ['f', ['o', 'b'], 'a']
            operation.apply(origin, 'hello')
            self.assertEqual(origin, ['f', [['o', 'l'], 'b'], 'a'])

        it 'annotates annotations':
            operation = InsertOperation(1, 2)
            annotation = ['f', 'o', 'o']
            operation.annotate(annotation, 'foo', 'bar')
            self.assertEqual(annotation, ['f', 'o+I(r)', 'o'])

    it 'can be represented':
        operation = InsertOperation(1, 5)
        self.assertEqual(str(operation), 'Insert 5 at position 1')


describe TestCase 'DeleteOperation':
    describe 'apply':
        it 'deletes a character in the origin list':
            operation = DeleteOperation(1)
            origin = ['f', 'o', 'o']
            operation.apply(origin)
            self.assertEqual(origin, ['f', None, 'o'])

        it 'handles nested origin lists':
            operation = DeleteOperation(1)
            origin = ['f', ['o', 'b'], 'a']
            operation.apply(origin)
            self.assertEqual(origin, ['f', [None, 'b'], 'a'])

        it 'annotates annotations':
            operation = DeleteOperation(1)
            annotation = ['b', 'a', 'r']
            operation.annotate(annotation, 'bar', 'foo')
            self.assertEqual(annotation, ['b', '+D(a)', 'r'])

        it 'annotates annotations when the affected segment has already been annotated':
            operation = DeleteOperation(1)
            annotation = ['b', 'a+I(f)', 'r']
            operation.annotate(annotation, 'bar', 'foo')
            self.assertEqual(annotation, ['b', '+D(a)+I(f)', 'r'])

    it 'can be represented':
        operation = DeleteOperation(3)
        self.assertEqual(str(operation), 'Delete at position 3')
