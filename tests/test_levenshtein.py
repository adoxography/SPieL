# coding: spec

import nose2
from mspl import levenshtein


describe 'distance':
    it 'returns 0 for identical strings':
        distance = levenshtein.distance('foo', 'foo')
        assert distance == 0

    it 'returns 1 for strings with one substitution':
        distance = levenshtein.distance('bar', 'baz')
        assert distance == 1

    it 'returns 1 for strings with one deletion':
        distance = levenshtein.distance('foo', 'fo')
        assert distance == 1

    it 'returns 1 for strings with one insertion':
        distance = levenshtein.distance('fo', 'foo')
        assert distance == 1

    it 'returns 3 for strings with exactly one substitution, deletion, and insertion':
        distance = levenshtein.distance('banana', 'faanaa')
        assert distance == 3


describe 'operations':
    it 'returns an empty generator for identical strings':
        operations = levenshtein.operations('foo', 'foo')
        assert list(operations) == []

    it 'returns a ReplaceOperation for strings with one substitution':
        operations = levenshtein.operations('bar', 'baz')
        assert next(operations) == levenshtein.ReplaceOperation(2, 2)
        assert list(operations) == []

    it 'returns a DeleteOperation for strings with one deletion':
        operations = levenshtein.operations('bar', 'ba')
        assert next(operations) == levenshtein.DeleteOperation(2)
        assert list(operations) == []

    it 'returns an InsertOperation for strings with one deletion':
        operations = levenshtein.operations('ba', 'bar')
        assert next(operations) == levenshtein.InsertOperation(1, 2)
        assert list(operations) == []

    it 'returns operations for each needed operation':
        operations = levenshtein.operations('abcde', 'fcdeg')
        assert next(operations) == levenshtein.InsertOperation(4, 4)
        assert next(operations) == levenshtein.DeleteOperation(1)
        assert next(operations) == levenshtein.ReplaceOperation(0, 0)
        assert list(operations) == []


describe 'apply_operations':
    it 'does nothing if there are no operations':
        operations = []
        output = levenshtein.apply_operations('foo', 'bar', operations)
        assert output == 'foo'

    it 'applies a list of operations':
        operations = [
            levenshtein.InsertOperation(2, 1),
            levenshtein.ReplaceOperation(0, 0)
        ]
        output = levenshtein.apply_operations('foo', 'bar', operations)
        assert output == 'booa'

    it 'excludes Nones from the final output':
        operations = [levenshtein.DeleteOperation(1)]
        output = levenshtein.apply_operations(['f', 'o', 'o'], 'bar', operations)
        assert output == ['f', 'o']

    it 'returns a list if a list was passed in':
        operations = [
            levenshtein.InsertOperation(2, 1),
            levenshtein.ReplaceOperation(0, 0)
        ]
        output = levenshtein.apply_operations(['f', 'o', 'o'], 'bar', operations)
        assert output == ['b', 'o', 'o', 'a']

    it 'applies multiple operations on the same index':
        operations = [
            levenshtein.InsertOperation(1, 0),
            levenshtein.ReplaceOperation(1, 2),
            levenshtein.DeleteOperation(1)
        ]

        output = levenshtein.apply_operations('foo', 'bar', operations)
        assert output == 'fbo'


describe 'ReplaceOperation':
    describe 'apply':
        it 'replaces a character in an origin list with a character in a reference list':
            operation = levenshtein.ReplaceOperation(1, 3)
            origin = ['f', 'o', 'o']
            operation.apply(origin, ['h', 'e', 'l', 'l', 'o'])
            assert origin == ['f', 'l', 'o']

        it 'handles nested origin lists':
            operation = levenshtein.ReplaceOperation(1, 3)
            origin = ['f', ['o', 'b'], 'a']
            operation.apply(origin, 'hello')
            assert origin == ['f', ['l', 'b'], 'a']


describe 'InsertOperation':
    describe 'apply':
        it 'inserts a character in the origin list using a reference':
            operation = levenshtein.InsertOperation(2, 1)
            origin = ['f', 'o', 'o']
            operation.apply(origin, ['h', 'e', 'l', 'l', 'o'])
            assert origin == ['f', 'o', ['o', 'e']]

        it 'handles nested origin lists':
            operation = levenshtein.InsertOperation(1, 2)
            origin = ['f', ['o', 'b'], 'a']
            operation.apply(origin, 'hello')
            assert origin == ['f', [['o', 'l'], 'b'], 'a']


describe 'DeleteOperation':
    describe 'apply':
        it 'deletes a character in the origin list':
            operation = levenshtein.DeleteOperation(1)
            origin = ['f', 'o', 'o']
            operation.apply(origin)
            assert origin == ['f', None, 'o']

        it 'handles nested origin lists':
            operation = levenshtein.DeleteOperation(1)
            origin = ['f', ['o', 'b'], 'a']
            operation.apply(origin)
            assert origin == ['f', [None, 'b'], 'a']
