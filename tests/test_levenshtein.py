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


describe 'annotate':
    it 'does nothing if no operations are provided':
        annotation = levenshtein.annotate(['f', 'o', 'o'], '', [])
        assert annotation == ['f', 'o', 'o']

    it 'adds multiple annotations':
        operations = [
            levenshtein.ReplaceOperation(0, 0),
            levenshtein.InsertOperation(1, 2),
            levenshtein.DeleteOperation(2)
        ]
        annotation = levenshtein.annotate(['f', 'o', 'o'], 'bar', operations)
        assert annotation == ['f+R(f,b)', 'o+I(r)+D(o)']

    it 'can annotate strings':
        operations = [
            levenshtein.ReplaceOperation(0, 0),
            levenshtein.InsertOperation(1, 2),
            levenshtein.DeleteOperation(2)
        ]
        annotation = levenshtein.annotate('foo', 'bar', operations)
        assert annotation == ['f+R(f,b)', 'o+I(r)+D(o)']

    it 'can use an alternate list as a starting point':
        operations = [
            levenshtein.ReplaceOperation(0, 0),
            levenshtein.InsertOperation(1, 2),
            levenshtein.DeleteOperation(2)
        ]
        annotation = ['b', 'a', 'z']
        annotation = levenshtein.annotate('foo', 'bar', operations, annotation)
        assert annotation == ['b+R(f,b)', 'a+I(r)+D(o)']


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

        it 'annotates annotations':
            operation = levenshtein.ReplaceOperation(0, 0)
            annotation = ['f', 'o', 'o']
            operation.annotate(annotation, 'foo', 'bar')
            assert annotation == ['f+R(f,b)', 'o', 'o']


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

        it 'annotates annotations':
            operation = levenshtein.InsertOperation(1, 2)
            annotation = ['f', 'o', 'o']
            operation.annotate(annotation, 'foo', 'bar')
            assert annotation == ['f', 'o+I(r)', 'o']


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

        it 'annotates annotations':
            operation = levenshtein.DeleteOperation(1)
            annotation = ['b', 'a', 'r']
            operation.annotate(annotation, 'bar', 'foo')
            assert annotation == ['b', '+D(a)', 'r']

        it 'annotates annotations when the affected segment has already been annotated':
            operation = levenshtein.DeleteOperation(1)
            annotation = ['b', 'a+I(f)', 'r']
            operation.annotate(annotation, 'bar', 'foo')
            assert annotation == ['b', '+D(a)+I(f)', 'r']
