# coding: spec
from unittest import TestCase
from spiel.command_line import main

from util import captured_output, command_line_args


describe TestCase 'main':
    @command_line_args('tests/test_command_line/resources/instances.txt')
    it 'runs':
        with captured_output() as (out, err):
            main()
        output = out.getvalue().strip()
        self.assertEqual(output, """Shape: foo
Predicted: f/A-o/B-o/B\tActual: f/A-o/B-o/B
Shape: ba
Predicted: ba/C\tActual: ba/C
Shape: bar
Predicted: ba/C-r/D\tActual: ba/C-r/D
Shape: baz
Predicted: ba/C-z/E\tActual: ba/C-z/E""")
