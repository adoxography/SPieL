# coding: spec
import sys
from io import StringIO
from contextlib import contextmanager
from unittest import TestCase
from spiel.command_line import main


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


describe TestCase 'main':
    it 'runs':
        with captured_output() as (out, err):
            main('tests/test_command_line/resources/instances.txt')
        output = out.getvalue().strip()
        self.assertEqual(output, """Shape: foo
Predicted: f/A-o/B-o/B\tActual: f/A-o/B-o/B
Shape: ba
Predicted: ba/C\tActual: ba/C
Shape: bar
Predicted: ba/C-r/D\tActual: ba/C-r/D
Shape: baz
Predicted: ba/C-z/E\tActual: ba/C-z/E""")
