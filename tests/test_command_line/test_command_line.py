# coding: spec
from util import captured_output, command_line_args
from spiel.command_line import main


describe 'main':
    @command_line_args()
    it 'exits if no training file is supplied':
        with self.assertRaises(ValueError):
            with captured_output():
                main()

    @command_line_args('--train',
                       'tests/test_command_line/resources/train_instances.txt')
    it 'runs if only training instances are supplied':
        with captured_output() as (out, err):
            main()
        output = out.getvalue().strip()
        self.assertEqual(output, """Train results
Shape: foo
Predicted: f/A-o/B-o/B\tActual: f/A-o/B-o/B
Shape: ba
Predicted: ba/C\tActual: ba/C
Shape: bar
Predicted: ba/C-r/D\tActual: ba/C-r/D
Shape: baz
Predicted: ba/C-z/E\tActual: ba/C-z/E""")

    @command_line_args('--train',
                       'tests/test_command_line/resources/train_instances.txt',
                       '--test',
                       'tests/test_command_line/resources/test_instances.txt')
    it 'runs if train and test instances are supplied':
        with captured_output() as (out, err):
            main()
        output = out.getvalue().strip()
        self.assertEqual(output, """Train results
Shape: foo
Predicted: f/A-o/B-o/B\tActual: f/A-o/B-o/B
Shape: ba
Predicted: ba/C\tActual: ba/C
Shape: bar
Predicted: ba/C-r/D\tActual: ba/C-r/D
Shape: baz
Predicted: ba/C-z/E\tActual: ba/C-z/E

Test results
Shape: forba
Predicted: f/A-o/B-r/D-ba/C\tActual: f/A-o/B-r/D-ba/C""")
