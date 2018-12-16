# coding: spec
from util import captured_output, command_line_args
from spiel.command_line import main


describe 'main':
    @command_line_args('--train',
                       'tests/test_command_line/resources/train_instances.txt')
    it 'runs if only training instances are supplied':
        with captured_output() as (out, err):
            main()
        output = out.getvalue().strip()
        error = err.getvalue().strip()

        self.assertEqual(output, """Train results
Accuracy: 0.8""")

    @command_line_args('--train',
                       'tests/test_command_line/resources/train_instances.txt',
                       '--test',
                       'tests/test_command_line/resources/test_instances.txt')
    it 'runs if train and test instances are supplied':
        with captured_output() as (out, err):
            main()
        output = out.getvalue().strip()
        self.assertEqual(output, """Train results
Accuracy: 0.8

Test results
Shape 'fo' segmented to 'f/A-o/B'.
Accuracy: 0.5""")
