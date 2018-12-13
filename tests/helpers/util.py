import sys
from contextlib import contextmanager
from io import StringIO

import nose


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@contextmanager
def cl_args(*args):
    old_args = sys.argv
    try:
        sys.argv[1:] = args
        yield
    finally:
        sys.argv = old_args


def command_line_args(*argv):
    def deco(test):
        @nose.tools.make_decorator(test)
        def reassign(*args, **kwargs):
            with cl_args(*argv):
                return test(*args, **kwargs)
        return reassign
    return deco
