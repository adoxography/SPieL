"""
nn.util

Utility class for working with t2t
"""
from abc import ABCMeta
from tensor2tensor.data_generators import problem


class SingleProcessProblem(problem.Problem, metaclass=ABCMeta):
    """
    Mixin to mark a class as using a single process and therefore not needing
    to override num_generate_tasks or prepare_to_generate
    """
    @property
    def num_generate_tasks(self):
        """
        Unused since multiprocess_generate is False
        """

    @property
    def prepare_to_generate(self, data_dir, tmp_dir):
        """
        Unused since multiprocess_generate is False
        """
