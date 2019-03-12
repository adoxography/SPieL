"""
nn.util

Utility class for working with t2t
"""
from tensor2tensor.data_generators import problem


class SingleProcessProblem(problem.Problem):
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

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        """
        Just pass this one down to the child
        """
        raise NotImplementedError()

    @property
    def num_training_examples(self):
        """
        Just pass this one down to the child
        """
        raise NotImplementedError()
