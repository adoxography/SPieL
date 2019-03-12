from tensor2tensor.data_generators import problem


class SingleProcessProblem(problem.Problem):
    @property
    def num_generate_tasks(self):
        """
        Unused since multiprocess_generate is False
        """

    @property
    def prepare_to_generate(self):
        """
        Unused since multiprocess_generate is False
        """
