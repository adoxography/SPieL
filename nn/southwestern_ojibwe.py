"""
nn.southwestern_ojibwe

tensor2tensor problem definitions for Southwestern Ojibwe
"""
from tensor2tensor.utils import registry

from . import spiel_problems


@registry.register_problem
class LanguageModelSouthwesternOjibwe(spiel_problems.LanguageModel):
    """
    Basic language model for Southwestern Ojibwe
    """
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        return 'swo'


@registry.register_problem
class SegmentSouthwesternOjibwe(spiel_problems.SegmentationProblem):
    """
    Segmentation task for Southwestern Ojibwe
    """
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        return 'swo'


@registry.register_problem
class RecognizeSouthwesternOjibwe(spiel_problems.RecognitionProblem):
    """
    Recognizes the characters of proto-algonquian
    """
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        return 'swo'

    @property
    def min_size(self):
        """
        The smallest exemplified example is 4 characters long
        """
        return 5

    @property
    def max_size(self):
        """
        The longest exemplified example is 20 characters long
        """
        return 26

    @property
    def num_train_instances(self):
        """
        The Southwestern Ojibwe training corpus is around 500 examples, so use
        around 8x that.
        """
        return 4000


@registry.register_problem
class MultitaskSouthwesternOjibwe(spiel_problems.MultitaskProblem):
    """
    Defines a multitask problem that trains a language model and a segmentation
    task at the same time
    """
    @property
    def problem_list(self):
        """
        The list of tasks this problem handles
        """
        return [
            LanguageModelSouthwesternOjibwe(),
            SegmentSouthwesternOjibwe()
        ]


@registry.register_problem
class MultitaskSouthwesternOjibweWithNoise(spiel_problems.MultitaskProblem):
    """
    Defines a multitask problem that trains a language model, a segmentation
    task, and a recognition task
    """
    @property
    def problem_list(self):
        """
        The list of tasks this problem handles
        """
        return [
            LanguageModelSouthwesternOjibwe(),
            SegmentSouthwesternOjibwe(),
            RecognizeSouthwesternOjibwe()
        ]
