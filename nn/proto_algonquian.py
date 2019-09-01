"""
nn.proto_algonquian

tensor2tensor problem definitions for Proto-Algonquian
"""
from tensor2tensor.utils import registry

from . import spiel_problems


@registry.register_problem
class LanguageModelProtoAlgonquian(spiel_problems.LanguageModel):
    """
    Basic language model for Proto-Algonquian
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        return 'pa'


@registry.register_problem
class SegmentProtoAlgonquian(spiel_problems.SegmentationProblem):
    """
    Segmentation task for Proto-Algonquian
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        return 'pa'


@registry.register_problem
class RecognizeProtoAlgonquian(spiel_problems.RecognitionProblem):
    """
    Recognizes the characters of proto-algonquian
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        return 'pa'

    @property
    def min_size(self):
        """
        The smallest exemplified example is 4 characters long
        """
        return 4

    @property
    def max_size(self):
        """
        The longest exemplified example is 20 characters long
        """
        return 20

    @property
    def num_train_instances(self):
        """
        The Proto-Algonquian training corpus is around 200 examples, so use
        around 8x that.
        """
        return 1500


@registry.register_problem
class MultitaskProtoAlgonquian(spiel_problems.MultitaskProblem):
    """
    Defines a multitask problem that trains a language model and a segmentation
    task at the same time
    """
    # pylint: disable=W0223
    @property
    def problem_list(self):
        """
        The list of tasks this problem handles
        """
        return [
            LanguageModelProtoAlgonquian(),
            SegmentProtoAlgonquian()
        ]


@registry.register_problem
class MultitaskProtoAlgonquianWithNoise(spiel_problems.MultitaskProblem):
    """
    Defines a multitask problem that trains a language model, a segmentation
    task, and a recognition task
    """
    # pylint: disable=W0223
    @property
    def problem_list(self):
        """
        The list of tasks this problem handles
        """
        return [
            LanguageModelProtoAlgonquian(),
            SegmentProtoAlgonquian(),
            RecognizeProtoAlgonquian()
        ]
