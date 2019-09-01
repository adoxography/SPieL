"""
nn.aztec

tensor2tensor problem definitions for various Aztec languages, using corpora
from Kann et al. 2018
"""
from tensor2tensor.utils import registry
from . import spiel_problems


@registry.register_problem
class SegmentMexicanero(spiel_problems.SegmentationProblem):
    """
    Segmentation task for Mexicanero
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        return 'azd'


@registry.register_problem
class SegmentNahuatl(spiel_problems.SegmentationProblem):
    """
    Segmentation task for Nahuatl
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        return 'nah'


@registry.register_problem
class SegmentWixarika(spiel_problems.SegmentationProblem):
    """
    Segmentation task for Wixarika
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        return 'hch'


@registry.register_problem
class SegmentYoremNokki(spiel_problems.SegmentationProblem):
    """
    Segmentation task for Yorem Nokki
    """
    # pylint: disable=W0223
    @property
    def language_code(self):
        return 'mfy'
