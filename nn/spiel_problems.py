"""
nn.spiel_problems

Base definitions for problems handled by SPieL. Handles interaction with the
filesystem.
"""
import os
from shutil import copyfile
from pathlib import Path
import tarfile

import tensorflow as tf
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import multi_problem

from . import segmentation, recognition, util
from . import data as data_module


DATA_PATH = os.path.split(os.path.realpath(data_module.__file__))[0]


def _data_filenames(language_code, dataset_split, tmp_dir,
                    with_segmented=True):
    """
    The names of the files that contain training data

    :param tmp_dir: The location of the directory holding temporary
                    tensor2tensor data
    :param with_segmented: Whether to include the file with segmented data
    :return: If with_segmented is True, a list of unsegmented and segmented
             file name tuples. If it is False, a list of unsegmented file names
    """
    if dataset_split == problem.DatasetSplit.TRAIN:
        data_group = 'train'
    else:
        data_group = 'dev'

    listing = [
        (os.path.join(tmp_dir, f'{language_code}-corpus', data_group,
                      f'{language_code}-{data_group}.original'),
         os.path.join(tmp_dir, f'{language_code}-corpus', data_group,
                      f'{language_code}-{data_group}.segmented'))
    ]

    if not with_segmented:
        return [group[0] for group in listing]
    return listing


def _maybe_copy_corpus(language_code, tmp_dir):
    """
    Unzips the data file into tensor2tensor's temporary directory if it isn't
    already there
    """
    corpus_filename = language_code + '-corpus.tgz'
    corpus_filepath = os.path.join(tmp_dir, corpus_filename)
    data_tgz = os.path.join(DATA_PATH, corpus_filename)

    if os.path.exists(corpus_filepath):
        return

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    copyfile(data_tgz, corpus_filepath)
    with tarfile.open(corpus_filepath, 'r:gz') as corpus_tar:
        corpus_tar.extractall(tmp_dir)


class SpielProblem:
    """
    SPieL problems use a common filesystem which requires a language_code. This
    class enforces that interface.
    """
    # pylint: disable=R0903
    @property
    def language_code(self):
        """
        The code for the language used in the filesystem
        """
        raise NotImplementedError()


class LanguageModel(text_problems.Text2SelfProblem, util.SingleProcessProblem,
                    SpielProblem):
    """
    Base SPieL language model. Just requires a language_code to work.
    """
    # pylint: disable=W0223
    @property
    def is_generate_per_split(self):
        """
        Train and eval data come separately
        """
        return True

    @property
    def vocab_type(self):
        """
        Base the language model on characters
        """
        return text_problems.VocabType.CHARACTER

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Samples come from the unsegmented data files
        """
        _maybe_copy_corpus(self.language_code, tmp_dir)
        files = _data_filenames(self.language_code, dataset_split,
                                tmp_dir, False)
        for filepath in files:
            tf.logging.info(f'filepath = {filepath}')
            for line in tf.gfile.Open(filepath):
                yield {'targets': line}


class SegmentationProblem(segmentation.SegmentationProblem, SpielProblem):
    """
    Base SPieL segmentation problem. Just requires a language_code to work.
    """
    # pylint: disable=W0223
    def source_data_files(self, data_dir, tmp_dir, dataset_split):
        """
        Data files are the unsegmented (source) and segmented (target) files
        """
        _maybe_copy_corpus(self.language_code, tmp_dir)
        return _data_filenames(self.language_code, dataset_split, tmp_dir)


class RecognitionProblem(recognition.RecognitionProblem, SpielProblem):
    """
    Base SPieL recognition problem.
    """
    # pylint: disable=W0223
    def source_data_files(self, data_dir, tmp_dir, dataset_split):
        """
        Only recognize the unsegmented data
        """
        _maybe_copy_corpus(self.language_code, tmp_dir)
        return _data_filenames(self.language_code, problem.DatasetSplit.TRAIN,
                               tmp_dir, with_segmented=False)


class MultitaskProblem(multi_problem.MultiProblem, util.SingleProcessProblem):
    """
    Base SPieL multitask problem
    """
    def __init__(self, was_reversed=False, was_copy=False):
        super(MultitaskProblem, self).__init__(was_reversed, was_copy)
        self.task_list += self.problem_list

    @property
    def problem_list(self):
        """
        The list of tasks this problem handles
        """
        raise NotImplementedError()

    @property
    def vocab_type(self):
        """
        Use character level tokens
        """
        return text_problems.VocabType.CHARACTER

    def get_task_id(self, task_idx):
        """
        Helper method for determining a task's ID via the REPL. Simulates vocab
        initialization so the correct ID can be determined.

        :param task_idx: The index of the task (i.e. in the order it was
                         entered in __init__)
        :return: The task_id of the subtask
        """
        hparams = self.task_list[0].get_hparams()
        vocab_size = hparams.vocabulary['targets'].vocab_size
        self.update_task_ids(vocab_size)
        return self.task_list[task_idx].task_id
