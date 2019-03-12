"""
nn.proto_algonquian

tensor2tensor problem definitions for Proto-Algonquian
"""
import os
import tarfile
from shutil import copyfile
from pathlib import Path

import tensorflow as tf
from tensor2tensor.data_generators import problem, text_problems, multi_problem
from tensor2tensor.utils import registry

from . import segmentation, recognition
from . import util
from . import data as data_module


DATA_PATH = os.path.split(os.path.realpath(data_module.__file__))[0]
DATA_TGZ = os.path.join(DATA_PATH, 'pa-corpus.tgz')


def _train_data_filenames(tmp_dir, with_segmented=True):
    """
    The names of the files that contain training data

    :param tmp_dir: The location of the directory holding temporary
                    tensor2tensor data
    :param with_segmented: Whether to include the file with segmented data
    :return: If with_segmented is True, a list of unsegmented and segmented
             file name tuples. If it is False, a list of unsegmented file names
    """
    listing = [
        (os.path.join(tmp_dir, 'pa-corpus', 'train', 'pa-train.original'),
         os.path.join(tmp_dir, 'pa-corpus', 'train', 'pa-train.segmented'))
    ]

    if not with_segmented:
        return [group[0] for group in listing]
    return listing


def _dev_data_filenames(tmp_dir, with_segmented=True):
    """
    The names of the files that contain dev data

    :param tmp_dir: The location of the directory holding temporary
                    tensor2tensor data
    :param with_segmented: Whether to include the file with segmented data
    :return: If with_segmented is True, a list of unsegmented and segmented
             file name tuples. If it is False, a list of unsegmented file names
    """
    listing = [
        (os.path.join(tmp_dir, 'pa-corpus', 'dev', 'pa-dev.original'),
         os.path.join(tmp_dir, 'pa-corpus', 'dev', 'pa-dev.segmented'))
    ]

    if not with_segmented:
        return [group[0] for group in listing]
    return listing


def _maybe_copy_corpus(tmp_dir):
    """
    Unzips the data file into tensor2tensor's temporary directory if it isn't
    already there
    """
    corpus_filename = os.path.basename(DATA_TGZ)
    corpus_filepath = os.path.join(tmp_dir, corpus_filename)

    if os.path.exists(corpus_filepath):
        return

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    copyfile(DATA_TGZ, corpus_filepath)
    with tarfile.open(corpus_filepath, 'r:gz') as corpus_tar:
        corpus_tar.extractall(tmp_dir)


@registry.register_problem
class LanguageModelProtoAlgonquian(text_problems.Text2SelfProblem,
                                   util.SingleProcessProblem):
    """
    Basic language model for Proto-Algonquian
    """
    @property
    def is_generate_per_split(self):
        """
        Train and eval data come separately
        """
        return True

    @property
    def num_training_examples(self):
        """
        Unused since is_generate_per_split is True
        """

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Samples come from the unsegmented data files
        """
        split_files = {
            problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir, False),
            problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir, False)
        }

        _maybe_copy_corpus(tmp_dir)
        files = split_files[dataset_split]
        for filepath in files:
            tf.logging.info(f'filepath = {filepath}')
            for line in tf.gfile.Open(filepath):
                yield {'targets': line}

    @property
    def vocab_type(self):
        """
        Base the language model on characters
        """
        return text_problems.VocabType.CHARACTER


@registry.register_problem
class SegmentProtoAlgonquian(segmentation.SegmentationProblem):
    """
    Segmentation task for Proto-Algonquian
    """
    def source_data_files(self, data_dir, tmp_dir, dataset_split):
        """
        Data files are the unsegmented (source) and segmented (target) files
        """
        split_files = {
            problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir),
            problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir),
        }

        _maybe_copy_corpus(tmp_dir)
        return split_files[dataset_split]


@registry.register_problem
class RecognizeProtoAlgonquian(recognition.RecognitionProblem):
    """
    Recognizes the characters of proto-algonquian
    """
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

    def source_data_files(self, data_dir, tmp_dir, dataset_split):
        """
        Only recognize the unsegmented data
        """
        return _train_data_filenames(tmp_dir, with_segmented=False)


@registry.register_problem
class MultitaskProtoAlgonquian(multi_problem.MultiProblem,
                               util.SingleProcessProblem):
    """
    Defines a multitask problem that trains a language model and a segmentation
    task at the same time
    """
    def __init__(self, was_reversed=False, was_copy=False):
        super(MultitaskProtoAlgonquian, self).__init__(was_reversed, was_copy)
        self.task_list.append(LanguageModelProtoAlgonquian())
        self.task_list.append(SegmentProtoAlgonquian())

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


@registry.register_problem
class MultitaskProtoAlgonquianWithNoise(MultitaskProtoAlgonquian):
    """
    Defines a multitask problem that trains a language model, a segmentation
    task, and a recognition task
    """
    def __init__(self, was_reversed=False, was_copy=False):
        super(MultitaskProtoAlgonquianWithNoise, self).__init__(was_reversed,
                                                                was_copy)
        self.task_list.append(RecognizeProtoAlgonquian())
