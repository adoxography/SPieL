"""
nn.recognition

Base module for tasks involving simple character recognition
"""
import random
import tensorflow as tf
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import problem

from . import util


def _extract_vocab_data(source_files):
    """
    Extracts the vocabulary from a list of files. Assumes character level
    tokens.

    :param source_files: The file names to extract from
    :type source_files: list of str
    :return: The unique tokens contained in the source files
    :rtype: list of str
    """
    vocab = set()

    for source_file in source_files:
        with tf.gfile.Open(source_file) as vocab_file:
            for line in vocab_file:
                tokens = line.split()
                vocab.update(tokens)

    return list(vocab)


class RecognitionProblem(text_problems.Text2TextProblem,
                         util.SingleProcessProblem):
    """
    Defines a text to text problem that trains a network to recognize random
    sequences of characters as those random sequences of characters (i.e. an
    identity.
    """
    @property
    def is_generate_per_split(self):
        """
        Eval data is generated separately from train data.
        """
        return True

    @property
    def num_training_examples(self):
        """
        Unused since is_generate_per_split is True
        """

    @property
    def vocab_type(self):
        """
        Recognition problems are done on the token level
        """
        return text_problems.VocabType.CHARACTER

    @property
    def min_size(self):
        """
        The minimum size of a randomly generated string
        """
        raise NotImplementedError()

    @property
    def max_size(self):
        """
        The maximum size of a randomly generated string
        """
        raise NotImplementedError()

    @property
    def num_train_instances(self):
        """
        The number of training instances to generate
        """
        raise NotImplementedError()

    @property
    def num_eval_instances(self):
        """
        The number of eval instances to generate - defaults to a quarter of the
        number of training instances
        """
        return self.num_train_instances // 4

    def source_data_files(self, data_dir, tmp_dir, dataset_split):
        """
        The data files to use as source material. The problem will only use
        these file to generate a vocabulary.

        :return: A list of file names
        :rtype: list of str
        """
        raise NotImplementedError()

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Generates samples for the problem
        """
        files = self.source_data_files(data_dir, tmp_dir, dataset_split)
        vocab = _extract_vocab_data(files)

        # Determine the number of instances to generate
        if dataset_split == problem.DatasetSplit.TRAIN:
            num_instances = self.num_train_instances
        else:
            num_instances = self.num_eval_instances

        for _ in range(num_instances):
            instance_size = random.randint(self.min_size, self.max_size)
            tokens = random.choices(vocab, k=instance_size)
            instance = ''.join(tokens)
            yield {'inputs': instance, 'targets': instance}
