"""
nn.segmentation

Base module for tasks involving morpheme segmentation
"""
import tensorflow as tf
from tensor2tensor.data_generators import text_problems


class SegmentationProblem(text_problems.Text2TextProblem):
    """
    Defines a segmentation problem, which is essentially a translation problem
    from unsegmented to segmented tokens.
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
        Segmentation problems are done on the token level
        """
        return text_problems.VocabType.CHARACTER

    def source_data_files(self, data_dir, tmp_dir, dataset_split):
        """
        The data files to use as source material.

        :return: A list of train/eval file pairs
        :rtype: list of (str, str)
        """
        raise NotImplementedError()

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """
        Generates the samples for the problem
        """
        files = self.source_data_files(data_dir, tmp_dir, dataset_split)
        for src_filepath, trg_filepath in files:
            for src, trg in zip(tf.gfile.Open(src_filepath),
                                tf.gfile.Open(trg_filepath)):
                yield {'inputs': src, 'targets': trg}
