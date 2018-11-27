"""
spiel.sequence_labelling.features

Handles featurization for the sequence labelling stage
"""


class Featurizer:
    """
    Converts sequences to features for use in sequence labelling
    """
    def __init__(self, ngrams=3):
        self.ngrams = ngrams

    def convert(self, sequence):
        """
        Converts a single sequence to features

        :param sequence: The sequence to convert
        :type sequence: list
        :return: The features for each segment in the sequence
        :rtype: list of dict
        """
        return [self.__convert_segment(sequence, index)
                for index, _ in enumerate(sequence)]

    def convert_many(self, sequences):
        """
        Converts multiple sequences to features

        :param sequence: The sequences to convert
        :type sequence: list of list
        :return: A list of lists of features for each sequence
        :rtype: list of list of dict
        """
        return [self.convert(sequence) for sequence in sequences]

    def __convert_segment(self, sequence, index):
        segment = sequence[index]

        features = {
            'bias': 1.0,
            'shape': segment,
        }

        for i in range(1, self.ngrams+1):
            features[f"prefix{i}"] = segment[:i]
            features[f"suffix{i}"] = segment[-i:]

        if index > 0:
            features['prev_shape'] = sequence[index-1]
        else:
            features['BOS'] = True

        if index < len(sequence) - 1:
            features['next_shape'] = sequence[index+1]
        else:
            features['EOS'] = True

        return features
