"""
spiel.sequence_labelling

Module for labelling morphemes
"""
import pickle

from scipy.stats import expon
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score

from spiel.util import flatten


DEFAULT_ALGORITHM = 'lbfgs'
DEFAULT_C1 = 0.1
DEFAULT_C2 = 0.1
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_ALL_POSSIBLE_TRANSITIONS = True


class SequenceLabeller:
    """
    Labels morphemes using an underlying CRF classifier
    """
    def __init__(self, model):
        """
        Initializes the classifier

        SequenceLabeller.build() or SequenceLabeller.grid_search() should
        normally be used instead.

        :param model: The underlying CRF model
        :type model: CRF
        """
        self.model = model

    @staticmethod
    def build(xs, ys, **kwargs):
        """
        Builds a sequence classifier from x/y pairs

        :param xs: A list of sequences, with each member of the sequence
                   represented as features
        :type xs: list of list of dict
        :param ys: The corresponding labels for each sequence
        :type ys: list of list of str
        :param kwargs: arguments to override the defaults given to the
                       underlying CRF
        :return: A trained sequence classifier based on the provided training
                 data
        :rtype: SequenceLabeller
        """
        params = {
            'algorithm': DEFAULT_ALGORITHM,
            'c1': DEFAULT_C1,
            'c2': DEFAULT_C2,
            'max_iterations': DEFAULT_MAX_ITERATIONS,
            'all_possible_transitions': DEFAULT_ALL_POSSIBLE_TRANSITIONS
        }

        if kwargs:
            params.update(kwargs)

        model = CRF(**params)
        model.fit(xs, ys)
        return SequenceLabeller(model)

    @staticmethod
    def grid_search(xs, ys):
        """
        Conducts a grid search to find an optimal CRF model, given a set of x/y
        pairs

        :param xs: A list of sequences, with each member of the sequence
                   represented as features
        :type xs: list of list of dict
        :param ys: The corresponding labels for each sequence
        :type ys: list of list of str
        :param kwargs: arguments to override the defaults given to the
                       underlying CRF
        :return: A trained sequence classifier based on the provided training
                 data
        :rtype: SequenceLabeller
        """
        rs = _grid_search(xs, ys)
        return SequenceLabeller(rs.best_estimator_)

    def save(self, path):
        """
        Saves the model to the specified path
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path):
        """
        Loads a saved model from a specified path
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return SequenceLabeller(model)

    def predict(self, x):
        """
        Predicts the label sequence of a single sequence

        :param x: The sequence to label
        :type x: list of dict
        :return: The label sequence predicted for the given sequence
        :rtype: list of str
        """
        return self.model.predict_single(x)

    def predict_many(self, xs):
        """
        Predicts the label sequence of a list of sequences

        :param xs: The sequences to label
        :type xs: list of list of dict
        :return: The label sequence predicted for each sequence
        :rtype: list of list of str
        """
        return self.model.predict(xs)

    def evaluate(self, xs, ys):
        """
        Returns the F1 score give a set of sequences and expected labels

        :param xs: A list of sequences, with each member of the sequence
                   represented as features
        :type xs: list of list of dict
        :param ys: The corresponding labels for each sequence
        :type ys: list of list of str
        :return: The F1 score from classifying the sequences
        :rtype: float
        """
        pred = self.predict_many(xs)
        return flat_f1_score(ys, pred, average='weighted')


def _grid_search(x, y):
    labels = list(set(flatten(y)))
    model = CRF(algorithm=DEFAULT_ALGORITHM,
                max_iterations=DEFAULT_MAX_ITERATIONS,
                all_possible_transitions=DEFAULT_ALL_POSSIBLE_TRANSITIONS)
    params_space = {'c1': expon(scale=0.5), 'c2': expon(scale=0.05)}
    f1_scorer = make_scorer(flat_f1_score, average='weighted',
                            labels=labels)
    rs = RandomizedSearchCV(model, params_space, cv=3, n_jobs=-1, n_iter=50,
                            scoring=f1_scorer)
    rs.fit(x, y)
    return rs
