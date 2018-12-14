"""
spiel.command_line

Command line interface into SPieL

Usage:
spiel [data_file]
"""
from optparse import OptionParser

from spiel.data import load_file as load_instances
from spiel.segmentation import ConstraintSegmenter, Featurizer
from spiel.sequence_labelling import SequenceLabeller


def parse_args():
    parser = OptionParser()
    parser.add_option('--train', dest='train_file')
    return parser.parse_args()


def init_segmenter(instances):
    featurizer = Featurizer(mode='basic')
    segmenter = ConstraintSegmenter(featurizer=featurizer)

    data = [(instance.shape, instance.annotations) for instance in instances]
    segmenter.train(*zip(*data))

    return segmenter


def init_labeller(instances):
    labeller = SequenceLabeller()
    data = [(instance.segments, instance.labels) for instance in instances]
    labeller.train(*zip(*data), grid_search=True)
    return labeller


def main():
    options, args = parse_args()
    train_file = options.train_file

    if train_file is None:
        raise ValueError('The --train flag is required')

    instances = load_instances(train_file)
    segmenter = init_segmenter(instances)
    labeller = init_labeller(instances)

    for instance in instances:
        segments = segmenter.segment(instance.shape)
        labels = labeller.label(segments)

        prediction = '-'.join([f"{segment}/{label}"
                               for segment, label in zip(segments, labels)])
        target = '-'.join([f"{segment}/{label}"
                           for segment, label
                           in zip(instance.segments, instance.labels)])

        print(f"Shape: {instance.shape}")
        print(f"Predicted: {prediction}\tActual: {target}")
