"""
spiel.command_line

Command line interface into SPieL

Usage:
spiel --train TRAIN_FILE [--test TEST_FILE]
"""
import sys
from argparse import ArgumentParser

from spiel.data import load_file as load_instances
from spiel.segmentation import ConstraintSegmenter, Featurizer
from spiel.sequence_labelling import SequenceLabeller


def parse_args():
    """
    Parses the arguments from the command line
    """
    parser = ArgumentParser()
    parser.add_argument('--train', dest='train_file', required=True)
    parser.add_argument('--test', dest='test_file')
    return parser.parse_args()


def init_segmenter(instances):
    """
    Initializes the segmenter

    :param instances: The instances to train the segmenter on
    :type instances: list of Instance
    :rtype: ConstraintSegmenter
    """
    featurizer = Featurizer(mode='basic')
    segmenter = ConstraintSegmenter(featurizer=featurizer)

    data = [(instance.shape, instance.annotations) for instance in instances]
    segmenter.train(*zip(*data))

    return segmenter


def init_labeller(instances):
    """
    Initializes the labeller

    :param instances: The instances to train the labeller on
    :type instances: list of Instance
    :rtype: SequenceLabeller
    """
    labeller = SequenceLabeller()
    data = [(instance.segments, instance.labels) for instance in instances]
    labeller.train(*zip(*data), grid_search=True)
    return labeller


def run_pipeline(segmenter, labeller, instances):
    """
    Runs the segmenter/labeller pipeline on a list of instances and prints the
    results to the console

    :param segmenter: An object to segment the instances into tokens
    :type segmenter: ConstraintSegmenter
    :param labeller: An object to label the tokens
    :type labeller: SequenceLabeller
    """
    num_tests = 0
    num_right = 0

    for instance in instances:
        segments = segmenter.segment(instance.shape)
        labels = labeller.label(segments)

        prediction = '-'.join([f"{segment}/{label}"
                               for segment, label in zip(segments, labels)])

        if instance.segments:
            num_tests += 1
            if instance.annotation_string() == prediction:
                num_right += 1
            else:
                print(f"'{instance.shape}': expected \
'{instance.annotation_string()}'; got '{prediction}'.",
                      file=sys.stderr)
        else:
            print(f"Shape '{instance.shape}' segmented to '{prediction}'.")

    if num_tests > 0:
        print(f"Accuracy: {num_right/num_tests}")


def main():
    """
    Entry point into the script
    """
    args = parse_args()

    train_instances = load_instances(args.train_file)
    segmenter = init_segmenter(train_instances)
    labeller = init_labeller(train_instances)

    print('Train results')
    run_pipeline(segmenter, labeller, train_instances)

    if args.test_file:
        test_instances = load_instances(args.test_file, strict=False)
        print('\nTest results')
        run_pipeline(segmenter, labeller, test_instances)
