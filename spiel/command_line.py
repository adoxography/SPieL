"""
spiel.command_line

Command line interface into SPieL

Usage:
spiel --train TRAIN_FILE [--test TEST_FILE]
"""
from optparse import OptionParser

from spiel.data import load_file as load_instances
from spiel.segmentation import ConstraintSegmenter, Featurizer
from spiel.sequence_labelling import SequenceLabeller


def parse_args():
    parser = OptionParser()
    parser.add_option('--train', dest='train_file')
    parser.add_option('--test', dest='test_file')
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


def run_pipeline(segmenter, labeller, instances):
    for instance in instances:
        segments = segmenter.segment(instance.shape)
        labels = labeller.label(segments)

        prediction = '-'.join([f"{segment}/{label}"
                               for segment, label in zip(segments, labels)])
        results = [f"Predicted: {prediction}"]

        if instance.segments:
            results.append(f"Actual: {instance.annotation_string()}")

        print(f"Shape: {instance.shape}")
        print('\t'.join(results))


def main():
    options, args = parse_args()

    if options.train_file is None:
        raise ValueError('The --train flag is required')

    train_instances = load_instances(options.train_file)
    segmenter = init_segmenter(train_instances)
    labeller = init_labeller(train_instances)

    print('Train results')
    run_pipeline(segmenter, labeller, train_instances)

    if options.test_file:
        test_instances = load_instances(options.test_file, strict=False)
        print('\nTest results')
        run_pipeline(segmenter, labeller, test_instances)
