"""
nn.util

Utility class for working with t2t
"""
import random


def extract_data_from_samples(sample_iter):
    """
    Extracts the vocab, the number of samples, and the minimum and maximum
    lengths of the inputs in a sample iterator
    """
    vocab = set()
    min_size = 1e9
    max_size = 0
    num_samples = 0

    for sample in sample_iter:
        line = sample['inputs']
        vocab.update(line)
        min_size = min(min_size, len(line))
        max_size = max(max_size, len(line))
        num_samples += 1

    return num_samples, list(vocab), min_size, max_size


def prefix_iter_key(iterator, prefix, key):
    """
    Adds a prefix to all the items with a given key in an iterator
    """
    for item in iterator:
        item[key] = prefix + item[key]
        yield item


def random_test_cases(num_samples, vocab, min_size, max_size):
    """
    Returns a generator of randomly generated samples based on a vocabulary
    """
    for _ in range(num_samples):
        size = random.randint(min_size, max_size)
        line = ''.join(random.choices(vocab, k=size))
        yield {'inputs': line, 'targets': line}


def random_iter(*args):
    """
    Returns the elements from a list of iterators in a random order

    :param args: A variable number of tuples, where the first element is a
                 generator and the second element is the size of the generator
    """
    total_items = sum(arg[1] for arg in args)
    indices = list(range(total_items))
    random.shuffle(indices)
    for i in indices:
        for iterator, size in args:
            if i < size:
                try:
                    yield next(iterator)
                except StopIteration:
                    return
                continue
            i -= size
