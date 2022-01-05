import sys
sys.path.append('../skipatom')

from skipatom import TrainingData
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str,
                        help='The path to the pairs .csv file.')
    parser.add_argument('--out', '-o', type=str,
                        help='The path to the training data file to be created.')
    args = parser.parse_args()

    training_data = TrainingData.from_atom_pairs(args.data)

    print("number of atoms: %s" % len(training_data.atom_to_index))
    print("number of examples: %s" % len(training_data.data))

    TrainingData.save(training_data, args.out)


if __name__ == '__main__':
    main()
