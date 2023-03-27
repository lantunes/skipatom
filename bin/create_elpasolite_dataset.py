import sys

sys.path.extend([".", ".."])
import argparse
import gzip
from sys import argv

import numpy as np

from skipatom import (
    AtomVectors,
    OneHotVectors,
    RandomVectors,
    SkipAtomInducedModel,
    SkipAtomModel,
)

try:
    import cPickle as pickle
except ImportError:
    import pickle

REPRESENTATIONS = [
    "one-hot",
    "random",
    "atom2vec",
    "mat2vec",
    "skipatom",
    "skipatom-induced",
]
ONE_HOT = REPRESENTATIONS[0]
RANDOM = REPRESENTATIONS[1]
ATOM2VEC = REPRESENTATIONS[2]
MAT2VEC = REPRESENTATIONS[3]
SKIPATOM = REPRESENTATIONS[4]
SKIPATOM_INDUCED = REPRESENTATIONS[5]


def get_concatenated_vectors(formula, dictionary, embeddings):
    atoms = formula.split(" ")
    vectors = [[]] * 4
    for atom in set(atoms):
        count = atoms.count(atom)
        if count == 6:  # D
            vectors[3] = embeddings[dictionary[atom]]
        elif count == 2:  # C
            vectors[2] = embeddings[dictionary[atom]]
        else:
            if (
                vectors[0] == []
            ):  # A or B TODO not sure how to determine A vs B in ABC2D6, does it matter?
                vectors[0] = embeddings[dictionary[atom]]
            else:
                vectors[1] = embeddings[dictionary[atom]]
    return np.concatenate(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an Elpasolite dataset for training and evaluation on the Elapsolite formation energy task."
    )
    parser.add_argument(
        "--data",
        nargs="?",
        required=True,
        type=str,
        help="path to Elpasolite data file: this is a file containing two lists, the compositions and "
        "their corresponding formation energies",
    )

    parser.add_argument(
        "--out",
        nargs="?",
        required=True,
        type=str,
        help="path to the output file; a .pkl.gz extension should be used (the file will be gzipped)",
    )
    parser.add_argument(
        "--atoms",
        nargs="?",
        required=True,
        type=argparse.FileType("rt"),
        default=sys.stdin,
        help="path to atoms file: a file containing a list of the supported atoms, "
        "one atom per line; only compounds containing atoms in this list will be included "
        "in the dataset",
    )

    parser.add_argument(
        "--representation",
        required=True,
        choices=REPRESENTATIONS,
        help="type of representation to use",
    )

    parser.add_argument(
        "--vectors",
        required=((ATOM2VEC in argv) or (MAT2VEC in argv)),
        type=str,
        help="path to the file containing the vectors, if 'atom2vec' or 'mat2vec' "
        "were selected as the representation type; if 'random' was the selected type, "
        "then the given random vectors will be used instead of generating them",
    )

    parser.add_argument(
        "--skipatom-model",
        required=((SKIPATOM in argv) or (SKIPATOM_INDUCED in argv)),
        type=str,
        help="path the SkipAtom model file if 'skipatom' or 'skipatom-induced' were selected as "
        "the representation type",
    )
    parser.add_argument(
        "--skipatom-td",
        required=((SKIPATOM in argv) or (SKIPATOM_INDUCED in argv)),
        type=str,
        help="path the SkipAtom training data file if 'skipatom' or 'skipatom-induced' were "
        "selected as the representation type",
    )

    parser.add_argument(
        "--skipatom-min-count",
        required=False,
        default=2e7,
        type=lambda x: int(float(x)),
        help="min count to use if 'skipatom-induced' was selected as the representation type",
    )
    parser.add_argument(
        "--skipatom-top-n",
        required=False,
        default=5,
        type=int,
        help="top N to use if 'skipatom-induced' was selected as the representation type",
    )

    parser.add_argument(
        "--random-dim",
        required=False,
        type=int,
        default=200,
        help="the number of dimensions to assign to random vectors, if 'random' was selected as the "
        "representation type, and the --vectors argument was not provided",
    )
    parser.add_argument(
        "--random-mean",
        required=False,
        type=float,
        default=0.0,
        help="the mean to use for random vectors, if 'random' was selected as the "
        "representation type, and the --vectors argument was not provided",
    )
    parser.add_argument(
        "--random-std",
        required=False,
        type=float,
        default=1.0,
        help="the std. dev. to use for random vectors, if 'random' was selected as the "
        "representation type, and the --vectors argument was not provided",
    )

    args = parser.parse_args()

    print("loading atoms from %s ..." % args.atoms.name)
    atoms = [line.strip() for line in args.atoms.readlines()]

    print("loading Elpasolite data from %s ..." % args.data)
    with open(args.data, "rb") as pickle_file:
        X, y = pickle.load(pickle_file)

    representation = None
    if args.representation == ONE_HOT:
        print("creating one-hot vectors...")
        representation = OneHotVectors(elems=atoms)

    elif args.representation == RANDOM:
        if args.vectors:
            print("loading random vectors from %s ..." % args.vectors)
            representation = RandomVectors.load(args.vectors)
        else:
            mean = args.random_mean
            std = args.random_std
            dim = args.random_dim
            print(
                "creating random vectors with dim=%s, mean=%s, std=%s ..."
                % (dim, mean, std)
            )
            representation = RandomVectors(elems=atoms, dim=dim, mean=mean, std=std)

    elif args.representation == ATOM2VEC:
        print("loading Atom2Vec vectors from %s ..." % args.vectors)
        representation = AtomVectors.load(args.vectors)

    elif args.representation == MAT2VEC:
        print("loading Mat2Vec vectors from %s ..." % args.vectors)
        representation = AtomVectors.load(args.vectors)

    elif args.representation == SKIPATOM:
        print(
            "loading SkipAtom vectors from %s and %s ..."
            % (args.skipatom_model, args.skipatom_td)
        )
        representation = SkipAtomModel.load(args.skipatom_model, args.skipatom_td)

    elif args.representation == SKIPATOM_INDUCED:
        print(
            "loading SkipAtom (induced, min count: %s, top n: %s) vectors from %s and %s ..."
            % (
                args.skipatom_min_count,
                args.skipatom_top_n,
                args.skipatom_model,
                args.skipatom_td,
            )
        )
        representation = SkipAtomInducedModel.load(
            args.skipatom_model,
            args.skipatom_td,
            args.skipatom_min_count,
            args.skipatom_top_n,
        )
    else:
        raise Exception("unsupported representation type: %s" % args.representation)

    print("creating dataset...")
    dataset = []
    formulas = []
    for i, x in enumerate(X):
        dataset.append(
            [
                get_concatenated_vectors(
                    x, representation.dictionary, representation.vectors
                ),
                y[i],
            ]
        )
        formulas.append(x)

    print("dataset num rows: %s" % len(dataset))

    print("writing dataset to %s ..." % args.out)
    with gzip.open(args.out, "wb") as f:
        pickle.dump((formulas, dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("done.")
