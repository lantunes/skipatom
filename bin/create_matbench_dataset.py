import sys
sys.path.extend([".", ".."])
from sys import argv
import argparse
import gzip
import json
from skipatom import OneHotVectors, RandomVectors, AtomVectors, SkipAtomModel, SkipAtomInducedModel, \
    sum_pool, mean_pool, max_pool
from pymatgen.core import Composition, Structure

try:
    import cPickle as pickle
except ImportError:
    import pickle

REPRESENTATIONS = ["one-hot", "random", "atom2vec", "mat2vec", "skipatom", "skipatom-induced"]
ONE_HOT = REPRESENTATIONS[0]
RANDOM = REPRESENTATIONS[1]
ATOM2VEC = REPRESENTATIONS[2]
MAT2VEC = REPRESENTATIONS[3]
SKIPATOM = REPRESENTATIONS[4]
SKIPATOM_INDUCED = REPRESENTATIONS[5]

POOLINGS = ["sum", "mean", "max"]
SUM_POOLING = POOLINGS[0]
MEAN_POOLING = POOLINGS[1]
MAX_POOLING = POOLINGS[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a MatBench dataset for training and evaluation with ElemNet-like models."
    )
    parser.add_argument("--data", nargs="?", required=True, type=str,
                        help="path to MatBench data file")

    parser.add_argument("--out", nargs="?", required=True, type=str,
                        help="path to the output file; a .pkl.gz extension should be used (the file will be gzipped)")
    parser.add_argument("--atoms", nargs="?", required=True, type=argparse.FileType("rt"), default=sys.stdin,
                        help="path to atoms file: a file containing a list of the supported atoms, "
                             "one atom per line; only compounds containing atoms in this list will be included "
                             "in the dataset")

    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS,
                        help="type of representation to use")

    parser.add_argument("--vectors", required=((ATOM2VEC in argv) or (MAT2VEC in argv)), type=str,
                        help="path to the file containing the vectors, if 'atom2vec' or 'mat2vec' "
                             "were selected as the representation type; if 'random' was the selected type, "
                             "then the given random vectors will be used instead of generating them")

    parser.add_argument("--skipatom-model", required=((SKIPATOM in argv) or (SKIPATOM_INDUCED in argv)), type=str,
                        help="path the SkipAtom model file if 'skipatom' or 'skipatom-induced' were selected as "
                             "the representation type")
    parser.add_argument("--skipatom-td", required=((SKIPATOM in argv) or (SKIPATOM_INDUCED in argv)), type=str,
                        help="path the SkipAtom training data file if 'skipatom' or 'skipatom-induced' were "
                             "selected as the representation type")

    parser.add_argument("--skipatom-min-count", required=False, default=2e7, type=lambda x: int(float(x)),
                        help="min count to use if 'skipatom-induced' was selected as the representation type")
    parser.add_argument("--skipatom-top-n", required=False, default=5, type=int,
                        help="top N to use if 'skipatom-induced' was selected as the representation type")

    parser.add_argument("--random-dim", required=False, type=int, default=200,
                        help="the number of dimensions to assign to random vectors, if 'random' was selected as the "
                             "representation type, and the --vectors argument was not provided")
    parser.add_argument("--random-mean", required=False, type=float, default=0.0,
                        help="the mean to use for random vectors, if 'random' was selected as the "
                             "representation type, and the --vectors argument was not provided")
    parser.add_argument("--random-std", required=False, type=float, default=1.0,
                        help="the std. dev. to use for random vectors, if 'random' was selected as the "
                             "representation type, and the --vectors argument was not provided")

    parser.add_argument("--pooling", required=True, choices=POOLINGS,
                        help="the type of pooling operation to use")

    args = parser.parse_args()

    print("loading atoms from %s ..." % args.atoms.name)
    atoms = [line.strip() for line in args.atoms.readlines()]

    # atoms that are unsupported by SkipAtom
    unsupported_atoms = ["He", "Ar", "Ne"]

    print("loading MatBench data from %s ..." % args.data)
    with gzip.open(args.data, "rb") as f:
        matbench_data = json.loads(f.read())

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
            print("creating random vectors with dim=%s, mean=%s, std=%s ..." % (dim, mean, std))
            representation = RandomVectors(elems=atoms, dim=dim, mean=mean, std=std)

    elif args.representation == ATOM2VEC:
        print("loading Atom2Vec vectors from %s ..." % args.vectors)
        representation = AtomVectors.load(args.vectors)

    elif args.representation == MAT2VEC:
        print("loading Mat2Vec vectors from %s ..." % args.vectors)
        representation = AtomVectors.load(args.vectors)

    elif args.representation == SKIPATOM:
        print("loading SkipAtom vectors from %s and %s ..." % (args.skipatom_model, args.skipatom_td))
        representation = SkipAtomModel.load(args.skipatom_model, args.skipatom_td)

    elif args.representation == SKIPATOM_INDUCED:
        print("loading SkipAtom (induced, min count: %s, top n: %s) vectors from %s and %s ..." %
              (args.skipatom_min_count, args.skipatom_top_n, args.skipatom_model, args.skipatom_td))
        representation = SkipAtomInducedModel.load(args.skipatom_model, args.skipatom_td,
                                                   args.skipatom_min_count, args.skipatom_top_n)
    else:
        raise Exception("unsupported representation type: %s" % args.representation)

    pool = None
    if args.pooling == SUM_POOLING:
        pool = sum_pool
    elif args.pooling == MEAN_POOLING:
        pool = mean_pool
    elif args.pooling == MAX_POOLING:
        pool = max_pool
    else:
        raise Exception("unsupported pooling: %s" % args.pooling)

    print("creating %s-pooled dataset..." % args.pooling)
    dataset = []
    formulas = []
    for pair in matbench_data["data"]:

        if type(pair[0]) is dict:
            composition = Structure.from_dict(pair[0]).composition
            formula = composition.reduced_formula
        else:
            formula = pair[0]
            composition = Composition(formula)

        val = pair[1]
        if type(val) is bool:
            val = 1. if val else 0.

        if any([e.name not in atoms for e in composition.elements]):
            print("WARNING: skipping %s, as it contains unsupported atoms" % formula)
            continue

        if (args.representation == SKIPATOM or args.representation == SKIPATOM_INDUCED) \
                and any([e.name in unsupported_atoms for e in composition.elements]):
            print("WARNING: skipping %s, as it contains atoms that SkipAtom does not support" % formula)
            continue

        dataset.append([
            pool(composition, representation.dictionary, representation.vectors), val
        ])
        formulas.append(formula)
        # NOTE: if the format is 'structure', there may be duplicate formulas in the resulting dataset

    print("dataset num rows: %s" % len(dataset))

    print("writing dataset to %s ..." % args.out)
    with gzip.open(args.out, 'wb') as f:
        pickle.dump((formulas, dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print("done.")
