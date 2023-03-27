import sys

sys.path.extend([".", ".."])
import argparse
from sys import argv

from pymatgen import Element

from skipatom import SkipAtomInducedModel, SkipAtomModel

"""
e.g. 
--model ../data/mp_2020_10_09.dim200.keras.model 
--data ../data/mp_2020_10_09.training.data 
--induced --min-count 2e7 --top-n 5
--out ../data/skipatom_20201009_induced.csv
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a CSV file with the SkipAtom vectors."
    )
    parser.add_argument(
        "--model",
        nargs="?",
        required=True,
        type=str,
        help="path to SkipAtom .model file",
    )
    parser.add_argument(
        "--data",
        nargs="?",
        required=True,
        type=str,
        help="path to SkipAtom .training.data file",
    )
    parser.add_argument(
        "--out",
        nargs="?",
        required=True,
        type=str,
        help="path to the output file; a .csv extension should be used",
    )
    parser.add_argument(
        "--induced", action="store_true", help="whether to use induced SkipAtom vectors"
    )
    parser.add_argument(
        "--min-count",
        required=("induced" in argv),
        type=lambda x: int(float(x)),
        help="the min. count to use if induced vectors are specified",
    )
    parser.add_argument(
        "--top-n",
        required=("induced" in argv),
        type=int,
        help="the top N to use if induced vectors are specified",
    )

    args = parser.parse_args()

    if args.induced:
        model = SkipAtomInducedModel.load(
            args.model, args.data, min_count=args.min_count, top_n=args.top_n
        )
    else:
        model = SkipAtomModel.load(args.model, args.data)

    sorted_elems = sorted(
        [(e, Element(e).number) for e in model.dictionary], key=lambda v: v[1]
    )

    dim = len(model.vectors[0])

    with open(args.out, "w") as f:
        header = ["element"]
        header.extend([str(i) for i in range(dim)])
        f.write("%s\n" % ",".join(header))
        for elem, _ in sorted_elems:
            vec = model.vectors[model.dictionary[elem]].tolist()
            row = [elem]
            row.extend([str(v) for v in vec])
            f.write("%s\n" % ",".join(row))
