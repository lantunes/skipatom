import numpy as np


def get_cooccurrence_pairs(struct):
    """
    Given a pymatgen Structure, returns a list with the co-occurring atom pairs.

    :param struct: a pymatgen Structure object

    :return: a list of co-occurring atom pairs (i.e. a list of 2-tuples)
    """
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import CrystalNN

    pairs = []
    struct_graph = StructureGraph.with_local_env_strategy(struct, CrystalNN())
    labels = {i: spec.name for i, spec in enumerate(struct.species)}
    G = struct_graph.graph.to_undirected()
    for n in labels:
        target = labels[n]
        # TODO what if the atom doesn't have any neighbors?
        neighbors = [labels[i] for i in G.neighbors(n)]
        for neighbor in neighbors:
            pairs.append((target, neighbor))
    return pairs


def sum_pool(comp, dictionary, embeddings):
    """
    Returns a sum-pooled distributed representation of the given composition using the given embeddings.

    :param comp: a pymatgen Composition

    :param dictionary: a dictionary of atom name to embedding table row index

    :param embeddings: a list of the embeddings for each atom type

    :return: a sum-pooled vector representation of the given composition
    """
    vectors = []
    for e in comp.elements:
        amount = float(comp.to_reduced_dict[e.name])
        vectors.append(amount * np.array(embeddings[dictionary[e.name]]))
    return np.sum(vectors, axis=0).tolist()


def mean_pool(comp, dictionary, embeddings):
    """
    Returns a mean-pooled distributed representation of the given composition using the given embeddings.

    :param comp: a pymatgen Composition

    :param dictionary: a dictionary of atom name to embedding table row index

    :param embeddings: a list of the embeddings for each atom type

    :return: a mean-pooled vector representation of the given composition
    """
    vectors = []
    tot_amount = 0
    for e in comp.elements:
        amount = float(comp.to_reduced_dict[e.name])
        vectors.append(amount * np.array(embeddings[dictionary[e.name]]))
        tot_amount += amount
    return (np.sum(vectors, axis=0) / tot_amount).tolist()


def max_pool(comp, dictionary, embeddings):
    """
    Returns a max-pooled distributed representation of the given composition using the given embeddings.

    :param comp: a pymatgen Composition

    :param dictionary: a dictionary of atom name to embedding table row index

    :param embeddings: a list of the embeddings for each atom type

    :return: a max-pooled vector representation of the given composition
    """
    vectors = []
    for e in comp.elements:
        amount = float(comp.to_reduced_dict[e.name])
        vectors.append(amount * np.array(embeddings[dictionary[e.name]]))
    return np.max(vectors, axis=0).tolist()


ATOMS = {
    "H": [1, 1, 2.2],
    "He": [18, 1, np.nan],
    "Li": [1, 2, 0.98],
    "Be": [2, 2, 1.57],
    "B": [13, 2, 2.04],
    "C": [14, 2, 2.55],
    "N": [15, 2, 3.04],
    "O": [16, 2, 3.44],
    "F": [17, 2, 3.98],
    "Ne": [18, 2, np.nan],
    "Na": [1, 3, 0.93],
    "Mg": [2, 3, 1.31],
    "Al": [13, 3, 1.61],
    "Si": [14, 3, 1.9],
    "P": [15, 3, 2.19],
    "S": [16, 3, 2.58],
    "Cl": [17, 3, 3.16],
    "Ar": [18, 3, np.nan],
    "K": [1, 4, 0.82],
    "Ca": [2, 4, 1.0],
    "Sc": [3, 4, 1.36],
    "Ti": [4, 4, 1.54],
    "V": [5, 4, 1.63],
    "Cr": [6, 4, 1.66],
    "Mn": [7, 4, 1.55],
    "Fe": [8, 4, 1.83],
    "Co": [9, 4, 1.88],
    "Ni": [10, 4, 1.91],
    "Cu": [11, 4, 1.9],
    "Zn": [12, 4, 1.65],
    "Ga": [13, 4, 1.81],
    "Ge": [14, 4, 2.01],
    "As": [15, 4, 2.18],
    "Se": [16, 4, 2.55],
    "Br": [17, 4, 2.96],
    "Kr": [18, 4, 3.0],
    "Rb": [1, 5, 0.82],
    "Sr": [2, 5, 0.95],
    "Y": [3, 5, 1.22],
    "Zr": [4, 5, 1.33],
    "Nb": [5, 5, 1.6],
    "Mo": [6, 5, 2.16],
    "Tc": [7, 5, 1.9],
    "Ru": [8, 5, 2.2],
    "Rh": [9, 5, 2.28],
    "Pd": [10, 5, 2.2],
    "Ag": [11, 5, 1.93],
    "Cd": [12, 5, 1.69],
    "In": [13, 5, 1.78],
    "Sn": [14, 5, 1.96],
    "Sb": [15, 5, 2.05],
    "Te": [16, 5, 2.1],
    "I": [17, 5, 2.66],
    "Xe": [18, 5, 2.6],
    "Cs": [1, 6, 0.79],
    "Ba": [2, 6, 0.89],
    "La": [3, 8, 1.1],
    "Ce": [4, 8, 1.12],
    "Pr": [5, 8, 1.13],
    "Nd": [6, 8, 1.14],
    "Pm": [7, 8, 1.13],
    "Sm": [8, 8, 1.17],
    "Eu": [9, 8, 1.2],
    "Gd": [10, 8, 1.2],
    "Tb": [11, 8, 1.1],
    "Dy": [12, 8, 1.22],
    "Ho": [13, 8, 1.23],
    "Er": [14, 8, 1.24],
    "Tm": [15, 8, 1.25],
    "Yb": [16, 8, 1.1],
    "Lu": [17, 8, 1.27],
    "Hf": [4, 6, 1.3],
    "Ta": [5, 6, 1.5],
    "W": [6, 6, 2.36],
    "Re": [7, 6, 1.9],
    "Os": [8, 6, 2.2],
    "Ir": [9, 6, 2.2],
    "Pt": [10, 6, 2.28],
    "Au": [11, 6, 2.54],
    "Hg": [12, 6, 2.0],
    "Tl": [13, 6, 1.62],
    "Pb": [14, 6, 2.33],
    "Bi": [15, 6, 2.02],
    "Po": [16, 6, 2.0],
    "At": [17, 6, 2.2],
    "Rn": [18, 6, 2.2],
    "Fr": [1, 7, 0.7],
    "Ra": [2, 7, 0.9],
    "Ac": [3, 9, 1.1],
    "Th": [4, 9, 1.3],
    "Pa": [5, 9, 1.5],
    "U": [6, 9, 1.38],
    "Np": [7, 9, 1.36],
    "Pu": [8, 9, 1.28],
    "Am": [9, 9, 1.3],
    "Cm": [10, 9, 1.3],
    "Bk": [11, 9, 1.3],
    "Cf": [12, 9, 1.3],
    "Es": [13, 9, 1.3],
    "Fm": [14, 9, 1.3],
    "Md": [15, 9, 1.3],
    "No": [16, 9, 1.3],
    "Lr": [17, 9, 1.3],
    "Rf": [4, 7, np.nan],
    "Db": [5, 7, np.nan],
    "Sg": [6, 7, np.nan],
    "Bh": [7, 7, np.nan],
    "Hs": [8, 7, np.nan],
    "Mt": [9, 7, np.nan],
    "Ds": [10, 7, np.nan],
    "Rg": [11, 7, np.nan],
    "Cn": [12, 7, np.nan],
    "Nh": [13, 7, np.nan],
    "Fl": [14, 7, np.nan],
    "Mc": [15, 7, np.nan],
    "Lv": [16, 7, np.nan],
    "Ts": [17, 7, np.nan],
    "Og": [18, 7, np.nan]
}


class Atom:
    """
    Creates a representation of an atom, consisting of the atomic symbol, the atomic group number, the atomic
    row number, and the atom's Pauling electronegativity.
    """
    def __init__(self, symbol, group, row, X):
        """
        Create an instance of `Atom`.

        :param symbol: the atomic symbol
        :param group: the atomic group number
        :param row: the atomic row number
        :param X: the atom's Pauling electronegativity
        """
        self.symbol = symbol
        self.group = group
        self.row = row
        self.X = X


def get_atoms():
    """
    Returns a dictionary mapping atoms to their properties.

    :return: a dict, where the keys are atomic symbols, and the associated values are
             the corresponding `Atom` instances
    """
    return {atom: Atom(atom, props[0], props[1], props[2]) for atom, props in ATOMS.items()}
