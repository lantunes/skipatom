import numpy as np
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN


def get_cooccurrence_pairs(struct):
    """
    Given a pymatgen Structure, returns a list with the co-occurring atom pairs.

    :param struct: a pymatgen Structure object

    :return: a list of co-occurring atom pairs (i.e. a list of 2-tuples)
    """
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
