from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN


def get_cooccurrence_pairs(struct):
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
