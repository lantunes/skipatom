try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import gzip


class TrainingData:
    def __init__(self, data, atom_to_index, index_to_atom):
        self.data = data
        self.atom_to_index = atom_to_index
        self.index_to_atom = index_to_atom

    def to_one_hot(self):
        data = np.array(self.data)
        atom1_indices = data[:, 0]
        atom2_indices = data[:, 1]
        return self._one_hot(atom1_indices, len(self.atom_to_index)), self._one_hot(atom2_indices, len(self.atom_to_index))

    @staticmethod
    def _one_hot(x, k, dtype=np.float32):
        """Create a one-hot encoding of x of size k."""
        return np.array(x[:, None] == np.arange(k), dtype)

    @staticmethod
    def from_atom_pairs(atom_pairs_csv):

        # create the atom indices
        o = gzip.open(atom_pairs_csv, 'rt') if atom_pairs_csv.endswith(".gz") else open(atom_pairs_csv, 'rt')
        atoms = set()
        atom_to_index = {}
        index_to_atom = {}
        atom_index = 0
        pairs = []
        with o as f:
            for line in f.readlines():
                atom1, atom2 = line.strip().split(",")
                if atom1 not in atoms:
                    atoms.add(atom1)
                    atom_to_index[atom1] = atom_index
                    index_to_atom[atom_index] = atom1
                    atom_index += 1
                if atom2 not in atoms:
                    atoms.add(atom2)
                    atom_to_index[atom2] = atom_index
                    index_to_atom[atom_index] = atom2
                    atom_index += 1
                pairs.append((atom1, atom2))

        # create the training data, one data point per atom pair
        data = []
        for a1, a2 in pairs:
            data.append((atom_to_index[a1], atom_to_index[a2]))

        return TrainingData(data, atom_to_index, index_to_atom)

    @staticmethod
    def save(training_data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
