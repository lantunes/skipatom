import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


class RandomVectors:
    def __init__(self, elems, dim, mean=0.0, std=1.0):
        """
        elems: a list of strings with the names of the elements to be assigned a random vector
        dim: the size of each vector to be generated
        mean: float or array_like of floats
              Mean ("centre") of the distribution.
        std: float or array_like of floats
             Standard deviation (spread or "width") of the distribution. Must be
             non-negative.
        """
        self.elems = elems
        self.dim = dim
        self.mean = mean
        self.std = std

        self.vectors = np.random.normal(mean, std, size=(len(elems), dim))

        self.dictionary = {}
        for i, elem in enumerate(elems):
            self.dictionary[elem] = i

    def save(self, filename):
        with open(filename, 'wb') as f:
            data = (self.elems, self.dim, self.mean, self.std, self.vectors, self.dictionary)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            elems, dim, mean, std, vectors, dictionary = pickle.load(f)
            rv = RandomVectors(elems=elems, dim=dim, mean=mean, std=std)
            rv.vectors = vectors
            rv.dictionary = dictionary
            return rv
