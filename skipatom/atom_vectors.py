try:
    import cPickle as pickle
except ImportError:
    import pickle


class AtomVectors:
    """
    A generic container for atom vectors that have been previously generated, and pickled to a file
    containing a list of the supported elements, a list of the atom vectors, and a dict specifying
    a mapping from an atom symbol to the corresponding atom vector's index in the atom vector list.
    """
    def __init__(self):
        self.vectors = []
        self.elems = []
        self.dictionary = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            data = (self.elems, self.vectors, self.dictionary)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            elems, vectors, dictionary = pickle.load(f)
            av = AtomVectors()
            av.elems = elems
            av.vectors = vectors
            av.dictionary = dictionary
            return av
