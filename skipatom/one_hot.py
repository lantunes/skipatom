import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle


class OneHotVectors:
    def __init__(self, elems):
        """
        elems: a list of strings with the names of the elements to be assigned a random vector
        """
        self.elems = elems

        self.vectors = np.zeros(shape=(len(elems), len(elems)))

        self.dictionary = {}
        for i, elem in enumerate(elems):
            self.dictionary[elem] = i
            self.vectors[i][i] = 1.0

    def save(self, filename):
        with open(filename, 'wb') as f:
            data = (self.elems, self.vectors, self.dictionary)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            elems, vectors, dictionary = pickle.load(f)
            ohv = OneHotVectors(elems=elems)
            ohv.vectors = vectors
            ohv.dictionary = dictionary
            return ohv
