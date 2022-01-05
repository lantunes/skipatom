try:
    import cPickle as pickle
except ImportError:
    import pickle


class Trainer:
    def __init__(self, embedding_dim, word_dim):
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        self._model = Sequential()
        self._model.add(Dense(embedding_dim, input_dim=word_dim, name="embeddings"))
        self._model.add(Dense(word_dim, activation="softmax"))

    def train(self, words, tags, step_size=0.001, num_epochs=10, batch_size=128, **kwargs):
        from tensorflow.keras.optimizers import Adam

        self._model.compile(loss="categorical_crossentropy",
                            optimizer=Adam(lr=step_size, epsilon=1e-8, **kwargs))

        print("# of examples: %s " % words.shape[0])
        print("Starting training...")

        self._model.fit(words, tags, batch_size=batch_size, epochs=num_epochs)

        return self._model.get_layer("embeddings").get_weights()[0]

    @staticmethod
    def save_embeddings(embeddings, filename):
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_embeddings(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
