
class ElpasoliteNet:
    def __init__(self, input_dim, activation="relu", l2_lambda=0.00001):
        """
        As described in: Zhou, Quan, et al. "Learning atoms for materials discovery."
        Proceedings of the National Academy of Sciences 115.28 (2018): E6411-E6417.
        """
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2

        self._model = Sequential()
        self._model.add(Dense(10, activation=activation, input_dim=input_dim, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(1, kernel_regularizer=l2(l2_lambda)))

    def train(self, train_x, train_y, test_x, test_y, batch_size=32, step_size=0.0001, num_epochs=10, callbacks=None):
        from tensorflow.keras.optimizers import Adam

        self._model.compile(loss="mean_absolute_error", optimizer=Adam(lr=step_size, epsilon=1e-8), metrics=["mae"])

        validation_data = (test_x, test_y)
        if test_x is None and test_y is None:
            validation_data = None
        self._model.fit(train_x, train_y, batch_size=batch_size, epochs=num_epochs,
                        validation_data=validation_data, callbacks=callbacks)

    def evaluate(self, X, y, batch_size):
        return self._model.evaluate(X, y, batch_size=batch_size)

    def predict(self, X):
        return self._model.predict(X)

    def summary(self):
        return self._model.summary()

    @staticmethod
    def load(model_path):
        """
        NOTE: this returns an instance of the Keras model, not this class
        """
        from tensorflow import keras
        return keras.models.load_model(model_path)
