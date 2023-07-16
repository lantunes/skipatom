
class ElemNet:
    def __init__(self, input_dim, activation="relu", l2_lambda=0.00001):
        """
        ElemNet architecture, as in: Jha, Dipendra, et al. "Elemnet: Deep learning the chemistry of materials from
        only elemental composition." Scientific reports 8.1 (2018): 1-13.
        """
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.regularizers import l2

        self._model = Sequential()
        self._model.add(Dense(1024, activation=activation, input_dim=input_dim, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(1024, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(1024, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(1024, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(512, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(512, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(512, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(256, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(256, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(256, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(128, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(128, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(128, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(64, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(64, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(32, activation=activation, kernel_regularizer=l2(l2_lambda)))
        self._model.add(Dense(1, kernel_regularizer=l2(l2_lambda)))

    def train(self, train_x, train_y, test_x, test_y, batch_size=32, step_size=0.0001, num_epochs=10, callbacks=None):
        from keras.optimizers import Adam

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
