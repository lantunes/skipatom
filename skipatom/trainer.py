import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Identity, LogSoftmax

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Trainer:
    def __init__(self, dim_in, dim_out):
        self.init_random_params, self.predict = stax.serial(
            Dense(dim_in), Identity,
            Dense(dim_out), LogSoftmax
        )
        self.rng = random.PRNGKey(0)

    def loss(self, params, batch):
        inputs, targets = batch
        preds = self.predict(params, inputs)
        return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def train(self, words, tags, step_size=0.001, num_epochs=10, batch_size=128, **kwargs):
        num_train = words.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        def data_stream():
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    yield words[batch_idx], tags[batch_idx]

        batches = data_stream()

        opt_init, opt_update, get_params = optimizers.adam(step_size, **kwargs)

        @jit
        def update(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, grad(self.loss)(params, batch), opt_state)

        _, init_params = self.init_random_params(self.rng, (-1, len(words[0])))
        opt_state = opt_init(init_params)
        itercount = itertools.count()

        print("# of examples: %s " % num_train)
        print("Starting training...")
        for epoch in range(num_epochs):
            start_time = time.time()
            tot_loss = 0.0
            for _ in range(num_batches):
                batch = next(batches)
                opt_state = update(next(itercount), opt_state, batch)
                tot_loss += self.loss(get_params(opt_state), batch)
            epoch_time = time.time() - start_time

            print("Epoch {} in {:0.2f} sec; loss: {:0.5f}".format(epoch, epoch_time, tot_loss))

        return get_params(opt_state)[0][0]  # TODO is this the proper way to get the learned embedding layer?

    @staticmethod
    def save_embeddings(embeddings, filename):
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_embeddings(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
