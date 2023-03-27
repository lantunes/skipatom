import sys

sys.path.extend([".", ".."])
import argparse
import gzip
import os
import shutil
from sys import argv
from time import time

import numpy as np
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
)

from skipatom import ElemNet, ElpasoliteNet

try:
    import cPickle as pickle
except ImportError:
    import pickle


ARCHITECTURES = ["elemnet", "elpasolite"]
ELEMNET_ARCH = ARCHITECTURES[0]
ELPASOLITE_ARCH = ARCHITECTURES[1]


class LogMetrics(Callback):
    def __init__(self, current_repeat, current_fold):
        super().__init__()
        self.current_repeat = current_repeat
        self.current_fold = current_fold

    def on_epoch_end(self, epoch, logs=None):
        logs["repeat"] = self.current_repeat
        logs["fold"] = self.current_fold


def load_all_data(filename, gzipped=True):
    o = gzip.open if gzipped else open
    with o(filename, "rb") as f:
        _, data = pickle.load(f)
        dataset = np.array(data)
        return dataset[:, 0].tolist(), dataset[:, 1].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a feedforward neural network model using repeated k-fold cross-validation."
    )
    parser.add_argument(
        "--dataset", nargs="?", required=True, type=str, help="path to dataset file"
    )
    parser.add_argument(
        "--architecture",
        required=True,
        choices=ARCHITECTURES,
        help="type of architecture to use",
    )
    parser.add_argument(
        "--results",
        nargs="?",
        required=True,
        type=str,
        help="path to the directory where the results .csv file will be written "
        "(this directory must already exist)",
    )
    parser.add_argument(
        "--models",
        nargs="?",
        required=True,
        type=str,
        help="path to the directory where models will be persisted (this directory must already exist)",
    )

    parser.add_argument(
        "--folds",
        required=False,
        type=int,
        default=5,
        help="the number of folds to use",
    )
    parser.add_argument(
        "--repeats", required=False, type=int, default=2, help="the number of repeats"
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=17022021,
        help="the random state to use for creating the k-fold splits",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=100,
        help="the maximum number of epochs",
    )
    parser.add_argument(
        "--batch", required=False, type=int, default=32, help="the batch size"
    )
    parser.add_argument(
        "--lr", required=False, type=float, default=0.0001, help="the learning rate"
    )
    parser.add_argument(
        "--activation",
        required=False,
        type=str,
        default="relu",
        help="the type of activation to use",
    )
    parser.add_argument(
        "--l2",
        required=False,
        type=float,
        default=0.00001,
        help="the L2 lambda value to use",
    )

    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        default=False,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "--patience",
        required=("--early-stopping" in argv),
        type=int,
        help="the patience to use if early stopping was specified",
    )

    args = parser.parse_args()

    architecture = None
    if args.architecture == ELEMNET_ARCH:
        architecture = ElemNet
    elif args.architecture == ELPASOLITE_ARCH:
        architecture = ElpasoliteNet
    else:
        raise Exception("unsupported architecture: %s" % args.architecture)

    experiment = "experiment_%s" % int(time())
    model_dir = os.path.join(args.models, experiment)
    if os.path.exists(model_dir):
        # delete the model dir, if it exists
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    print("saving models in %s" % model_dir)

    print("dataset: %s" % args.dataset)
    print("architecture: %s" % architecture)
    print("results: %s" % args.results)
    print("repeats: %s" % args.repeats)
    print("folds: %s" % args.folds)
    print("random state: %s" % args.seed)
    print("max epochs: %s" % args.epochs)
    print("batch: %s" % args.batch)
    print("learning rate: %s" % args.lr)
    print("activation: %s" % args.activation)
    print("L2 lambda: %s" % args.l2)

    if args.early_stopping:
        print("early stopping with patience %s" % args.patience)

    print("loading dataset...")
    X, y = load_all_data(args.dataset, gzipped=args.dataset.endswith(".gz"))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dim = X[0].shape[0]
    print("dim: %s" % dim)

    kfold = RepeatedKFold(
        n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed
    )

    repeat = 1
    fold = 1
    for train, test in kfold.split(X, y):
        print(f"REPEAT: {repeat}, FOLD {fold}")

        log_metrics = LogMetrics(current_repeat=repeat, current_fold=fold)
        csv_log_filename = os.path.join(args.results, "%s-results.csv" % experiment)
        csv_logger = CSVLogger(csv_log_filename, separator=",", append=True)

        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(model_dir, f"best_model_repeat_{repeat}_fold_{fold}"),
            save_best_only=True,
            monitor="val_mae",
            mode="min",
            verbose=1,
        )

        callbacks = [log_metrics, csv_logger, model_checkpoint]
        if args.early_stopping:
            callbacks.append(
                EarlyStopping(monitor="val_mae", mode="min", patience=args.patience)
            )

        model = architecture(
            input_dim=dim, activation=args.activation, l2_lambda=args.l2
        )

        model.train(
            X[train],
            y[train],
            X[test],
            y[test],
            num_epochs=args.epochs,
            batch_size=args.batch,
            step_size=args.lr,
            callbacks=callbacks,
        )

        if fold == args.folds:
            fold = 1
            repeat += 1
        else:
            fold += 1

    print("done.")
