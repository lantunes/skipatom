import sys
sys.path.extend([".", ".."])
import os
import shutil
import argparse
from sys import argv
import gzip
from skipatom import ElemNetLike, ElemNetLikeClassifier
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping
from time import time
import csv

try:
    import cPickle as pickle
except ImportError:
    import pickle


class LogMetrics(Callback):
    def __init__(self, current_fold):
        super().__init__()
        self.current_fold = current_fold

    def on_epoch_end(self, epoch, logs=None):
        logs["fold"] = self.current_fold


def load_all_data(filename, gzipped=True):
    o = gzip.open if gzipped else open
    with o(filename, 'rb') as f:
        _, data = pickle.load(f)
        dataset = np.array(data)
        return dataset[:, 0].tolist(), dataset[:, 1].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and evaluate an ElemNet-like neural network model using the Matbench v0.1 protocol "
                    "(https://hackingmaterials.lbl.gov/automatminer/datasets.html#benchmarking-and-reporting-your-algorithm)."
    )
    parser.add_argument("--dataset", nargs="?", required=True, type=str,
                        help="path to dataset file")
    parser.add_argument("--results", nargs="?", required=True, type=str,
                        help="path to the directory where the results .csv file will be written "
                             "(this directory must already exist)")
    parser.add_argument("--models", nargs="?", required=True, type=str,
                        help="path to the directory where models will be persisted "
                             "(NOTE: this directory will be deleted if it exists)")

    parser.add_argument("--folds", required=False, type=int, default=5,
                        help="the number of folds to use (NOTE: the protocol requires 5)")
    parser.add_argument("--seed", required=False, type=int, default=18012019,
                        help="the random state to use for creating the k-fold splits "
                             "(NOTE: the protocol requires 18012019)")
    parser.add_argument("--val_size", required=False, type=float, default=0.10,
                        help="a number between 0 and 1 representing the fraction of the training set to "
                             "withold as a validation set during training")
    parser.add_argument("--epochs", required=False, type=int, default=100,
                        help="the maximum number of epochs")
    parser.add_argument("--batch", required=False, type=int, default=32,
                        help="the batch size")
    parser.add_argument("--lr", required=False, type=float, default=0.0001,
                        help="the learning rate")
    parser.add_argument("--activation", required=False, type=str, default="relu",
                        help="the type of activation to use")
    parser.add_argument("--l2", required=False, type=float, default=0.00001,
                        help="the L2 lambda value to use")

    parser.add_argument("--classification", action="store_true", default=False,
                        help="whether this is a classification task")

    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true",
                        default=False,
                        help="whether to use early stopping")
    parser.add_argument("--patience", required=("--early-stopping" in argv), type=int,
                        help="the patience to use if early stopping was specified")

    args = parser.parse_args()

    architecture = None
    if args.classification:
        architecture = ElemNetLikeClassifier
    else:
        architecture = ElemNetLike

    experiment = "experiment_%s" % int(time())
    model_dir = os.path.join(args.models, experiment)
    if os.path.exists(model_dir):
        # delete the model dir, if it exists
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    print("saving models in %s" % model_dir)

    if not os.path.exists(args.results):
        os.makedirs(args.results)
    print("saving results in: %s" % args.results)

    print("dataset: %s" % args.dataset)
    print("architecture: %s" % architecture)
    print("folds: %s" % args.folds)
    print("seed: %s" % args.seed)
    print("validation size: %s" % args.val_size)
    print("max epochs: %s" % args.epochs)
    print("batch: %s" % args.batch)
    print("learning rate: %s" % args.lr)
    print("activation: %s" % args.activation)
    print("L2 lambda: %s" % args.l2)
    print("classification: %s" % args.classification)

    if args.folds != 5:
        print("WARNING: the Matbench v0.1 protocol requires 5 folds")

    if args.seed != 18012019:
        print("WARNING: the Matbench v0.1 protocol requires a seed of 18012019")

    if args.early_stopping:
        print("early stopping with patience %s" % args.patience)

    print("loading dataset...")
    X, y = load_all_data(args.dataset, gzipped=args.dataset.endswith(".gz"))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dim = X[0].shape[0]
    print("dim: %s" % dim)

    if args.classification:
        kfold = StratifiedKFold(n_splits=args.folds, random_state=args.seed, shuffle=True)
    else:
        kfold = KFold(n_splits=args.folds, random_state=args.seed, shuffle=True)

    test_results = []

    fold = 0
    for train_val, test in kfold.split(X, y):

        fold += 1
        print("FOLD %s" % fold)

        log_metrics = LogMetrics(current_fold=fold)
        csv_log_filename = os.path.join(args.results, "%s-fold-results.csv" % experiment)
        csv_logger = CSVLogger(csv_log_filename, separator=",", append=True)

        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model_fold_%s" % fold),
            save_best_only=True,
            monitor="val_auc" if args.classification else "val_mae",
            mode="max" if args.classification else "min",
            verbose=1,
        )

        callbacks = [log_metrics, csv_logger, model_checkpoint]
        if args.early_stopping:
            callbacks.append(EarlyStopping(
                monitor="val_auc" if args.classification else "val_mae",
                mode="max" if args.classification else "min",
                patience=args.patience
            ))

        model = architecture(input_dim=dim, activation=args.activation, l2_lambda=args.l2)

        train, val = train_test_split(train_val, test_size=args.val_size, random_state=args.seed, shuffle=True)

        model.train(X[train], y[train], X[val], y[val],
                    num_epochs=args.epochs, batch_size=args.batch, step_size=args.lr, callbacks=callbacks)

        checkpoints = [os.path.join(model_dir, name) for name in os.listdir(model_dir)]
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("restoring from: %s" % latest_checkpoint)
        best_model = model.load(latest_checkpoint)

        print("evaluating on test set...")
        results = best_model.evaluate(X[test], y[test], batch_size=32)
        evaluation = dict(zip(best_model.metrics_names, results))
        test_results.append(evaluation)
        print(evaluation)

    with open(os.path.join(args.results, "%s-test-results.csv" % experiment), "wt") as f:
        fieldnames = test_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_results:
            writer.writerow(row)

    print("done.")
