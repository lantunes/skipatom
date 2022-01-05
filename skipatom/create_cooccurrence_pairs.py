import sys
sys.path.append('../skipatom')

import argparse
import warnings
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import gzip
import logging

from skipatom import get_cooccurrence_pairs

warnings.simplefilter("ignore", category=UserWarning)


def listener(queue, filename, zip, n):
    pbar = tqdm(total=n)
    o = gzip.open(filename, "wt") if zip else open(filename, "w")
    with o as f:
        while True:
            message = queue.get()
            if message == "kill":
                break

            for pair in message:
                f.write("%s,%s\n" % pair)

            f.flush()

            pbar.update(1)


def worker(structs, queue, verbose):
    for i in range(len(structs)):
        try:
            queue.put(get_cooccurrence_pairs(structs[i]))
        except Exception as e:
            if verbose:
                logging.warning(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path to the dataset pickle file.')
    parser.add_argument('--out', '-o', type=str,
                        help='The path to the pairs file to be created.')
    parser.add_argument('--zip', '-z', action='store_true',
                        help='If present, indicates that the generated pairs file should be gzipped.')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(),
                        help='The number of processes to create. Default is the CPU count.')
    parser.add_argument('--workers', type=int, default=1,
                        help='The number of worker processes to use. Default is 1.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='If present, warnings will be logged.')
    parser.add_argument('--head', '-d', type=int, action='store',
                        help='If present, the first N records of the dataset will be used only')
    parser.add_argument('--tail', '-t', type=int, action='store',
                        help='If present, the last N records of the dataset will be used only. '
                             'If negative, then the first N records will be skipped and the remainder will be used only.')
    args = parser.parse_args()

    import pandas as pd

    print("reading pickle file...")
    df = pd.read_pickle(args.data)

    if args.tail:
        df = df.tail(args.tail)

    if args.head:
        df = df.head(args.head)

    chunks = np.array_split(np.array(df), args.workers)

    print("generating pairs...")

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(args.processes)

    watcher = pool.apply_async(listener, (queue, args.out, args.zip, len(df),))

    jobs = []
    for i in range(args.workers):
        chunk = chunks[i]
        job = pool.apply_async(worker, (chunk[:,0], queue, args.verbose))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
