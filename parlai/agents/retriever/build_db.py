<<<<<<< HEAD
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""A script to read in and store ParlAI tasks in a sqlite database.

Adapted from Adam Fisch's work at github.com/facebookresearch/DrQA/
"""
=======
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""
>>>>>>> e59f0d7f3652e6007ed329d2e5bef290d677c535

import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from . import utils

<<<<<<< HEAD
from collections import deque
import random
from parlai.core.agents import create_task_agent_from_taskname

=======
>>>>>>> e59f0d7f3652e6007ed329d2e5bef290d677c535
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

<<<<<<< HEAD
=======

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


>>>>>>> e59f0d7f3652e6007ed329d2e5bef290d677c535
# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


<<<<<<< HEAD
def store_contents(opt, task, save_path, context_length=-1, include_labels=True):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        task: ParlAI tasks of text (and possibly values) to store.
        save_path: Path to output sqlite db.
=======
def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = {}
    with open(filename) as f:
        for line in f:
            # Parse document
            try:
                doc = json.loads(line)
            except json.decoder.JSONDecodeError:
                print('JSON error processing line, skipping it: ', line[:30], '...')
                continue
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents[doc['id']] = (doc['text'], doc.get('value'))
    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
>>>>>>> e59f0d7f3652e6007ed329d2e5bef290d677c535
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
<<<<<<< HEAD
    c.execute('CREATE TABLE documents (id INTEGER PRIMARY KEY, text, value);')

    ordered_opt = opt.copy()
    dt = opt.get('datatype', '').split(':')
    ordered_opt['datatype'] = ':'.join([dt[0], 'ordered'] + dt[1:])
    ordered_opt['batchsize'] = 1
    ordered_opt['numthreads'] = 1
    ordered_opt['task'] = task
    teacher = create_task_agent_from_taskname(ordered_opt)[0]

    episode_done = False
    current = []
    triples = []
    context_length = context_length if context_length >= 0 else None
    context = deque(maxlen=context_length)
    with tqdm(total=teacher.num_episodes()) as pbar:
        while not teacher.epoch_done():
            # collect examples in episode
            while not episode_done:
                action = teacher.act()
                current.append(action)
                episode_done = action['episode_done']

            for ex in current:
                if 'text' in ex:
                    text = ex['text']
                    context.append(text)
                    if len(context) > 1:
                        text = '\n'.join(context)

                # add labels to context
                labels = ex.get('labels', ex.get('eval_labels'))
                if labels is not None:
                    label = random.choice(labels)
                    if include_labels:
                        context.append(label)
                # use None for ID to auto-assign doc ids--we don't need to
                # ever reverse-lookup them
                triples.append((None, text, label))

            c.executemany('INSERT OR IGNORE INTO documents VALUES (?,?,?)',
                          triples)
            pbar.update()

            # reset flags and content
            episode_done = False
            triples.clear()
            current.clear()
            context.clear()

    logger.info('Read %d examples from %d episodes.' % (
        teacher.num_examples(), teacher.num_episodes()))
    logger.info('Committing...')
    conn.commit()
    conn.close()
=======
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, value);")

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for triples in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(triples)
            triple_list = [(k, v[0], v[1]) for k, v in triples.items()]
            c.executemany("INSERT OR IGNORE INTO documents VALUES (?,?,?)", triple_list)
            pbar.update()
    logger.info('Read %d docs from %d files.' % (count, len(files)))
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path, args.preprocess, args.num_workers
    )
>>>>>>> e59f0d7f3652e6007ed329d2e5bef290d677c535
