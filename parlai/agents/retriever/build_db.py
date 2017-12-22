# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""A script to read in and store ParlAI tasks in a sqlite database.

Adapted from Adam Fisch's work at github.com/facebookresearch/DrQA/
"""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from . import utils

from collections import deque
from parlai.core.agents import create_task_agent_from_taskname

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


# def iter_files(path):
#     """Walk through all files located under a root path."""
#     if os.path.isfile(path):
#         yield path
#     elif os.path.isdir(path):
#         for dirpath, _, filenames in os.walk(path):
#             for f in filenames:
#                 yield os.path.join(dirpath, f)
#     else:
#         raise RuntimeError('Path %s is invalid' % path)
#
#
# def get_contents(filename):
#     """Parse the contents of a file. Each line is a JSON encoded document."""
#     global PREPROCESS_FN
#     documents = {}
#     with open(filename) as f:
#         for line in f:
#             # Parse document
#             try:
#                 doc = json.loads(line)
#             except json.decoder.JSONDecodeError:
#                 print('JSON error processing line, skipping it: ', line[:30], '...')
#                 continue
#             # Maybe preprocess the document with custom function
#             if PREPROCESS_FN:
#                 doc = PREPROCESS_FN(doc)
#             # Skip if it is empty or None
#             if not doc:
#                 continue
#             # Add the document
#             documents[doc['id']] = (doc['text'], doc.get('value'))
#     return documents


def store_contents(opt, task, save_path, context_length=-1, include_labels=True):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        task: ParlAI tasks of text (and possibly values) to store.
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute('CREATE TABLE documents (id INTEGER PRIMARY KEY, text, value);')

    ordered_opt = opt.copy()
    dt = opt.get('datatype', '').split(':')
    ordered_opt['datatype'] = ':'.join([dt[0], 'ordered'] + dt[1:])
    ordered_opt['batchsize'] = 1
    ordered_opt['numthreads'] = 1
    opt['task'] = task
    teacher = create_task_agent_from_taskname(opt)[0]

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
                triples.append((None, text, label))

            c.executemany('INSERT OR IGNORE INTO documents VALUES (?,?,?)',
                          triples)
            pbar.update()

            # reset flags and content
            episode_done = False
            current.clear()
            context.clear()

    logger.info('Read %d examples from %d episodes.' % (
        teacher.num_examples(), teacher.num_episodes()))
    logger.info('Committing...')
    conn.commit()
    conn.close()
