# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .build_db import store_contents as build_db
from .build_tfidf import run as build_tfidf
from numpy.random import choice
import math
import os


# TODO: use parlai.core.utils.AttrDict
class AttrDict(dict):
    """Helper class to have a dict-like object with dot access.

    For example, instead of `d = {'key': 'value'}` use
    `d = AttrDict(key='value')`.
    To access keys, instead of doing `d['key']` use `d.key`.

    While this has some limitations on the possible keys (for example, do not
    set the key `items` or you will lose access to the `items()` method), this
    can make some code more clear.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class RetrieverAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Retriever Arguments')
        parser.add_argument(
            '--retriever-task', type=str, default=None,
            help='ParlAI task to use to "train" retriever')
        parser.add_argument(
            '--retriever-dbpath', type=str, required=True,
            help='/path/to/saved/db.db')
        parser.add_argument(
            '--retriever-tfidfpath', type=str, required=True,
            help='Directory for saving output files')
        parser.add_argument(
            '--retriever-numworkers', type=int, default=None,
            help='Number of CPU processes (for tokenizing, etc)')
        parser.add_argument(
            '--retriever-ngram', type=int, default=2,
            help='Use up to N-size n-grams (e.g. 2 = unigrams + bigrams)')
        parser.add_argument(
            '--retriever-hashsize', type=int, default=int(math.pow(2, 24)),
            help='Number of buckets to use for hashing ngrams')
        parser.add_argument(
            '--retriever-tokenizer', type=str, default='simple',
            help='String option specifying tokenizer type to use '
                 '(e.g. "corenlp")')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RetrieverAgent'

        # we'll need to build the tfid if it's not already
        rebuild_tfidf = not os.path.exists(opt['retriever_tfidfpath'] + '.npz')
        # sets up db
        if not os.path.exists(opt['retriever_dbpath']):
            if not opt.get('retriever_task'):
                raise RuntimeError('Retriever task param required to build db')
            build_db(opt, opt['retriever_task'], opt['retriever_dbpath'],
                     context_length=opt.get('context_length', -1),
                     include_labels=opt.get('include_labels', True))
            # we rebuilt the db, so need to force rebuilding of tfidf
            rebuild_tfidf = True


        if rebuild_tfidf:
            # build tfidf if we built the db or if it doesn't exist
            build_tfidf(AttrDict({
                'db_path': opt['retriever_dbpath'],
                'out_dir': opt['retriever_tfidfpath'],
                'ngram': opt['retriever_ngram'],
                'hash_size': opt['retriever_hashsize'],
                'tokenizer': opt['retriever_tokenizer'],
                'num_workers': opt['retriever_numworkers'],
            }))

        self.db = DocDB(db_path=opt['retriever_dbpath'])
        self.ranker = TfidfDocRanker(
            tfidf_path=opt['retriever_tfidfpath'] + '.npz', strict=False)

    def train(mode=True):
        self.training = mode

    def eval():
        self.training = False

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        if 'text' in obs:
            doc_ids, doc_scores = self.ranker.closest_docs(obs['text'], 30)
            total = sum(doc_scores)
            import pdb; pdb.set_trace()
            if len(doc_ids) == 0:
                reply['text'] = choice([
                    'Can you say something more interesting?',
                    'Why are you being so short with me?',
                    'What are you really thinking?',
                    'Can you expand on that?',
                ])
            else:
                doc_scores = [d / total for d in doc_scores]
                pick = choice(doc_ids, p=doc_scores)
                reply['text_candidates'] = [
                    self.db.get_doc_value(did) for did in doc_ids]
                text = self.db.get_doc_value(pick)
                if len(text) > 100:
                    # shrink it a bit so it's not too long to read
                    idx = text.rfind('.', 10, 125)
                    if idx > 0:
                        text = text[:idx + 1]
                    else:
                        idx = text.rfind('?', 10, 125)
                        if idx > 0:
                            text = text[:idx + 1]
                        else:
                            idx = text.rfind('!', 10, 125)
                            if idx > 0:
                                text = text[:idx + 1]
                            else:
                                idx = text.rfind(' ', 0, 75)
                                if idx > 0:
                                    text = text[:idx]
                                else:
                                    text = text[:50]
                reply['text'] = text

        return reply
