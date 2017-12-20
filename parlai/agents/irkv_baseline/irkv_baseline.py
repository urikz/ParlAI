# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from drqa import retriever
from numpy.random import choice


class IrkvBaselineAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        agent = parser.add_argument_group('IR KV Arguments')
        agent.add_argument(
            '-tfp', '--tfidf-path', required=True,
            help='path to stored tfidf weightings (use script to pregenerate)')
        agent.add_argument(
            '-dbp', '--docdb-path', required=True,
            help='path to stored doc db (use script to pregenerate)')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'IRKVBaselineAgent'
        self.ranker = retriever.get_class('tfidf')(tfidf_path=opt['tfidf_path'],
                                                   strict=False)
        self.db = retriever.get_class('sqlite')(db_path=opt['docdb_path'])

    # def observe(self, obs):
    #     self.observation = obs
    #     return obs

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        if 'text' in obs:
            doc_ids, doc_scores = self.ranker.closest_docs(obs['text'], 30)
            total = sum(doc_scores)
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

    # def save(self, fname=None):
    #     fname = self.opt.get('model_file', None) if fname is None else fname
    #     if fname:
    #         self.dictionary.save(fname + '.dict')
    #
    # def load(self, fname):
    #     self.dictionary.load(fname + '.dict')
