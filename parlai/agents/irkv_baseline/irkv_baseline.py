# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Simple IR baselines.
# We plan to implement the following variants:
# Given an input message, either:
# (i) find the most similar message in the (training) dataset and output the response from that exchange; or
# (ii) find the most similar response to the input directly.
# (iii) if label_candidates are provided, simply ranks them according to their similarity to the input message.
# Currently only (iii) is used.
#
# Additonally, TFIDF is either used (requires building a dictionary) or not,
# depending on whether you train on the train set first, or not.

from parlai.core.agents import Agent
from drqa import retriever


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
        self.ranker = retriever.get_class('tfidf')(tfidf_path=opt['tfidf_path'])
        self.db = retriever.get_class('sqlite')(db_path=opt['docdb-path'])

    # def observe(self, obs):
    #     self.observation = obs
    #     return obs

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()

        if 'text' in obs:
            doc_ids, doc_scores = self.ranker.closest_docs(obs['text'], 1)
            reply['text_candidates'] = [
                self.db.get_doc_value(did) for did in doc_ids]
            reply['text'] = reply['text_candidates'][0]

        return reply

    # def save(self, fname=None):
    #     fname = self.opt.get('model_file', None) if fname is None else fname
    #     if fname:
    #         self.dictionary.save(fname + '.dict')
    #
    # def load(self, fname):
    #     self.dictionary.load(fname + '.dict')
