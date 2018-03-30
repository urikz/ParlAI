# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This seq2seq model was trained on convai2:self.
"""

from parlai.core.build_data import download_models_from_aws
from parlai.core.params import ParlaiParser
from examples.eval_model import setup_args, eval_model
from parlai.agents.seq2seq.seq2seq import Seq2seqAgent


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='convai2:self',
        model='seq2seq',
        model_file='models:convai2/seq2seq/convai2_self_seq2seq_model',
        dict_file='models:convai2/seq2seq/dict_convai2_self',
        datatype='valid',
        batchsize=128,
    )
    opt = parser.parse_args()
    opt['model_type'] = 'seq2seq'
    fnames = ['convai2_self_seq2seq_model.tgz', 'dict_convai2_self']
    download_models_from_aws(opt, fnames, 'convai2')
    eval_model(parser, printargs=False)
