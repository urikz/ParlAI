# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable,Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

class Starspace(nn.Module):
    def __init__(self, opt, num_features):
        super().__init__()
        self.lt = nn.Embedding(num_features, opt['embeddingsize'], 0)
        self.opt = opt
        self.encoder = Encoder(self.lt)

    def forward(self, xs, ys, cands):
        self.sf = StarspaceF(self.encoder, cands).apply
        return self.sf(xs, ys)

class StarspaceF(Function):
    def __init__(self, encoder, cands):
        self.encoder = encoder
        self.cands = cands

    @staticmethod
    def forward(ctx, xs, ys):
        import pdb; pdb.set_trace()
        scores = None
        xs_enc = self.encoder(xs)
        ys_enc = self.encoder(ys)
        scores = torch.matmul(xs_enc, ys_enc.t())
        c_scores = [scores]
        for c in cands:
            c_enc = self.encoder(c)
            c_scores.append(torch.matmul(xs_enc, c_enc.t()))
        scores = torch.cat(c_scores)
        scores =  scores.squeeze()
        pred = F.softmax(scores)
        return pred.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output):
      import pdb; pdb.set_trace()

class Encoder(nn.Module):
    def __init__(self, shared_lt):
        super().__init__()
        self.lt = shared_lt

    def forward(self, xs):
        xs_emb = self.lt(xs).mean(1)
        return xs_emb
