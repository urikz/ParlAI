# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from .modules import Starspace

import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch import optim
import torch.nn as nn
import time
from collections import deque

import copy
import os
import random
import math

class StarspaceAgent(Agent):
    """Simple implementation of the starspace algorithm: https://arxiv.org/abs/1709.03856
    """

    OPTIM_OPTS = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        StarspaceAgent.dictionary_class().add_cmdline_args(argparser)
        agent = argparser.add_argument_group('StarSpace Arguments')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-lr', '--learningrate', type=float, default=0.005,
                           help='learning rate')
        agent.add_argument('-opt', '--optimizer', default='sgd',
                           choices=StarspaceAgent.OPTIM_OPTS.keys(),
                           help='Choose between pytorch optimizers. '
                                'Any member of torch.optim is valid and will '
                                'be used with default params except learning '
                                'rate (as specified by -lr).')
        agent.add_argument('-hist', '--history-length', default=1, type=int,
                           help='Number of past utterances to remember. '
                                'These include self-utterances. Default 1.')
        agent.add_argument('-tr', '--truncate', type=int, default=-1,
                           help='truncate input & output lengths to speed up '
                           'training (may reduce accuracy). This fixes all '
                           'input and output to have a maximum length.')
        agent.add_argument('-k', '--neg-samples', type=int, default=50,
                           help='number k of negative samples per example')

    def __init__(self, opt, shared=None):
        """Set up model if shared params not set, otherwise no work to do."""
        super().__init__(opt, shared)

        self.reset_metrics()
        # all instances needs truncate param
        self.NULL_IDX = 0
        self.start2=99;
        self.ys_cache = []
        self.ys_cache_sz = opt['neg_samples']
        self.truncate = opt['truncate'] if opt['truncate'] > 0 else None
        self.history = deque(maxlen=(
            opt['history_length'] if opt['history_length'] > 0 else None))
        if shared:
            # set up shared properties
            self.dict = shared['dict']
            # answers contains a batch_size list of the last answer produced
            self.answers = shared['answers']
        else:
            # this is not a shared instance of this class, so do full init
            # answers contains a batch_size list of the last answer produced
            self.answers = [None] * 1
            states = {}
            if opt.get('model_file') and os.path.isfile(opt['model_file']):
                # load model parameters if available
                print('Loading existing model params from ' + opt['model_file'])
                new_opt, states = self.load(opt['model_file'])
                # override model-specific options with stored ones
                opt = self.override_opt(new_opt)

            if opt['dict_file'] is None and opt.get('model_file'):
                # set default dict-file if not set
                opt['dict_file'] = opt['model_file'] + '.dict'

            # load dictionary and basic tokens & vectors
            self.dict = DictionaryAgent(opt)
            self.id = 'Starspace'
            self.model = Starspace(opt, len(self.dict))

            # set up tensors once
            self.xs = torch.LongTensor(1, 1)
            self.ys = torch.LongTensor(1, 1)
            self.cands = torch.LongTensor(1, 1, 1)

            # set up modules
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

            if states:
                # set loaded states if applicable
                self.model.load_state_dict(states['model'])

            # set up optimizer
            lr = opt['learningrate']
            optim_class = StarspaceAgent.OPTIM_OPTS[opt['optimizer']]
            kwargs = {'lr': lr}
#            if opt['optimizer'] == 'sgd':
#                kwargs['momentum'] = 0.95
#                kwargs['nesterov'] = True
            self.optimizer = optim_class(self.model.parameters(), **kwargs)
            if states:
                if states['optimizer_type'] != opt['optimizer']:
                    print('WARNING: not loading optim state since optim class '
                          'changed.')
                else:
                    self.optimizer.load_state_dict(states['optimizer'])
        self.reset()

    def override_opt(self, new_opt):
        """Set overridable opts from loaded opt file.

        Print out each added key and each overriden key.
        Only override args specific to the model.
        """
        model_args = {'hiddensize', 'embeddingsize', 'numlayers', 'optimizer',
                      'encoder', 'decoder', 'lookuptable', 'attention',
                      'attention_length'}
        for k, v in new_opt.items():
            if k not in model_args:
                # skip non-model args
                continue
            if k not in self.opt:
                print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
            elif self.opt[k] != v:
                print('Overriding option [ {k}: {old} => {v}]'.format(
                      k=k, old=self.opt[k], v=v))
            self.opt[k] = v
        return self.opt

    def parse(self, text):
        """Convert string to token indices."""
        return self.dict.txt2vec(text)

    def v2t(self, vec):
        """Convert token indices to string of tokens."""
        if type(vec) == Variable:
            vec = vec.data
        new_vec = []
        for i in vec:
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def zero_grad(self):
        """Zero out optimizer."""
        self.optimizer.zero_grad()

    def update_params(self):
        """Do one optimization step."""
        self.optimizer.step()

    def reset(self):
        """Reset observation and episode_done."""
        self.observation = None
        self.episode_done = True

    def share(self):
        """Share internal states between parent and child instances."""
        shared = super().share()
        shared['answers'] = self.answers
        shared['dict'] = self.dict
        return shared

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        observation = observation.copy()

        if 'text' in observation:
            if observation['text'] == '':
                observation.pop('text')
            else:
                dialog = deque(maxlen=self.truncate)
                if self.episode_done:
                    self.history.clear()
                else:
                    # get last y if avail and add to history
                    batch_idx = self.opt.get('batchindex', 0)
                    if self.answers[batch_idx] is not None:
                        # use our last answer, which is the label during train
                        lastY = self.answers[batch_idx]
                        y_utt = deque(lastY, maxlen=self.truncate)
                        self.history.append(y_utt)
                        self.answers[batch_idx] = None  # forget last y now
                    # remember past dialog of history_length utterances
                    dialog += (tok for utt in self.history for tok in utt)

                # put START and END around text
                parsed_x = deque(self.parse(observation['text']), maxlen=self.truncate)
                # add curr x to history
                self.history.append(parsed_x)

                dialog += parsed_x
                observation['text'] = dialog

        self.observation = observation
        self.episode_done = observation['episode_done']

        return observation


    def report(self):
        metrics = self.metrics
        if metrics['total'] == 0:
            report = { 'mean_rank': opt['neg_samples'] }
        else:
            report = { 'mean_rank': metrics['mean_rank'] / metrics['total'],
                       'mlp_time': metrics['mlp_time'] / metrics['total'],
                       'tot_time': metrics['tot_time'] / metrics['total']}
        return report

    def reset_metrics(self):
        self.metrics = { 'mean_rank':0, 'total':0, 'mlp_time':0, 'tot_time':0 }

    def update_metrics(self, scores, mlp_time, non_mlp_time):
            # update metrics
            pos = scores[0]
            cnt = 0
            for i in range(1, len(scores)):
                if scores[i] >= pos:
                    cnt += 1
            self.metrics['mean_rank'] += cnt
            self.metrics['total'] += 1

            self.metrics['mlp_time'] += mlp_time
            self.metrics['tot_time'] += mlp_time + non_mlp_time

    def predict(self, xs, ys=None, cands=None, valid_cands=None, lm=False):
        """Produce a prediction from our model.

        Update the model using the targets if available, otherwise rank
        candidates as well if they are available and param is set.
        """
        self.start = time.time()
        if True: #with autograd.profiler.profile() as p: 
            is_training = ys is not None
            text_cand_inds, loss_dict = None, None
            if is_training and len(self.ys_cache) > 0:
                self.model.train()
                self.zero_grad()
                pred = self.model(xs, ys, self.ys_cache)
                loss = self.criterion(pred, Variable(torch.LongTensor([0])))
                loss.backward()
                self.update_params()
            else:
                return [{}]
                #self.model.eval()
                #predictions, scores, text_cand_inds = self.model(xs, ys, cands,
                #                                                 valid_cands)
        #print("********")
        rest = 0
        if self.start2 != 99:
            rest = self.start-self.start2
        self.start2 = time.time()
        #print("model::" + str(self.start2-self.start) + "  rest:" + str(rest))
        self.update_metrics(pred.data, self.start2-self.start, rest)

        #print(p.table())
        #import pdb; pdb.set_trace()
        return [{}]

# sparse True,  dot prod
#[time:30s parleys:3995 ] {'mean_rank': 0.0, 'mlp_time': 0.009091309138706751, 'tot_time': 0.009531252724783761}
#[ time:30s parleys:4335 ] {'mean_rank': 0.0, 'mlp_time': 0.00784910213751871, 'tot_time': 0.00819989306027772}

# sparse True,  matmul
#[ time:30s parleys:5356 ] {'mean_rank': 0.0, 'mlp_time': 0.006610582444082082, 'tot_time': 0.00694408548744492}
#[ time:30s parleys:5588 ] {'mean_rank': 0.0, 'mlp_time': 0.006343643530658709, 'tot_time': 0.006658966755154125}

#sparse=False
#with backward:  model::0.40613722801208496  rest:0.0005350112915039062
#w/o  backward:  model::0.0026650428771972656  rest:0.00032401084899902344

#sparse=True
#with backward:  model::0.006658077239990234  rest:0.00027108192443847656
#w/o  backward:  model::0.002260923385620117  rest:0.0003902912139892578


    def batchify(self, observations, lm=False):
        """Convert a list of observations into input & target tensors."""
        def valid(obs):
            # check if this is an example our model should actually process
            return 'text' in obs
        try:
            # valid examples and their indices
            valid_inds, exs = zip(*[(i, ex) for i, ex in
                                    enumerate(observations) if valid(ex)])
        except ValueError:
            # zero examples to process in this batch, so zip failed to unpack
            return None, None

        # set up the input tensors
        bsz = len(exs)

        # `x` text is already tokenized and truncated
        # sort by length so we can use pack_padded
        parsed_x = [ex['text'] for ex in exs]
        x_lens = [len(x) for x in parsed_x]
        ind_sorted = sorted(range(len(x_lens)), key=lambda k: -x_lens[k])

        exs = [exs[k] for k in ind_sorted]
        valid_inds = [valid_inds[k] for k in ind_sorted]
        parsed_x = [parsed_x[k] for k in ind_sorted]

        labels_avail = any(['labels' in ex for ex in exs])

        max_x_len = max([len(x) for x in parsed_x])
        for x in parsed_x:
            x += [[self.NULL_IDX]] * (max_x_len - len(x))
        xs = torch.LongTensor(parsed_x)
        xs = Variable(xs)

        # set up the target tensors
        ys = None
        labels = None
        if labels_avail:
            # randomly select one of the labels to update on, if multiple
            labels = [random.choice(ex.get('labels', [''])) for ex in exs]
            # parse each label and append END
            parsed_y = [deque(maxlen=self.truncate) for _ in labels]
            for dq, y in zip(parsed_y, labels):
                dq.extendleft(reversed(self.parse(y)))
            if lm:
                for x, y in zip(parsed_x, parsed_y):
                    if y.maxlen is not None:
                        y = deque(y, maxlen=y.maxlen * 2)
                    y.extendleft(reversed(x))

            max_y_len = max(len(y) for y in parsed_y)
            for y in parsed_y:
                y += [self.NULL_IDX] * (max_y_len - len(y))
            ys = torch.LongTensor(parsed_y)
            ys = Variable(ys)
        return xs, ys

    def add_to_ys_cache(self, ys):
        if ys is None or len(ys) == 0:
            return
        if len(self.ys_cache) < self.ys_cache_sz:
            self.ys_cache.append(copy.deepcopy(ys))
        else:
            ind = random.randint(0, self.ys_cache_sz - 1)
            self.ys_cache[ind] = copy.deepcopy(ys)

    def batch_act(self, observations):
        batchsize = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets
        # valid_inds tells us the indices of all valid examples
        # e.g. for input [{}, {'text': 'hello'}, {}, {}], valid_inds is [1]
        # since the other three elements had no 'text' field
        xs, ys = self.batchify(observations)
        batch_reply = self.predict(xs, ys)
        self.add_to_ys_cache(ys)

        return batch_reply

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            model['model'] = self.model.state_dict()
            model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']
            model['opt'] = self.opt

            with open(path, 'wb') as write:
                torch.save(model, write)

    def shutdown(self):
        """Save the state of the model when shutdown."""
        path = self.opt.get('model_file', None)
        if path is not None:
            self.save(path + '.shutdown_state')
        super().shutdown()

    def load(self, path):
        """Return opt and model states."""
        with open(path, 'rb') as read:
            model = torch.load(read)

        return model['opt'], model
