# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.worlds import validate, create_task
from parlai.mturk.core.worlds import MTurkTaskWorld, MTurkOnboardWorld


class ModelChatOnboardWorld(MTurkOnboardWorld):
    """This world gets the turker online."""

    def parley(self):
        """Send welcome message."""
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'Welcome onboard!'
        self.mturk_agent.observe(ad)
        _response = self.mturk_agent.act()
        self.episodeDone = True


class ModelChatWorld(MTurkTaskWorld):
    """World for letting Turkers chat with a local AI model."""

    evaluator_agent_id = 'Model Evaluator'

    def __init__(self, opt, model_agent, mturk_agent):
        """Set up world."""
        self.model_agent = model_agent
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        """Conduct one exchange between the turker and the model."""
        obs = self.mturk_agent.act()
        self.model_agent.observe(obs)
        response = self.model_agent.act()
        self.mturk_agent.observe(validate(response))

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.task_world.shutdown()
        self.mturk_agent.shutdown()

    def review_work(self):
        pass
