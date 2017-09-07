# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

task_config = {}

"""A short and descriptive title about the kind of task the HIT contains.
On the Amazon Mechanical Turk web site, the HIT title appears in search
results, and everywhere the HIT is mentioned.
"""
task_config['hit_title'] = 'Have a conversation.'


"""A description includes detailed information about the kind of task the
HIT contains.
On the Amazon Mechanical Turk web site, the HIT description appears in the
expanded view of search results, and in the HIT and assignment screens.
"""
task_config['hit_description'] = (
    'Have a conversation with the individual on the other side of the dialog')


"""One or more words or phrases that describe the HIT, separated by commas.
On MTurk website, these words are used in searches to find HITs.
"""
task_config['hit_keywords'] = 'chat,dialog'


"""A detailed task description that will be shown on the HIT task preview page
and on the left side of the chat page. Supports HTML formatting.
"""
task_config['task_description'] = \
    '''
In this task, you are going to have a conversation with the individual on the
other side of the dialog.
When you get bored, end the coversation.
Interesting conversations will be selected to receive bonuses.
<br><br>
If you are ready, please click "Accept HIT" to start this task.'''
