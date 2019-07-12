import sys
import random
from intrabuildingtransport.mansion.utils import EPSILON, HUGE
from intrabuildingtransport.mansion.utils import MansionAttribute


class DispatcherBase(object):
    '''
    A basic demonstration of dispatcher
    A dispatcher must provide policy and feedback function
    The policy function receives MansionState and output ElevatorAction Lists
    The feedback function receives reward
    '''

    def link_mansion(self, mansion_attr):
        assert isinstance(mansion_attr, MansionAttribute)
        self._mansion_attr = mansion_attr

    def load_settings(self):
        pass

    def feedback(self, state, action, r):
        return dict()

    def policy(self, state):
        raise NotImplementedError()
