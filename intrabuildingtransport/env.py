from intrabuildingtransport.mansion.person_generators.generator_proxy import set_seed
from intrabuildingtransport.mansion.person_generators.generator_proxy import PersonGenerator
from intrabuildingtransport.mansion.mansion_config import MansionConfig
from intrabuildingtransport.mansion.utils import ElevatorState, MansionState
from intrabuildingtransport.mansion.mansion_manager import MansionManager

from animation.rendering import Render

import configparser
import random
import sys
sys.path.append('.')
sys.path.append('..')


class IntraBuildingEnv():
    '''
    IntraBuildingTransportation Environment
    '''

    def __init__(self, file_name):
        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])

        # Readin different person generators
        gtype = config['PersonGenerator']['PersonGeneratorType']
        person_generator = PersonGenerator(gtype)
        person_generator.configure(config['PersonGenerator'])

        self._config = MansionConfig(
            dt=time_step,
            number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            floor_height=float(config['MansionInfo']['FloorHeight'])
        )

        if('LogLevel' in config['Configuration']):
            self._config.set_logger_level(config['Configuration']['LogLevel'])
        if('Lognorm' in config['Configuration']):
            self._config.set_std_logfile(config['Configuration']['Lognorm'])
        if('Logerr' in config['Configuration']):
            self._config.set_err_logfile(config['Configuration']['Logerr'])

        self._mansion = MansionManager(
            int(config['MansionInfo']['ElevatorNumber']),
            person_generator,
            self._config,
            config['MansionInfo']['Name']
        )

        self.viewer = Render(self._mansion)

    def seed(self, seed=None):
        set_seed(seed)

    def step(self, action):
        time_consume, energy_consume, given_up_persons = self._mansion.run_mansion(
            action)
        reward = - (time_consume + 0.01 * energy_consume +
                    1000 * given_up_persons) * 1.0e-5
        return (self._mansion.state, reward, {})

    def reset(self):
        self._mansion.reset_env()
        return self._mansion.state

    def render(self):
        self.viewer.view()
        # raise NotImplementedError()

    def close(self):
        pass

    @property
    def attribute(self):
        return self._mansion.attribute

    @property
    def state(self):
        return self._mansion.state

    @property
    def statistics(self):
        return self._mansion.get_statistics()

    @property
    def log_debug(self):
        return self._config.log_notice

    @property
    def log_notice(self):
        return self._config.log_notice

    @property
    def log_warning(self):
        return self._config.log_warning

    @property
    def log_fatal(self):
        return self._config.log_fatal
