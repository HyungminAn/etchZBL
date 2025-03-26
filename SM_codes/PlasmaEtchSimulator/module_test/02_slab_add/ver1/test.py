import os
import sys
import yaml

from PlasmaEtchSimulator.calc.strprocess import StrProcess
from PlasmaEtchSimulator.key import KEY
from PlasmaEtchSimulator.etch_simulator import Logger


class SlabAddTest:
    def __init__(self):
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)

        self.pot_params  = config['pot_params']
        self.etch_params = config['etch_params']
        self.run_params  = config['run_params']
        self.etc_params  = config['etc_params']

        self._set_log()

    def _set_log(self):
        '''
        Set the log file location
        '''
        log_path = '.'
        if KEY.LOG in self.etc_params.keys():
            logloc = os.path.join(f'{log_path}/log.txt')
            self.log = Logger(logloc)
        else:
            self.log = None

    def run(self):
        StrProcess.set_etch_params(self.etch_params)
        calc = StrProcess(0,
                          self.run_params,
                          self.etc_params,
                          elmlist=self.pot_params[KEY.ELMLIST],
                          log=self.log,
                          )
        calc.update()
        # calc.remove_byproduct()
        calc.slab_modify_height()
        # calc.save()


if __name__ == '__main__':
    slab_add_test = SlabAddTest()
    slab_add_test.run()
