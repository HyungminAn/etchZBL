import sys
from PlasmaEtchSimulator.etch_simulator import EtchSimulator


def main():
    if len(sys.argv) > 1:
        import yaml
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)

        pot_params  = config['pot_params']
        etch_params = config['etch_params']
        run_params  = config['run_params']
        etc_params  = config['etc_params']
        print(f"Load yaml : {sys.argv[1]}")
    else:
        raise ValueError('Please provide the yaml file')

    lmp = EtchSimulator(pot_params, etch_params,run_params, etc_params)
    lmp.init()
    lmp.run()


if __name__ == '__main__':
    main()
