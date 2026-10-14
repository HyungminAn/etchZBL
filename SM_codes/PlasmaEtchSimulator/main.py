"""
Entry point for PlasmaEtchSimulator.

Usage:
    python -m PlasmaEtchSimulator.main input.yaml
    # or after installation:
    etch-simulator input.yaml
"""
import sys
from PlasmaEtchSimulator import EtchSimulator


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: etch-simulator <config.yaml>')
        sys.exit(1)
    EtchSimulator(sys.argv[1]).run()


if __name__ == '__main__':
    main()
