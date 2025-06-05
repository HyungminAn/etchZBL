import pickle

from plotter import EnergyDiffPlotter, ParityPlotter


def main():
    with open('EF_original.pkl', 'rb') as f:
        data_ref = pickle.load(f)
    with open('EF_oneshot.pkl', 'rb') as f:
        data = pickle.load(f)

    data_to_plot = {
        'x': data_ref,
        'y': data,
    }
    # edp = EnergyDiffPlotter()
    # edp.run(data_to_plot)

    pp = ParityPlotter()
    pp.run(data_to_plot)


if __name__ == '__main__':
    main()
