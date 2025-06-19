import sys

import yaml

from imageloader import ImageLoaderExtended
from processor import HeightChangeProcessor, MixedRegionIdentifier, FilmRegionIdentifier, CarbonNeighborProcessor, ProfileProcessor
from processor import AtomCountProcessor
from processor import MixedFilmStackedProcessor
from processor import EtchedAmountProcessor
from plotter import DataPlotter
from plotter import DataPlotterSelected

class DataProcessor:
    def __init__(self, name, system=None):
        self.name = name
        self.system = system
        self.processors_1st = {
            'height_change': HeightChangeProcessor(name, 'shifted_height.txt', system=system),
            'neighbor_classification': CarbonNeighborProcessor(name, 'carbon_neighbors.txt', system=system),
            'profile': ProfileProcessor(name, 'atom_density.pkl', system=system),
        }
        self.processors_2nd = {
            'z_mixed': MixedRegionIdentifier(name, 'z_mixed.txt'),
            'z_film': FilmRegionIdentifier(name, 'z_film.txt'),
        }
        self.processors_3rd = {
            'atomcount_mixed': AtomCountProcessor(name, 'atomcount_mixed.txt', system=system),
            'atomcount_film': AtomCountProcessor(name, 'atomcount_film.txt', system=system),
            'atomcount_mixed_norm': AtomCountProcessor(name, 'atomcount_mixed.txt', system=system),
            'atomcount_film_norm': AtomCountProcessor(name, 'atomcount_film.txt', system=system),
            'stacked': MixedFilmStackedProcessor(name, 'mixed_film_stacked.txt'),
            'etchedamount': EtchedAmountProcessor(name, 'etched_amount.txt', system=system),
        }

    def run(self, src_list):
        all_exists = all([p.check_exists() for p in self.processors_1st.values()] +
                         [p.check_exists() for p in self.processors_2nd.values()] +
                         [p.check_exists() for p in self.processors_3rd.values()])
        if all_exists:
            images = {}
        else:
            # images = ImageLoader().run(src_list)
            images = ImageLoaderExtended(self.name, self.system).run(src_list)
            print('Loading images Done')

        result = {}
        # Run first-level processors (independent of images)
        for k, p in self.processors_1st.items():
            print(f'Running processor: {k}')
            if k == 'height_change':
                x, y, labels = p.run(images, src_list)
            elif k == 'profile':
                x, y, labels = p.run(images), None, None
            else:
                x, y, labels = p.run(images)
            result[k] = (x, y, labels)
            print(f'processor {k} Done')

        # Run second-level processors (dependent on profile)
        profile, _, _ = result['profile']
        for k, p in self.processors_2nd.items():
            print(f'Running processor: {k}')
            x, y, labels = p.run(profile)
            result[k] = (x, y, labels)
            print(f'processor {k} Done')

        # Run third-level processors (dependent on mixed/film regions)
        z_mix_dict = {xx: yy for xx, yy in zip(result['z_mixed'][0], result['z_mixed'][1])}
        z_film_dict = {xx: yy for xx, yy in zip(result['z_film'][0], result['z_film'][1])}
        z_height_dict = {xx: yy for xx, yy in zip(result['height_change'][0],
                                                  result['height_change'][1])}
        for k, p in self.processors_3rd.items():
            if 'mixed' in k:
                x, y, labels = p.run(images, z_mix_dict)
            elif 'film' in k:
                x, y, labels = p.run(images, z_film_dict)
            elif 'stacked' in k:
                x, y, labels = p.run(images, z_mix_dict, z_film_dict)
            elif 'etchedamount' in k:
                x, y, labels = p.run(z_height_dict, z_film_dict)
            else:
                raise ValueError(f'Unknown processor type: {k}')

            if 'atomcount' in k:
                # Normalize atom counts
                if 'norm' in k:
                    y = y[:, 5:]
                else:
                    y = y[:, :5]
            result[k] = (x, y, labels)

        return result

def main():
    if len(sys.argv) != 3:
        print('Usage: python get_height.py input.yaml <system:SiO2/Si3N4>')
        sys.exit(1)

    path_yaml = sys.argv[1]
    system = sys.argv[2]
    with open(path_yaml, 'r') as f:
        inputs = yaml.safe_load(f)
    data = {}
    for ion in inputs.keys():
        for energy, src_list in inputs[ion].items():
            key = f'{ion}_{energy}'
            dp = DataProcessor(key, system=system)
            data[key] = dp.run(src_list)
    ylim = (-15, 15)
    # dplot = DataPlotter(system, ylim=ylim)
    # dplot.run(data, twin_axis=True)
    dplot = DataPlotterSelected(system, ylim=ylim)
    dplot.run(data)

if __name__ == '__main__':
    main()
