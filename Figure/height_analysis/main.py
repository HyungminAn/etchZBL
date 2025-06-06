import sys

import yaml

# from imageloader import ImageLoader
from imageloader import ImageLoaderExtended
from processor import HeightChangeProcessor, MixedRegionIdentifier, FilmRegionIdentifier, CarbonNeighborProcessor, ProfileProcessor
from processor import AverageDensityProcessor, FCratioProcessor, SpxRatioProcessor
from processor import HydrogenEffectProcessor
from plotter import DataPlotter
from plotter import DataPlotterSelected

class DataProcessor:
    def __init__(self, name, system=None):
        self.name = name
        self.system = system
        self.processors_1st = {
            'height_change': HeightChangeProcessor(name, 'shifted_height.txt'),
            'neighbor_classification': CarbonNeighborProcessor(name, 'carbon_neighbors.txt', system=system),
            'profile': ProfileProcessor(name, 'atom_density.pkl', system=system),
        }
        self.processors_2nd = {
            'z_mixed': MixedRegionIdentifier(name, 'z_mixed.txt'),
            'z_film': FilmRegionIdentifier(name, 'z_film.txt'),
        }
        self.processors_3rd = {
            'density_mixed': AverageDensityProcessor(name, 'density_mixed.txt'),
            'density_film': AverageDensityProcessor(name, 'density_film.txt'),
            'fc_ratio_mixed': FCratioProcessor(name, 'fc_ratio_mixed.txt'),
            'fc_ratio_film': FCratioProcessor(name, 'fc_ratio_film.txt'),
            'spx_ratio_mixed': SpxRatioProcessor(name, 'spx_ratio_mixed.txt', system=system),
            'spx_ratio_film': SpxRatioProcessor(name, 'spx_ratio_film.txt', system=system),
            'h_effect_mixed': HydrogenEffectProcessor(name, 'h_effect_mixed.txt', system=system),
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
        for k, p in self.processors_3rd.items():
            if 'mixed' in k:
                x, y, labels = p.run(images, z_mix_dict)
            elif 'film' in k:
                x, y, labels = p.run(images, z_film_dict)
            else:
                raise ValueError(f'Unknown processor type: {k}')
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
    # dplot = DataPlotter()
    # dplot.run(data)
    # dplot = DataPlotterSelected()
    # dplot.run(data)

if __name__ == '__main__':
    main()
