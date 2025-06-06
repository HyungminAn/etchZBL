import sys

import yaml

# from imageloader import ImageLoader
from imageloader import ImageLoaderExtended
from processor import HeightChangeProcessor, MixedFilmRegionIdentifier, CarbonNeighborProcessor
from processor import AverageDensityProcessor, FCratioProcessor, SpxRatioProcessor
from plotter import DataPlotter
from plotter import DataPlotterSelected

class DataProcessor:
    def __init__(self, name, system=None):
        self.name = name
        self.system = system
        self.processors = {
            'height_change': HeightChangeProcessor(name, 'shifted_height.txt'),
            'z_data': MixedFilmRegionIdentifier(name, 'thickness.txt', system=system),
            'neighbor_classification': CarbonNeighborProcessor(name, 'carbon_neighbors.txt', system=system),
        }
        self.dependent_processors = {
            'density': AverageDensityProcessor(name, 'density.txt'),
            'fc_ratio': FCratioProcessor(name, 'fc_ratio.txt'),
            'spx_ratio': SpxRatioProcessor(name, 'spx_ratio.txt', system=system),
        }

    def run(self, src_list):
        all_exists = all([p.check_exists() for p in self.processors.values()])
        if all_exists:
            images = {}
        else:
            # images = ImageLoader().run(src_list)
            images = ImageLoaderExtended(self.system).run(src_list)

        result = {}
        for k, p in self.processors.items():
            if k == 'height_change':
                x, y, labels = p.run(images, src_list)
            else:
                x, y, labels = p.run(images)
            result[k] = (x, y, labels)

        height_dict = result['height_change'][2]
        for k, p in self.dependent_processors.items():
            x, y, labels = p.run(images)
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
    dplot = DataPlotterSelected()
    dplot.run(data)

if __name__ == '__main__':
    main()
