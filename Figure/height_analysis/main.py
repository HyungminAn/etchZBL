import sys

import yaml

# from imageloader import ImageLoader
from imageloader import ImageLoaderExtended
from processor import HeightChangeProcessor, CarbonChangeProcessor, FilmAnalyzer, CarbonNeighborProcessor
from plotter import DataPlotter
from plotter import DataPlotterSelected

class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.processors = {
            'height_change': HeightChangeProcessor(name),
            'carbon_thickness': CarbonChangeProcessor(name),
            'film_data': FilmAnalyzer(name),
            'neighbor_classification': CarbonNeighborProcessor(name),
        }

    def run(self, src_list):
        all_exists = all([p.check_exists() for p in self.processors.values()])
        if all_exists:
            images = {}
        else:
            # images = ImageLoader().run(src_list)
            images = ImageLoaderExtended().run(src_list)

        result = {}
        for k, p in self.processors.items():
            if k == 'height_change':
                x, y, labels = p.run(images, src_list)
            else:
                x, y, labels = p.run(images)
            result[k] = (x, y, labels)
        return result

def main():
    if len(sys.argv) != 2:
        print('Usage: python get_height.py input.yaml')
        sys.exit(1)

    path_yaml = sys.argv[1]
    with open(path_yaml, 'r') as f:
        inputs = yaml.safe_load(f)
    data = {}
    for ion in inputs.keys():
        for energy, src_list in inputs[ion].items():
            key = f'{ion}_{energy}'
            dp = DataProcessor(key)
            data[key] = dp.run(src_list)
    # dplot = DataPlotter()
    # dplot.run(data)
    dplot = DataPlotterSelected()
    dplot.run(data)

if __name__ == '__main__':
    main()
