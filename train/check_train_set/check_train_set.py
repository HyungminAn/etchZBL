from abc import ABC, abstractmethod
import os
from collections import Counter

import yaml
import pickle

from ase.io import read


class DataChecker(ABC):
    @abstractmethod
    def _read_data(self):
        pass

    @abstractmethod
    def _check_num_atoms(self):
        pass

    @abstractmethod
    def _check_cell_size(self):
        pass

    @abstractmethod
    def check_data(self):
        pass


class VTSChecker(DataChecker):
    def __init__(self, inputs):
        self.path = inputs['path']
        self.idx_start = inputs['idx_start']
        self.idx_ends = inputs['idx_end']
        self.idx_step = inputs['idx_step']
        self.densities = inputs['density']
        self.compositions = inputs['composition']
        self.cal_type = ['melt', 'quench', 'anneal']
        self.data_path = {}

        self.prefix_dict_density = {
                'normal': '01_d_normal',
                'low': '02_d_low',
                }
        self.prefix_dict_comp = {
                '12816': '01_12816',
                '36313': '02_36313',
                '36211': '03_36211',
                '36050505': '04_36050505',
                '33111': '05_33111',
                '050505605': '06_050505605',
                '054050505': '07_054050505',
                '11003': '08_11003',
                }

    def check_data(self):
        self._read_data()
        self._check_num_atoms()
        self._check_cell_size()
        self._summarize()

    def _read_data(self):
        path_load = 'vts_data.pkl'
        if os.path.exists(path_load):
            self.data_path = self._load_data(path_load)
            return

        for density in self.densities:
            for comp in self.compositions:
                for mqa, idx_end in zip(self.cal_type, self.idx_ends):
                    key = (density, comp, mqa)
                    self.data_path[key] = []

                    for idx in range(self.idx_start, idx_end+1, self.idx_step):
                        path = f'{self.path}/{self.prefix_dict_density[density]}/{self.prefix_dict_comp[comp]}/{mqa}'
                        value = f'{path}/POSCAR_{idx}/OUTCAR'
                        if not os.path.exists(value):
                            raise FileNotFoundError(f'{value} does not exist')
                        self.data_path[key].append(value)
                        print(f'{density} {comp} {mqa} {idx} {value}')

        path_save = path_load
        self._save_data(path_save, self.data_path)

    @staticmethod
    def _load_data(path_load):
        with open(path_load, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def _save_data(path_save, data):
        with open(path_save, 'wb') as f:
            pickle.dump(data, f)

    def _check_num_atoms(self):
        result = {}
        for (density, comp, mqa), path_outcars in self.data_path.items():
            # for path_outcar in path_outcars:
            path_outcar = path_outcars[0]
            outcar = read(path_outcar, format='vasp-out')
            elems, counts = self._get_chemical_compositions(outcar)
            # print(f'{density} {comp} {mqa} {elems} {counts}')
            result[(density, comp, mqa)] = (elems, counts)
        self.data_num_atoms = result

    @staticmethod
    def _get_chemical_compositions(outcar):
        symbols = outcar.get_chemical_symbols()
        symbol_index_dict = {
                'Si': 0,
                'O': 1,
                'C': 2,
                'H': 3,
                'F': 4,
                }
        symbol_count = Counter(symbols)
        result = [0] * len(symbol_index_dict)
        for symbol, count in symbol_count.items():
            result[symbol_index_dict[symbol]] = count
        elems = ':'.join(symbol_index_dict.keys())
        counts = ':'.join(map(str, result))
        return elems, counts

    def _check_cell_size(self):
        result = {}
        for (density, comp, mqa), path_outcars in self.data_path.items():
            path_outcar = path_outcars[0]
            outcar = read(path_outcar, format='vasp-out')
            cell = outcar.get_cell()
            cell_size = cell.diagonal()
            result[(density, comp, mqa)] = cell_size
            # print(f'{density} {comp} {mqa} {cell_size}')
        self.data_cell_size = result

    def _summarize(self):
        print(
            ('density comp mqa elems_normal counts_normal '
            'volume_normal elems_low counts_low volume_low '
            'cell_size_normal cell_size_low ')
            )
        for key in self.data_num_atoms:
            (density, comp, mqa) = key
            if density == 'low':
                continue
            key_normal = key
            key_low = ('low', comp, mqa)

            elems_normal, counts_normal = self.data_num_atoms[key_normal]
            elems_low, counts_low = self.data_num_atoms[key_low]

            cell_size_normal = self.data_cell_size[key_normal]
            cell_size_low = self.data_cell_size[key_low]

            volume_normal = cell_size_normal[0] * cell_size_normal[1] * cell_size_normal[2]
            volume_low = cell_size_low[0] * cell_size_low[1] * cell_size_low[2]

            print(
                (f'{density} {comp} {mqa} {elems_normal} {counts_normal} '
                f'{volume_normal} {elems_low} {counts_low} {volume_low} '
                f'{cell_size_normal} {cell_size_low} ')
                )


def main():
    with open('path_dict.yaml') as f:
        inputs = yaml.safe_load(f)

    vts = VTSChecker(inputs['vts'])
    vts.check_data()


if __name__ == '__main__':
    main()
