import copy
import re

import torch
import numpy as np

import ase
import ase.io
import ase.build
from ase.data import chemical_symbols

from PlasmaEtchSimulator.calc.functions import init_out
from PlasmaEtchSimulator.calc.functions import save_atom
from PlasmaEtchSimulator.calc.graphbuilder import GraphBuilder

class EmptyLogger:
    def __init__(self) -> None:
        pass
    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)


class ByProductRemover:
    def __init__(self, elmlist : list,byproduct : list , logger = EmptyLogger() ) -> None:
        self.elmlist = elmlist
        self.convert_dict = {chemical_symbols[idx+1]: symbol  for idx, symbol in enumerate(elmlist)}

        self.byproduct_list = byproduct
        self.byproduct_dict  = {}
        self.byproduct_label = {}
        self._parse_byproducts()
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, calc) -> None:
        """Remove byproducts of the calculation
        """
        if calc.calc_done():
            return

        if calc.str_out == None:
            assert calc.str_in != None, f'The input structure is not loaded {calc.name_in}'
            calc.str_out = init_out(calc.box_height, calc.str_in)

        ## Generate graph networks
        GraphBuilder.graph_call_in(calc)

        remove_cluster_height = calc.etch_params['remove_high_clutser_height_crit']
        fix_height = calc.etch_params['fix']
        ### ------------------------- For debugging --------------------
        is_remove_within_removal_region = calc.etch_params.get('remove_within_removal_region', False)
        if is_remove_within_removal_region:
            calc.str_out = self._remove_byproduct_within_removal_region(calc.str_out, calc.clusters_in, remove_cluster_height, fix_height)
            self.logger.print(f'Remove byproducts within removal region')
        ### ------------------------- For debugging --------------------
        else:
            calc.str_out = self._remove_byproduct(calc.str_out,
                                                  calc.clusters_in,
                                                  remove_cluster_height,
                                                  fix_height
                                                  )
        save_atom(calc, 'rm_byproduct_'+calc.name_in, calc.str_out)
        calc.byproduct_removed = True

    @staticmethod
    def _key_to_label(chemical_symbols : list, number_of_atom : list) -> str:
        '''
        Convert the chemical symbols and number of atoms to the string label
        '''
        ## Sort the chemical symbols and number of atoms to the string label
        sorted_label = sorted(zip(chemical_symbols, number_of_atom), key = lambda x: x[0])
        label = ''.join([f'{symbol}{num}' for symbol, num in sorted_label])
        return label

    def _parse_byproducts(self) -> None:
        # Define the regular expression pattern
        epattern = '|'.join(self.elmlist)
        pattern = re.compile(r'(?P<element>'+epattern+r')(?P<count>\d*)')
        for  molecule  in self.byproduct_list:
            matches = pattern.findall(molecule)
            odict = {element: int(count) if count else 1 for element, count in matches}
            # Change dict to chemical symbols and number of atoms to the string label
            label = self._key_to_label(odict.keys(), odict.values())

            self.byproduct_dict[molecule] = odict
            self.byproduct_label[label] = molecule

    def _remove_byproduct(self,
                          atoms : ase.Atoms,
                          cluster_list : list,
                          slab_height_crit : float,
                          fix_height : float) -> ase.Atoms:
        oatom = copy.deepcopy(atoms)
        slab_idx = np.argmax([len(c) for c in cluster_list])
        pos_z = atoms.get_positions()[:, 2].flatten()
        h_slab_max = np.max(pos_z[np.array(list(cluster_list[slab_idx]))])

        # Get chemical symbols and number of atoms for each cluster
        idx_to_remove = []

        for idx, cluster in enumerate(cluster_list):
            ## Pass the cluster is slab
            if idx == slab_idx:
                continue

            h_cluster_min = np.min(pos_z[np.array(list(cluster))])
            is_within_fixed_region = h_cluster_min < fix_height
            if is_within_fixed_region:
                continue

            label, is_in_byproduct = self._is_in_byproduct(cluster, atoms)
            self.logger.print(f'Cluster {idx} : {label} is in byproduct list : {is_in_byproduct}')
            if is_in_byproduct:
                idx_to_remove += list(cluster)
                continue

            is_unnecessary_cluster = h_cluster_min - h_slab_max > slab_height_crit
            if is_unnecessary_cluster:
                self.logger.print(f'Cluster {idx} : {label} is unnecessary (height : {h_cluster_min:.2f} - {h_slab_max:.2f} > {slab_height_crit:.2f})')
                idx_to_remove += list(cluster)
                continue

        ## Remove the byproduct from the ase.Atom object
        del oatom[idx_to_remove]

        return oatom

    def _remove_byproduct_within_removal_region(self,
                                                atoms : ase.Atoms,
                                                cluster_list : list,
                                                slab_height_crit : float,
                                                fix_height : float) -> ase.Atoms:
        oatom = copy.deepcopy(atoms)
        slab_idx = np.argmax([len(c) for c in cluster_list])
        pos_z = atoms.get_positions()[:, 2].flatten()
        h_slab_max = np.max(pos_z[np.array(list(cluster_list[slab_idx]))])

        thickness_removal_region = 20.0  # arbitrary value
        removal_crit = h_slab_max - thickness_removal_region

        # Get chemical symbols and number of atoms for each cluster
        idx_to_remove = []

        for idx, cluster in enumerate(cluster_list):
            ## Pass the cluster is slab
            if idx == slab_idx:
                continue

            h_cluster_min = np.min(pos_z[np.array(list(cluster))])
            is_within_fixed_region = h_cluster_min < fix_height
            if is_within_fixed_region:
                continue

            h_cluster_max = np.max(pos_z[np.array(list(cluster))])
            if h_cluster_max < removal_crit:
                self.logger.print(f'debug - Cluster {idx} : {h_cluster_max:.2f} < {removal_crit:.2f} is within removal region')
                continue

            label, is_in_byproduct = self._is_in_byproduct(cluster, atoms)
            self.logger.print(f'Cluster {idx} : {label} is in byproduct list : {is_in_byproduct}')
            if is_in_byproduct:
                idx_to_remove += list(cluster)
                continue

            is_unnecessary_cluster = h_cluster_min - h_slab_max > slab_height_crit
            if is_unnecessary_cluster:
                self.logger.print(f'Cluster {idx} : {label} is unnecessary (height : {h_cluster_min:.2f} - {h_slab_max:.2f} > {slab_height_crit:.2f})')
                idx_to_remove += list(cluster)
                continue

        ## Remove the byproduct from the ase.Atom object
        del oatom[idx_to_remove]

        return oatom

    def _is_in_byproduct(self, cluster, atoms):
        '''
        Check the byproduct in the cluster is the one should be removed
        '''
        ndict = {}
        for cidx in cluster:
            symbol = atoms[cidx].symbol
            if symbol in ndict:
                ndict[symbol] += 1
            else:
                ndict[symbol] = 1

        label = self._key_to_label(ndict.keys(), ndict.values())
        return label, label in self.byproduct_label
