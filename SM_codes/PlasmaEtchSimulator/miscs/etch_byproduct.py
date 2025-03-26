import argparse 
import copy
import ase
import ase.io
import ase.build
import numpy as np
from ase.data import  covalent_radii
import torch
import os, sys
import networkx as nx
import re

class EmptyLogger:
    def __init__(self) -> None:
        pass
    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

class ByProduct:
    def __init__(self, elmlist : list,byproduct : list , logger = EmptyLogger() ) -> None:
        self.elmlist = elmlist
        self.byproduct_list = byproduct
        self.byproduct_dict  = {}
        self.byproduct_label = {}
        self._parse_byproducts()
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _key_to_label(chemical_symbols : list, number_of_atom : list) -> str:
        '''Convert the chemical symbols and number of atoms to the string label'''
        ## Sort the chemical symbols and number of atoms to the string label
        sorted_label = sorted(zip(chemical_symbols, number_of_atom), key = lambda x: x[0])
        label = ''.join([f'{symbol}{num}' for symbol, num in sorted_label])
        return label

    def _parse_byproducts(self) -> None:
        # Define the regular expression pattern
        epattern = '|'.join(self.elmlist)
        pattern  = re.compile(r'(?P<element>'+epattern+r')(?P<count>\d*)')
        for  molecule  in self.byproduct_list:
            matches = pattern.findall(molecule)
            odict = {element: int(count) if count else 1 for element, count in matches}
            # Change dict to chemical symbols and number of atoms to the string label
            label = self._key_to_label(odict.keys(), odict.values())

            self.byproduct_dict[molecule] = odict
            self.byproduct_label[label] = molecule

    def remove_byproduct(self, atoms : ase.Atoms, graph : nx.Graph, cluster_list : list) -> ase.Atoms:
        oatom = copy.deepcopy(atoms)
        slab_idx = np.argmax([len(c) for c in cluster_list])

        # Get chemical symbols and number of atoms for each cluster
        idx_to_remove = []
        for idx, cluster in enumerate(cluster_list):
            ndict = {}
            ## Pass the cluster is slab
            if idx == slab_idx:
                continue

            ## Check the byproduct in the cluster is the one should be removed
            for cidx in cluster:
                symbol = atoms[cidx].symbol
                if symbol in ndict:
                    ndict[symbol] += 1
                else:
                    ndict[symbol] = 1

            label = self._key_to_label(ndict.keys(), ndict.values())
            is_in_byproduct = label in self.byproduct_label
            self.logger.print(f'Cluster {idx} : {label} is in byproduct list : {is_in_byproduct}')
            if is_in_byproduct:
                idx_to_remove += list(cluster)
        ## Remove the byproduct from the ase.Atom object 
        del oatom[idx_to_remove]

        return oatom

    def remove_byproduct_wo_graph(self, atoms : ase.Atoms, distance_matrix ) -> ase.Atoms:
        """Remove the byproduct from the atoms object without using graph networks

        Args:
            atoms (ase.Atoms): ase.Atoms object to remove the byproduct
            distance_matrix (_type_): bond cutoff criteria

        Returns:
            ase.Atoms: _description_
        """
        atomic_n = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int).to(self.device)

        distance = torch.tensor(atoms.get_all_distances(mic=True)).to(self.device)
        ## Convert 0 value to infinite value in distance matrix
        distance.fill_diagonal_(float('inf'))
        ## Torch module to create networks accelerate the calculation
        distance_matrix_at_n = distance_matrix[atomic_n].to(self.device)
        cutoff = (distance_matrix_at_n[:, None] + distance_matrix_at_n[None, :])
        is_connected = distance < cutoff
        ## Create graph networks which is the distance matrix is smaller than the cutoff
         # Step 2: Generate edges from connectivity matrix
        edges = torch.nonzero(is_connected, as_tuple=False).tolist()  # Convert to list of pairs
        # Step 3: Create an undirected graph using NetworkX
        graph = nx.Graph()
        graph.add_edges_from(edges)
        # Step 4: Analyze connected components (clusters)
        cluster_list = list(nx.connected_components(graph))  # List of sets of connected nodes
        oatom = self.remove_byproduct(atoms, graph, cluster_list)
        return oatom


    def remove_all(self, atoms : ase.Atoms, graph : nx.Graph, cluster_list : list) -> ase.Atoms:
        oatom = copy.deepcopy(atoms)
        slab_idx = np.argmax([len(c) for c in cluster_list])

        # Get chemical symbols and number of atoms for each cluster
        idx_to_remove = []
        for idx, cluster in enumerate(cluster_list):
            ndict = {}
            ## Pass the cluster is slab
            if idx == slab_idx:
                continue

            for cidx in cluster:
                symbol = atoms[cidx].symbol
                if symbol in ndict:
                    ndict[symbol] += 1
                else:
                    ndict[symbol] = 1

            label = self._key_to_label(ndict.keys(), ndict.values())
            is_in_byproduct = label in self.byproduct_label
            self.logger.print(f'Cluster {idx} : {label}')

            idx_to_remove += list(cluster)
        ## Remove the byproduct from the ase.Atom object 
        del oatom[idx_to_remove]
        return oatom

    def remove_all_byproduct(self, atoms : ase.Atoms, distance_matrix ) -> ase.Atoms:
        """Remove the byproduct from the atoms object without using graph networks

        Args:
            atoms (ase.Atoms): ase.Atoms object to remove the byproduct
            distance_matrix (_type_): bond cutoff criteria

        Returns:
            ase.Atoms: _description_
        """
        atomic_n = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int).to(self.device)

        distance = torch.tensor(atoms.get_all_distances(mic=True)).to(self.device)
        ## Convert 0 value to infinite value in distance matrix
        distance.fill_diagonal_(float('inf'))
        ## Torch module to create networks accelerate the calculation
        distance_matrix_at_n = distance_matrix[atomic_n].to(self.device)
        cutoff = (distance_matrix_at_n[:, None] + distance_matrix_at_n[None, :])
        is_connected = distance < cutoff
        ## Create graph networks which is the distance matrix is smaller than the cutoff
         # Step 2: Generate edges from connectivity matrix
        edges = torch.nonzero(is_connected, as_tuple=False).tolist()  # Convert to list of pairs
        # Step 3: Create an undirected graph using NetworkX
        graph = nx.Graph()
        graph.add_edges_from(edges)
        # Step 4: Analyze connected components (clusters)
        cluster_list = list(nx.connected_components(graph))  # List of sets of connected nodes
        oatom = self.remove_all(atoms, graph, cluster_list)
        return oatom

if __name__ == '__main__':
    ## TEST code to check the class ByProduct
    elmlist = 'Si N C H F'.split()
    byproduct_list = [
        'SiF2',
        'SiF4',
        'N2',
        'H2',
        'F2',
        'HF',
        'NH3',
        'FCN',
        'HCN',
        'CN',
        'CF4',
        'CF3H',
        'CF2H2',
        'CFH3',
        'CH4',
        'CF3',
        'CH2F',
        'CHF2',
        'CH3',
        'CF2',
        'CHF',
        'CH2',
        'CF',
        'CH'
    ]
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <to_remove> <to_save> <format>")
        sys.exit(1)
    to_remove = sys.argv[1]
    to_save   = sys.argv[2]
    remove_all = True if 'all' == sys.argv[3] else False
    str_format = sys.argv[4] if len(sys.argv) > 4 else 'lammps-data' 
    scale      = 1.3
    try: 
        if str_format == 'vasp':
            atom_to_rm = ase.io.read(to_remove, format=str_format)
        elif str_format == 'lammps-data':
            atom_to_rm = ase.io.read(to_remove, format=str_format, atom_style='atomic', sort_by_id=False)
        else:
            atom_to_rm = ase.io.read(to_remove, format=str_format)



    except Exception as e:
        print(f"Error in reading {to_remove}, check the format {e}")
        sys.exit(1)

    rm_byproduct = ByProduct(elmlist, byproduct_list)

    distance_matrix  = torch.tensor(covalent_radii).to(rm_byproduct.device) * scale # Scaling
    if remove_all:
        oatom = rm_byproduct.remove_all_byproduct(atom_to_rm, distance_matrix)
    else:
        oatom = rm_byproduct.remove_byproduct_wo_graph(atom_to_rm, distance_matrix)

    if str_format == 'lammps-data':
        ase.io.write(to_save, oatom, format='lammps-data', specorder = elmlist) 
    elif str_format == 'vasp':
        ase.io.write(to_save, oatom, format='vasp')
    print(f"Write the atoms object from {to_remove} to {to_save}")
 
