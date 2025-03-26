import ase
import ase.io
import ase.build
from ase.data import covalent_radii
import torch

from PlasmaEtchSimulator.calc.byproduct import ByProduct


def remove_byproduct_wo_graph(self, atoms : ase.Atoms, distance_matrix ) -> ase.Atoms:
    """Remove the byproduct from the atoms object without using graph networks

    Args:
        atoms (ase.Atoms): ase.Atoms object to remove the byproduct
        distance_matrix (_type_): bond cutoff criteria

    Returns:
        ase.Atoms: _description_

    Convert 0 value to infinite value in distance matrix
    Torch module to create networks accelerate the calculation
    Create graph networks which is the distance matrix is smaller than the cutoff
    Step 2: Generate edges from connectivity matrix
    Step 3: Create an undirected graph using NetworkX
    Step 4: Analyze connected components (clusters)

    """
    atomic_n = torch.tensor(atoms.get_atomic_numbers()).to(self.device)
    distance = torch.tensor(atoms.get_all_distances(mic=True)).to(self.device)
    distance.fill_diagonal_(float('inf'))
    distance_matrix_at_n = distance_matrix[atomic_n].to(self.device)
    cutoff = (distance_matrix_at_n[:, None] + distance_matrix_at_n[None, :])
    is_connected = distance < cutoff
    edges = torch.nonzero(is_connected, as_tuple=False).tolist()  # Convert to list of pairs
    graph = nx.Graph()
    graph.add_edges_from(edges)
    cluster_list = list(nx.connected_components(graph))  # List of sets of connected nodes
    oatom = self.remove_byproduct(atoms, graph, cluster_list)
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
    rm_byproduct = ByProduct(elmlist, byproduct_list)

    distance_matrix  = torch.tensor(covalent_radii).to(rm_byproduct.device) * 1.3 # Scaling
    oatom = rm_byproduct.remove_byproduct_wo_graph(ase.io.read('/data2/gasplant63/etch_gnn/7_SiNOCHF/15_small_cell_etch/0_codes/etc/POSCAR'), distance_matrix)
    ase.io.write('/data2/gasplant63/etch_gnn/7_SiNOCHF/15_small_cell_etch/0_codes/etc/POSCAR_wo_byproduct', oatom, format='vasp')
