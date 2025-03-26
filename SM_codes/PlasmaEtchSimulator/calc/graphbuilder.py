# import torch
import numpy as np
import networkx as nx
from ase.geometry import get_distances


class GraphBuilder:
    @staticmethod
    def graph_call_in(calc) -> None:
        structure = calc.str_in
        distance_matrix = calc.distance_matrix

        edges = GraphBuilder.find_edges(structure, distance_matrix)

        calc.G_in = nx.Graph()
        calc.G_in.add_edges_from(edges)
        calc.clusters_in = list(nx.connected_components(calc.G_in))

    @staticmethod
    def graph_call_out(calc) -> None:
        structure = calc.str_out
        distance_matrix = calc.distance_matrix

        edges = GraphBuilder.find_edges(structure, distance_matrix)

        calc.G_out = nx.Graph()
        calc.G_out.add_edges_from(edges)
        calc.clusters_out = list(nx.connected_components(calc.G_out))  # List of sets of connected nodes

    @staticmethod
    def _split_into_bins(pos, cell, dist_matrix):
        # Define bin sizes for each axis based on cutoff distances
        slice_crit_x = slice_crit_y = slice_crit_z = np.max(dist_matrix.flatten())
        cell_x, cell_y, _ = cell.diagonal()

        # Determine bins for each axis
        x_bins = np.arange(0, cell_x, slice_crit_x)[:-1]
        y_bins = np.arange(0, cell_y, slice_crit_y)[:-1]
        z_bins = np.arange(0, np.max(pos[:, 2]) + slice_crit_z, slice_crit_z)

        # Digitize positions into bins
        x_indices = np.digitize(pos[:, 0], x_bins)
        y_indices = np.digitize(pos[:, 1], y_bins)
        z_indices = np.digitize(pos[:, 2], z_bins)

        # Combine x, y, z indices into unique 3D bin keys
        bin_keys = list(zip(x_indices, y_indices, z_indices))

        # Create a dictionary mapping each bin to atom indices
        bin_dict = {}
        for i, key in enumerate(bin_keys):
            if key not in bin_dict:
                bin_dict[key] = []
            bin_dict[key].append(i)
        return (x_bins, y_bins, z_bins), bin_dict

    @staticmethod
    def find_edges(image, dist_matrix):
        dist_matrix = dist_matrix.detach().cpu().numpy()
        pos = image.get_positions()
        cell = image.get_cell()
        bins, bin_dict = GraphBuilder._split_into_bins(pos, cell, dist_matrix)

        atom_types = image.get_array('type')-1

        # Initialize a list to store connectivity edges
        all_edges = []

        # Iterate over bins and calculate connectivity within and between neighboring bins
        x_bins, y_bins, z_bins = bins
        for current_bin in bin_dict:
            current_bin_indices = bin_dict[current_bin]

            # Get neighboring bins (3D neighborhood)
            neighboring_bins = [
                (
                    ((current_bin[0] + dx - 1) % len(x_bins)) + 1,
                    ((current_bin[1] + dy - 1) % len(y_bins)) + 1,
                    ((current_bin[2] + dz - 1) % len(z_bins)) + 1
                )
                for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
            ]
            neighboring_bins = [b for b in neighboring_bins if b in bin_dict]

            # Get atom indices for neighboring bins
            neighboring_bin_indices = [
                atom_idx for b in neighboring_bins for atom_idx in bin_dict[b]
            ]

            # Calculate pairwise distances between atoms in the current and neighboring bins
            if not neighboring_bin_indices:
                continue

            pos_current = pos[current_bin_indices]
            pos_neighboring = pos[neighboring_bin_indices]

            # Calculate distances with minimum image convention (mic=True)
            _, D_len = get_distances(pos_current, p2=pos_neighboring, cell=cell, pbc=True)

            # Calculate cutoff distances for each pair
            atom_types_current = atom_types[current_bin_indices]
            atom_types_neighboring = atom_types[neighboring_bin_indices]
            cutoff_distances = dist_matrix[atom_types_current][:, atom_types_neighboring]

            # Identify pairs within the cutoff distance
            row, col = np.where(D_len < cutoff_distances)

            # Map local indices back to global indices and add to all_edges
            for r, c in zip(row, col):
                all_edges.append((current_bin_indices[r], neighboring_bin_indices[c]))

        return all_edges

    # @staticmethod
    # def find_edges(structure, device, distance_matrix) -> list:
    #     '''
    #     Convert 0 value to infinite value in distance matrix
    #     Torch module to create networks accelerate the calculation
    #     Create graph networks which is the distance matrix is smaller than the cutoff

    #     Step 2: Generate edges from connectivity matrix
    #     Step 3: Create an undirected graph using NetworkX
    #     Step 4: Analyze connected components (clusters)
    #     '''
    #     # atomic_n = torch.tensor(structure.get_atomic_numbers()).to(device)
    #     atomic_n = (torch.tensor(structure.get_array('type'))-1).to(device)
    #     distance = torch.tensor(structure.get_all_distances(mic=True)).to(device)
    #     distance.fill_diagonal_(float('inf'))

    #     # Ensure distance_matrix is on the same device as atomic_n
    #     distance_matrix = distance_matrix.to(device)

    #     cutoff = distance_matrix[atomic_n][:, atomic_n]
    #     # cutoff = (distance_matrix_at_n[:, None] + distance_matrix_at_n[None, :])
    #     is_connected = distance < cutoff
    #     edges = torch.nonzero(is_connected, as_tuple=False).tolist()  # Convert to list of pairs
    #     return edges
