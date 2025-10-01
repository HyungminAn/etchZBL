import os
import numpy as np
from ase.io import write
from ase.constraints import FixAtoms
import pickle

from utils import PARAMS, generate_bondlength_dict, GraphAnalyzer


class BulkGasIdentifier:
    def __init__(self):
        self.bond_length_dict = generate_bondlength_dict()

    def run(self):
        print("Running BulkIdentifier...")
        images = self.load_images()
        only_bulks, all_gases = self.partition_images(images)
        unique_bulks = self.remove_duplicate(only_bulks)
        unique_gases = self.remove_duplicate_gases(all_gases)
        self.save(unique_bulks, unique_gases)
        print("Done: bulks and gases saved.")

    def load_images(self):
        with open(PARAMS.path_to_rlx, 'r') as f:
            lines = f.readlines()[1:]

        keys = []
        for line in lines:
            incidence, *img_indices = map(int, line.split())
            for img_idx in img_indices:
                keys.append((incidence, img_idx))

        with open(PARAMS.path_nnp_pickle, 'rb') as f:
            images = pickle.load(f)

        result = {}
        for key, image in zip(keys, images):
            result[key] = image
            image.set_pbc((True, True, False))
        return result

    def partition_images(self, images):
        """
        For each MD snapshot, run graph analysis:
          - Identify the main slab cluster → bulk
          - Any other cluster that passes gas criteria → gas
        """
        only_bulks = {}
        all_gases  = {}
        ga = GraphAnalyzer()

        for key, img in images.items():
            _graph, _vprop, cluster, hist = ga.run_single(img)
            slab_idx = np.argmax(hist)
            # collect bulk
            bulk_height = np.max(img.positions[cluster.a == slab_idx, 2])

            n_atoms_total = len(img)
            n_bulk_atoms = 0
            bulk_atom_indices = []

            # identify gas‐clusters
            for cidx in set(cluster.a):
                atoms_idx = np.argwhere(cluster.a == cidx).flatten()
                avg_z = img.positions[atoms_idx,2].mean()
                if cidx == slab_idx:
                    bulk_atom_indices.extend(atoms_idx)
                    n_bulk_atoms += len(atoms_idx)
                    n_atoms_total -= n_bulk_atoms
                else:
                    n_gas_atoms = len(atoms_idx)
                    incidence, image_idx = key
                    if avg_z > bulk_height + PARAMS.gas_crit:
                        gas_atoms = self.copy_image_with_selected_atoms(img, atoms_idx)
                        all_gases[(incidence, image_idx, cidx)] = gas_atoms
                    else:
                        bulk_atom_indices.extend(atoms_idx)
                        n_bulk_atoms += n_gas_atoms
                    n_atoms_total -= n_gas_atoms

            if n_atoms_total > 0:
                raise ValueError("Some atoms were not classified as bulk or gas.")
            bulk_atoms = self.copy_image_with_selected_atoms(img, bulk_atom_indices)
            only_bulks[key] = bulk_atoms

        return only_bulks, all_gases

    def copy_image_with_selected_atoms(self, image, selected_atoms):
        result = image.copy()
        while result:
            result.pop()
        for idx, atom in enumerate(image):
            if idx in selected_atoms:
                result.append(atom)
        return result

    def remove_duplicate(self, only_bulks):
        '''
        remove duplicated bulk configuration
        '''
        ga = GraphAnalyzer()
        graphs, *_ = ga.run_batch([i for i in only_bulks.values()])

        to_delete_image_idx=[]
        for graph_idx, (graph_1, graph_2) in enumerate(zip(graphs[:-1], graphs[1:]), start=1):
            if ga.is_two_graph_same(graph_1, graph_2):
                to_delete_image_idx.append(graph_idx)

        n_total = len(only_bulks)
        n_to_delete = len(to_delete_image_idx)
        print(f"Found {n_to_delete} duplicate bulks out of {n_total} total images.")

        unique_bulks = {}
        for idx, (key, image) in enumerate(only_bulks.items()):
            if idx not in to_delete_image_idx:
                unique_bulks[key] = image

        return unique_bulks

    def remove_duplicate_gases(self, all_gases):
        """
        all_gases is a list (over images) of lists of Atoms objects.
        Flatten them, then dedupe by composition signature:
          composition = sorted tuple of (symbol, count).
        Return:
          - unique_gases: list of Atoms
          - unique_refs: list of (image_idx, gas_cluster_idx) tuples
        """
        seen = {}
        unique_gases = {}
        for key, gas in all_gases.items():
            symbols, counts = np.unique(gas.get_chemical_symbols(), return_counts=True)
            sig = tuple(sorted(zip(symbols, counts)))
            name = "".join(f"{s}{c}" for s, c in sig)
            if sig not in seen:
                seen[sig] = True
                unique_gases[key] = (gas, name)
        return unique_gases

    def save(self, unique_bulks, unique_gases):
        """
        Save bulks exactly as before, then save gases under a new root.
        unique_bulk_idx[i] = (incidence, snapshot) for bulk i,
        unique_gas_refs[i]  = (img_i, gas_cluster_idx).
        """
        # 1) bulk saving (same as you already have)…
        dst = PARAMS.path_post_process
        os.makedirs(dst,exist_ok=True)
        fix_h = PARAMS.fix_h

        for (incidence, snapshot_idx), image in unique_bulks.items():
            coo_path = f"{dst}/{incidence}/{incidence}_{snapshot_idx}"
            if not os.path.exists(f'{coo_path}/coo'):
                os.makedirs(coo_path,exist_ok=True)
                write(f"{coo_path}/coo", image, **PARAMS.SYSTEM_DEPENDENT.LAMMPS_SAVE_OPTS)
            else:
                print(f"Skipping {coo_path} as it already exists.")
                continue

            image_vasp = image.copy()
            s_flag = FixAtoms(indices=[atom.index for atom in image_vasp if atom.position[2] < fix_h])
            image_vasp.set_constraint(s_flag)

            if not os.path.exists(f"{coo_path}/POSCAR"):
                write(f"{coo_path}/POSCAR", image_vasp, **PARAMS.VASP_SAVE_OPTS)
            else:
                print(f"Skipping {coo_path}/POSCAR as it already exists.")
                continue

        with open(PARAMS.path_unique_bulk, 'w') as f:
            f.write("#incidence,snapshot_idx\n")
            for incidence, image_idx in unique_bulks.keys():
                f.write(f"{incidence},{image_idx}\n")

        new_coos=[i.copy() for i in unique_bulks.values()]
        write(PARAMS.path_unique_bulk_extxyz, new_coos, format='extxyz')

        # ------------------------------------------------------

        # 2) gas saving
        gas_root = os.path.join(PARAMS.path_post_process, "gases")
        for (image, name) in unique_gases.values():
            dst = os.path.join(gas_root, name)
            os.makedirs(dst, exist_ok=True)
            if not os.path.exists(os.path.join(dst, "coo")):
                write(os.path.join(dst, "coo"), image, **PARAMS.SYSTEM_DEPENDENT.LAMMPS_SAVE_OPTS)
            else:
                print(f"Skipping {dst}/coo as it already exists.")
                continue

        # 3) write index files
        with open(PARAMS.path_unique_bulk, 'w') as f:
            f.write("#incidence,snapshot_idx\n")
            for inc, snap in unique_bulks.keys():
                f.write(f"{inc},{snap}\n")

        with open(PARAMS.path_unique_gas, 'w') as f:
            f.write("#incidence/snapshot_idx/gas_cluster/elem_count\n")
            for (incidence, snapshot_idx, cluster_idx), (gas, _) in unique_gases.items():
                line = f"{incidence}/{snapshot_idx}/{cluster_idx}/"
                atom_dict = {v: k for k,v in PARAMS.SYSTEM_DEPENDENT.atom_idx.items()}

                composition=np.zeros((len(atom_dict)), dtype=int)
                for atom in gas:
                    composition[atom_dict[atom.symbol]] += 1
                line += ",".join(map(str, composition)) + '\n'
                f.write(line)
