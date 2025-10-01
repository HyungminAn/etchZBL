import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import pandas as pd

from utils import timeit


def nested_dict():
    return defaultdict(nested_dict)


class AtomInfo:
    """
    This class manages each atom's global index (global_idx), atom type,
    creation/deletion timestamps, and provides the mapping between
    (struct_idx, file_type, local_idx) and global_idx.
    """
    # __slots__ = ['global_idx', 'type', 'timestamp_created', 'timestamp_removed']

    def __init__(self, global_idx, atom_type, timestamp_created):
        self.global_idx = global_idx
        self.type = atom_type
        self.timestamp_created = timestamp_created
        self.timestamp_removed = None
        print(f'AtomInfo: global_idx {self.global_idx} type {self.type} created at struct_idx {self.timestamp_created}')


class AtomDict:
    initialized = False

    def __init__(self):
        if not AtomDict.initialized:
            AtomDict.initialized = True
        else:
            raise ValueError('AtomDict is a singleton class')

        self.map_local_to_global = nested_dict()  # [struct_idx][file_type][local_idx] -> AtomInfo instance
        self.map_global_to_local = nested_dict()  # [global_idx][struct_idx][file_type] -> local_idx
        self.global_idx_count = 0
        self.coo_prev = None

    @timeit
    def update(self, coo_dict, struct_idx):
        coo = coo_dict['coo']
        if self.coo_prev is not None:
            self.check_final_to_new_initial(self.coo_prev, coo, struct_idx-1, struct_idx)
        else:
            self.add_atoms_init(coo, struct_idx)

        coo_rm = coo_dict['coo_rm']
        self.check_atom_removal_by_rmbyproduct(coo, coo_rm, struct_idx)

        coo_add = coo_dict['coo_add']
        coo_before_anneal = coo_dict['coo_before_anneal']
        coo_sub = coo_dict['coo_sub']
        coo_save = coo_dict['coo_save']

        is_added = coo_add is not None
        is_subtracted = coo_sub is not None

        if is_added:
            self.check_atom_add(coo_rm, coo_add, struct_idx)
        elif is_subtracted:
            self.check_atom_sub(coo_rm, coo_sub, coo_save, struct_idx)

        coo_final = coo_dict['coo_final']
        self.check_atom_final(coo_rm,
                                  coo_sub,
                                  coo_before_anneal,
                                  coo_final,
                                  struct_idx,
                                  is_added=is_added,
                                  is_subtracted=is_subtracted)

        self.coo_prev = coo_final
        print(f'--- Finished processing structure {struct_idx} ---')

    # ------------------------------------------------------------------------
    # Helper methods for repeated registration logic
    # ------------------------------------------------------------------------
    def _register_new_atom(self,
                           struct_idx,
                           file_type,
                           local_idx,
                           symbol,
                           timestamp_created):
        """
        Register a newly created atom (without a global index yet).
        Creates a new AtomInfo instance and updates map_local_to_global / map_global_to_local.

        Returns True if the atom was successfully registered. (False if failed)
        """
        if symbol != 'C':
            return False

        if local_idx is None:  # 'C' atoms was added, but removed during the process
            global_idx = self.global_idx_count
            new_atom = AtomInfo(global_idx, symbol, timestamp_created)
            self.global_idx_count += 1
            new_atom.timestamp_removed = struct_idx
            return True

        global_idx = self.global_idx_count
        new_atom = AtomInfo(global_idx, symbol, timestamp_created)
        self.map_local_to_global[struct_idx][file_type][local_idx] = new_atom
        self.map_global_to_local[global_idx][struct_idx][file_type] = local_idx
        self.global_idx_count += 1
        return True

    def _register_existing_atom(self,
                                struct_idx,
                                old_file_type,
                                old_local_idx,
                                new_file_type,
                                new_local_idx,
                                struct_idx_new=None):
        """
        Register an atom that already has a global index under a different
        file_type / local_idx mapping.
        """
        global_atom = self.map_local_to_global[struct_idx][old_file_type].get(old_local_idx)
        if global_atom is None:
            return

        global_idx = global_atom.global_idx
        if struct_idx_new is None:
            self.map_local_to_global[struct_idx][new_file_type][new_local_idx] = global_atom
            self.map_global_to_local[global_idx][struct_idx][new_file_type] = new_local_idx
        else:
            self.map_local_to_global[struct_idx_new][new_file_type][new_local_idx] = global_atom
            self.map_global_to_local[global_idx][struct_idx_new][new_file_type] = new_local_idx

    @staticmethod
    def build_periodic_kdtree(A, cell, pbc, super_range=1):
        """
        A: (N, 3) shape
        cell: (3, 3) shape
        pbc: (3,) (ex: (True, True, True))
        super_range: 1 --> -1, 0, +1
        """
        replicate_x = range(-super_range, super_range+1) if pbc[0] else [0]
        replicate_y = range(-super_range, super_range+1) if pbc[1] else [0]
        replicate_z = range(-super_range, super_range+1) if pbc[2] else [0]

        images = []
        for ix in replicate_x:
            for iy in replicate_y:
                for iz in replicate_z:
                    shift = ix * cell[0] + iy * cell[1] + iz * cell[2]
                    coords_shifted = A + shift
                    images.append(coords_shifted)

        all_coords = np.concatenate(images, axis=0)  # shape: (N * #images, 3)
        tree = cKDTree(all_coords)
        return tree, all_coords

    @staticmethod
    def find_matching_indices(A, B, z_shift, cell, pbc=(True, True, True)):
        """
        A, B: (N, 3) shape
        B를 z_shift만큼 이동 후, A의 PBC 이미지를 포함하는 KDTree에서
        각 B_i에 대한 최근접 A_j의 index를 구한다.
        """
        # apply z_shift to B
        B_shifted = B.copy()
        B_shifted[:, 2] += z_shift

        # generate KDTree
        tree, all_coords = AtomDict.build_periodic_kdtree(A, cell, pbc=pbc, super_range=1)
        distances, idx_kdtree = tree.query(B_shifted)

        nA = len(A)
        matched_indices_b2a = idx_kdtree % nA  # (length=len(B))

        matched_indices_a2b = [[] for _ in range(nA)]
        for b_idx, a_idx in enumerate(matched_indices_b2a):
            matched_indices_a2b[a_idx].append(b_idx)

        return matched_indices_b2a, matched_indices_a2b

    def _remove_atoms(self, indices, file_type, struct_idx):
        for local_idx in indices:
            global_atom = self.map_local_to_global[struct_idx][file_type].get(local_idx)
            if global_atom is None:
                continue

            if file_type == 'final':
                global_atom.timestamp_removed = struct_idx + 1
            else:
                global_atom.timestamp_removed = struct_idx

    # ------------------------------------------------------------------------
    # Methods below have been refactored to reduce duplication
    # ------------------------------------------------------------------------

    def add_atoms_init(self, coo, struct_idx, file_type='initial'):
        """
        Register all atoms in the initial .coo file.
        """
        for local_idx, atom in enumerate(coo):
            self._register_new_atom(struct_idx,
                                    file_type,
                                    local_idx,
                                    symbol=atom.symbol,
                                    timestamp_created=struct_idx)
        print(f'{struct_idx} : Added {len(coo)} atoms to initial state')

    def check_atom_removal_by_rmbyproduct(self, coo, coo_rm, struct_idx):
        """
        Determine which atoms were removed (rm_byproduct) and mark their timestamps.
        Also register them in 'rm_byproduct' mapping if applicable.
        """
        indices_B2A, indices_A2B = self.find_matching_indices(coo.get_positions(),
                                                             coo_rm.get_positions(),
                                                             0,
                                                             coo.get_cell())
        removed_indices = [i for i in range(len(coo)) if i not in indices_B2A]
        self._remove_atoms(removed_indices, 'initial', struct_idx)

        # Register the remaining atoms in 'rm_byproduct'
        for local_idx_rm, local_idx in enumerate(indices_B2A):
            self._register_existing_atom(struct_idx,
                                        'initial', local_idx,
                                        'rm_byproduct', local_idx_rm)

        count_removed = len(removed_indices)
        count_remaining = len(indices_B2A)
        if count_removed > 0:
            print(f'{struct_idx} : Removed {count_removed} atoms, Remaining: {count_remaining}')

    def check_atom_add(self,
                       coo_rm,
                       coo_add,
                       struct_idx,
                       file_type='add'):
        """
        Check how many atoms are newly added from coo_rm to coo_add,
        and register them. Then map all relevant atoms to 'before_anneal'.
        """
        n_atoms_added = len(coo_add) - len(coo_rm)
        pos_rm = coo_rm.get_positions()
        pos_add = coo_add.get_positions()
        indices_added = np.argsort(pos_add[:, 2])[:n_atoms_added]
        shift_z = np.max(pos_rm[:, 2]) - np.max(pos_add[:, 2])

        # Register newly added atoms
        for local_idx in indices_added:
            self._register_new_atom(struct_idx,
                                   file_type,
                                   local_idx,
                                   symbol=coo_add[local_idx].symbol,
                                   timestamp_created=struct_idx)
        print(f'{struct_idx} : Added {n_atoms_added} atoms to add state')

        indices_existing = np.array([i for i in range(len(coo_add)) if i not in indices_added])
        indices_B2A, indices_A2B = self.find_matching_indices(pos_rm,
                                                             pos_add[indices_existing],
                                                             shift_z,
                                                             coo_rm.get_cell())
        # Register the remaining atoms in 'rm_byproduct'
        for local_idx_rm, local_idx_add in zip(indices_B2A, indices_existing):
            self._register_existing_atom(struct_idx,
                                        'rm_byproduct', local_idx_rm,
                                        'add', local_idx_add)
        print(f'{struct_idx} : Mapped {len(indices_existing)} atoms from rm_byproduct to add state')

        # All atoms in coo_add also map to 'before_anneal'
        for local_idx, _ in enumerate(coo_add):
            self._register_existing_atom(struct_idx,
                                        'add', local_idx,
                                        'before_anneal', local_idx)

    def check_atom_sub(self,
                       coo_rm,
                       coo_sub,
                       coo_save,
                       struct_idx,
                       file_type='sub'):
        """
        Determine how many atoms are removed/substituted from coo_rm to coo_sub/coo_save,
        mark removed atoms, and update mappings.
        """
        # Mark some atoms as removed
        n_atoms_sub = len(coo_rm) - len(coo_sub)
        pos_rm = coo_rm.get_positions()
        indices_sub_to_save = np.argsort(pos_rm[:, 2])[:n_atoms_sub]
        self._remove_atoms(indices_sub_to_save, 'rm_byproduct', struct_idx)

        pos_save = coo_save.get_positions()
        indices_in_save = np.argsort(pos_save[:, 2])[::-1][:n_atoms_sub]
        z_shift = np.max(pos_rm[indices_sub_to_save][:, 2]) - np.max(pos_save[indices_in_save][:, 2])

        # Map atoms to 'save'
        indices_B2A, indices_A2B = self.find_matching_indices(pos_rm[indices_sub_to_save],
                                                             pos_save[indices_in_save],
                                                             z_shift,
                                                             coo_rm.get_cell())
        for local_idx_rm, local_idx_save in zip(indices_sub_to_save[indices_B2A], indices_in_save):
            self._register_existing_atom(struct_idx,
                                        'rm_byproduct',
                                        local_idx_rm,
                                        'save',
                                        local_idx_save)

        # Map atoms to 'sub'
        pos_sub = coo_sub.get_positions()
        z_shift = np.max(pos_rm[:, 2]) - np.max(pos_sub[:, 2])
        indices_rm_to_sub = np.argsort(pos_rm[:, 2])[n_atoms_sub:]
        indices_B2A, indices_A2B = self.find_matching_indices(pos_rm[indices_rm_to_sub],
                                                             pos_sub,
                                                             z_shift,
                                                             coo_rm.get_cell())
        for local_idx_rm, local_idx_sub in zip(indices_rm_to_sub[indices_B2A], range(len(pos_sub))):
            self._register_existing_atom(struct_idx,
                                        'rm_byproduct',
                                        local_idx_rm,
                                        'sub',
                                        local_idx_sub)

    def check_atom_final(self,
                         coo_rm,
                         coo_sub,
                         coo_before_anneal,
                         coo_final,
                         struct_idx,
                         file_type='final',
                         is_added=False,
                         is_subtracted=False):
        """
        Register atoms into 'final' based on previous steps.
        """
        if is_added:
            # 'before_anneal' -> 'final'
            atom_id = coo_before_anneal.get_array('id')
            atom_final_id = coo_final.get_array('id')
            for local_idx, local_id in enumerate(atom_final_id):
                old_local_idx = np.where(atom_id == local_id)[0][0]
                self._register_existing_atom(struct_idx,
                                            'before_anneal', old_local_idx,
                                            'final', local_idx)
        elif is_subtracted:
            # 'sub' -> 'final'
            for local_idx, _ in enumerate(coo_sub):
                self._register_existing_atom(struct_idx,
                                            'sub', local_idx,
                                            'final', local_idx)
        else:
            # 'rm_byproduct' -> 'final'
            for local_idx, _ in enumerate(coo_rm):
                self._register_existing_atom(struct_idx,
                                            'rm_byproduct', local_idx,
                                            'final', local_idx)

    def check_final_to_new_initial(self,
                                   coo_final,
                                   coo_initial_new,
                                   struct_idx,
                                   struct_idx_new):
        """
        When moving from the final state of one structure to the initial state of another,
        determine which atoms disappear and which are newly added, then update mappings.
        """
        atom_id = coo_final.get_array('id')
        atom_initial_new_id = coo_initial_new.get_array('id')

        # Mark atoms that are no longer present
        mask_deleted = ~np.isin(atom_id, atom_initial_new_id)
        indices_deleted = np.where(mask_deleted)[0]
        self._remove_atoms(indices_deleted, 'final', struct_idx)
        print(f'{struct_idx} -> {struct_idx_new}: Removed {len(indices_deleted)} atoms, Remaining: {len(atom_id) - len(indices_deleted)}')

        # Register newly added atoms
        mask_added = ~np.isin(atom_initial_new_id, atom_id)
        indices_added = np.where(mask_added)[0]
        is_C_remains = False
        for local_idx in indices_added:
            is_C_remains = is_C_remains or self._register_new_atom(struct_idx_new,
                                                               'initial',
                                                               local_idx,
                                                               symbol=coo_initial_new[local_idx].symbol,
                                                               timestamp_created=struct_idx_new)
        if not is_C_remains:
            self._register_new_atom(struct_idx_new,
                                    'initial',
                                    None,
                                    symbol='C',
                                    timestamp_created=struct_idx_new)

        print(f'{struct_idx} -> {struct_idx_new}: Added {len(indices_added)} atoms, Remaining: {len(atom_initial_new_id)}')

        # Map the atoms that remain from final to new initial
        mask_same = np.isin(atom_id, atom_initial_new_id)
        indices_same = np.where(mask_same)[0]
        for local_idx in indices_same:
            local_idx_new = np.where(atom_initial_new_id == atom_id[local_idx])[0][0]
            self._register_existing_atom(struct_idx,
                                        'final',
                                        local_idx,
                                        'initial',
                                        local_idx_new,
                                        struct_idx_new=struct_idx_new)

    def atomdict_to_dataframe(self, save_struct_idx):
        H5_SAVE_OPTS = {
            'key': 'atom_dict',
            'mode': 'w',
            'format': 'table',
            'complevel': 9,
            'complib': 'blosc:zstd',
            'index': False,
        }
        records = []

        for struct_idx, filetype_dict in self.map_local_to_global.items():
            for file_type, local_dict in filetype_dict.items():
                for local_idx, atom in local_dict.items():
                    records.append({
                        'struct_idx': struct_idx,
                        'file_type': file_type,
                        'local_idx': local_idx,
                        'global_idx': atom.global_idx,
                        'type': atom.type,
                        'timestamp_created': atom.timestamp_created,
                        'timestamp_removed': atom.timestamp_removed
                    })

        df = pd.DataFrame(records)
        df.to_hdf(f'atom_dict_{save_struct_idx}.h5', **H5_SAVE_OPTS)
