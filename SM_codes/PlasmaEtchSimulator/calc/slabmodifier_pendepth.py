import os
import shutil
import copy

import numpy as np
import torch

import ase
import ase.io
import ase.build

from PlasmaEtchSimulator.calc.functions import init_out
from PlasmaEtchSimulator.calc.functions import save_atom
from PlasmaEtchSimulator.calc.functions import load_atom
from PlasmaEtchSimulator.calc.graphbuilder import GraphBuilder


class SlabModifier:
    @staticmethod
    def run(calc) -> None:
        """Append the atoms from bulk of the calculation
        """
        ## If already modified structure exists, pass
        if calc.calc_done():
            return

        if calc.str_out == None:
            calc.str_out = init_out(calc.box_height, calc.str_in)

        saved_file = 'save_'+calc.fname_in(calc.iteration-calc.run_params[calc.KEY.SLAB_UPDATE_ITERATION])

        n_atoms_limit = calc.n_atoms_limit
        h_CHF_crit = calc.h_CHF_crit
        CHF_ratio_crit = calc.CHF_ratio_crit
        h_penetrate_crit = calc.h_penetrate_crit
        h_sub_amount = calc.h_sub_amount
        h_add_amount = calc.h_add_amount

        CHF_ratio = SlabModifier._get_CHF_ratio(calc, h_CHF_crit)
        n_atoms = len(calc.str_in)
        if CHF_ratio > CHF_ratio_crit:  # Check for subtract
            line = f"CHF ratio {CHF_ratio:.3f} > {CHF_ratio_crit:.3f} "
            if n_atoms > n_atoms_limit:
                line += f"and n_atoms {n_atoms} > {n_atoms_limit}"
                calc.logger.print(line)
                SlabModifier._subtract_slab(calc, h_sub_amount, saved_file)
            else:
                line += f"but n_atoms {n_atoms} < {n_atoms_limit}; skip the slab modification"
                calc.logger.print(line)
            is_anneal_needed = False

        elif SlabModifier.is_penetrated(calc, h_penetrate_crit):  # Check for add
            line = ( "Slab is penetrated: ")
            calc.logger.print(line)
            if n_atoms > n_atoms_limit:
                raise NatomsExceedError
            SlabModifier._add_slab(calc, h_add_amount, saved_file)
            is_anneal_needed = True

        else:
            calc.logger.print(f"No slab modification is required")
            if os.path.exists(saved_file):
                calc.logger.print(f"Transfer the saved file {saved_file} to {'save_'+calc.name_in}")
                shutil.copy(saved_file, 'save_'+calc.name_in)
            is_anneal_needed = False
        return is_anneal_needed

    @staticmethod
    def _get_CHF_ratio(calc, h_crit):
        '''
        Get the ratio of CHF atoms in the structure
        whose height is higher than h_crit
        '''
        structure = calc.str_in
        pos = structure.get_positions()[:, 2]
        symbols = structure.get_chemical_symbols()
        idxes = np.arange(len(pos))
        mask_height = idxes[pos > h_crit]

        CHF_list = ['C', 'H', 'F']
        mask_symbol = idxes[np.isin(symbols, CHF_list)]

        mask_CHF = np.intersect1d(mask_height, mask_symbol)
        CHF_ratio = len(mask_CHF) / len(mask_height)
        return CHF_ratio

    @staticmethod
    def is_penetrated(calc, h_check_upper):
        '''
        Check if the structure is penetrated
        '''
        structure = calc.str_in
        pos = structure.get_positions()[:, 2]
        idxes = np.arange(len(pos))
        mask_height = idxes[pos < h_check_upper]
        symbols = structure.get_chemical_symbols()
        mask_CHF = idxes[np.isin(symbols, ['C', 'H', 'F'])]
        return len(np.intersect1d(mask_height, mask_CHF)) > 0

    @staticmethod
    def _get_position_in_PBC(atoms :ase.Atoms):
        cell = atoms.cell
        positions = atoms.positions
        for i in range(3):
            positions[:,i] = positions[:,i] % cell[i,i]
        return positions

    @staticmethod
    def _add_slab_from_saved(calc, h_add_amount, saved_file):
        '''
        Update the slab height by adding the atoms from the bulk to self.str_out
        Fetch information
        '''
        calc.logger.print(f"Adding bulk to the slab structure from the saved file {saved_file}")
        saved_atom = load_atom(saved_file, calc.convert_dict)
        h_saved = saved_atom.cell[2, 2]
        is_add_bulk_required = h_saved < h_add_amount
        if is_add_bulk_required:
            h_add_amount -= h_saved
            calc.str_out.positions[:,2] += h_saved
            for atom in saved_atom:
                calc.str_out.append(copy.deepcopy(atom))
        else:
            h_remain = h_saved - h_add_amount
            idx_to_append = (saved_atom.positions[:, 2] > h_remain)
            calc.str_out.positions[:,2] += h_add_amount
            for atom in saved_atom[idx_to_append]:
                atom = copy.deepcopy(atom)
                atom.position[2] -= h_remain
                calc.str_out.append(atom)
            del saved_atom[idx_to_append]
            saved_atom.cell[2, 2] -= h_add_amount

            path_save = 'save_'+calc.name_in
            save_atom(calc, path_save, saved_atom)

            calc.str_out = ase.build.sort(calc.str_out)
            path_save = 'add_'+calc.name_in
            save_atom(calc, path_save, calc.str_out)
            calc.logger.print(f"{path_save} written")

        return is_add_bulk_required, h_add_amount

    @staticmethod
    def _find_slab_h_origin(calc,
                            lowest_match=6,
                            err=5e-4):
        '''
        Find indices of matching atoms from upshift ~ fix layer from slab
        Select x,y position of the slab atoms : Select # lowest_match from lowest Z position one
        Find the matching atoms from the bulk
        This is the original points of the slab that the bulk atoms is append here
        Upshift slab atoms to the center of the slab
        slab_h_origin is the original height of the slab that the bulk atoms is append here
        To avoid duplicated atom (same z postiions of slab and bulk), small displacement is appliced
        '''
        pos_slab = torch.tensor(SlabModifier._get_position_in_PBC(calc.str_out)).to(calc.device)
        fix      = calc.fix
        fix_idx = torch.nonzero((pos_slab[:,2] < fix)).flatten()
        sel_slab = pos_slab[fix_idx]
        sel_pos = sel_slab[sel_slab[:,2].argsort()][:lowest_match]
        index, slab_h_origin = None, None

        pos_bulk = torch.tensor(SlabModifier._get_position_in_PBC(calc.atom_bulk)).to(calc.device)
        for idx in range(lowest_match):
            print(sel_pos[idx])
            is_atom = torch.all(torch.abs(pos_bulk[:,:2] - sel_pos[:,:2][idx]) < err, dim=1)
            if is_atom.any():
                index           = torch.nonzero(is_atom).flatten()
                slab_h_origin = sel_pos[idx,2]
                break

        if slab_h_origin is None:
            raise AtomNotMatchError

        return slab_h_origin, pos_bulk, index

    @staticmethod
    def _put_bulk_atoms_into_slab(calc,
                                  slab_h_origin,
                                  h_bulk_add,
                                  pos_bulk,
                                  index,
                                  err):
        '''
        Add atoms to the slab within PBC cell (1st addition),
        the about h_bulk_add is added to the slab
        '''
        calc.str_out.positions[:, 2] += h_bulk_add
        slab_h_shift = slab_h_origin + h_bulk_add
        bulk_h_origin = pos_bulk[index, 2]

        h_slice_upper = bulk_h_origin - err
        h_slice_lower = bulk_h_origin - h_bulk_add - err
        idx_append_in_cell = torch.nonzero((pos_bulk[:,2] < h_slice_upper) &
                                           (pos_bulk[:,2] > h_slice_lower)).flatten()
        for idx in idx_append_in_cell:
            atom = copy.deepcopy(calc.atom_bulk[int(idx)])
            atom.position[2] += (slab_h_shift - bulk_h_origin)
            calc.str_out.append(atom)

        '''
        Check the addition over the PBC cell is required :
            If the addition is required over the PBC cell, add the atoms to the slab
            (2nd+ addition)
        '''
        bulk_z_cell   = calc.atom_bulk.cell[2, 2]
        height_res_to_add = h_bulk_add - bulk_h_origin
        PBC_need = height_res_to_add > 0 # If the addition is required over the PBC cell

        slab_height = slab_h_origin + height_res_to_add
        while PBC_need:
            n_bulkcell_to_insert  = height_res_to_add // bulk_z_cell
            if n_bulkcell_to_insert > 0:
                for idx in range(len(calc.atom_bulk)):
                    atom = copy.deepcopy(calc.atom_bulk[idx])
                    atom.position[2] += (slab_height - bulk_z_cell)
                    calc.str_out.append(atom)
                slab_height         -= bulk_z_cell
                height_res_to_add   -= bulk_z_cell

            else:
                idx_append_res = torch.nonzero((pos_bulk[:,2] > bulk_z_cell - height_res_to_add )).flatten()
                for idx in idx_append_res:
                    atom = copy.deepcopy(calc.atom_bulk[int(idx)])
                    atom.position[2] += (slab_height - bulk_z_cell)
                    calc.str_out.append(atom)
                break

        calc.str_out = ase.build.sort(calc.str_out)
        path_save = 'add_'+calc.name_in
        save_atom(calc, path_save, calc.str_out)
        calc.logger.print(f"{path_save} written")

    @staticmethod
    def _add_slab(calc, h_add_amount, saved_file):
        lowest_match = 6  # At least (lowest_math) atoms should be matched
        err  = 5e-4
        slab_h_origin, pos_bulk, index =\
                SlabModifier._find_slab_h_origin(calc,
                                                 lowest_match,
                                                 err)

        is_add_bulk_required = True
        if os.path.exists(saved_file):
            is_add_bulk_required, h_add_amount = SlabModifier._add_slab_from_saved(calc, h_add_amount, saved_file)

        if is_add_bulk_required:
            SlabModifier._put_bulk_atoms_into_slab(calc, slab_h_origin, h_add_amount, pos_bulk, index, err)

    @staticmethod
    def _subtract_slab(calc, h_sub_amount, saved_file) -> ase.Atoms:
        """
        Extract the atoms by *h_sub_amount* from the slab
        Remove all of the atoms which substracted from the slab
        """
        print(f"Slab to substract : h < {h_sub_amount}")
        sub_atoms = copy.deepcopy(calc.str_out)
        idx_to_remove = sub_atoms.positions[:,2] > h_sub_amount
        del sub_atoms[idx_to_remove]
        if os.path.exists(saved_file):
            calc.logger.print(f"The file {saved_file} exists, append the atoms from the file")
            subs_to_add         = load_atom(saved_file, calc.convert_dict)
            atom_append_h       = subs_to_add.cell[2,2]
            pos_before          = subs_to_add.positions
            subs_to_add.cell[2,2] += h_sub_amount
            subs_to_add.positions = pos_before
        else:
            subs_to_add      = ase.Atoms(cell=sub_atoms.cell)
            atom_append_h    = 0
            subs_to_add.cell[2,2] = h_sub_amount

        for atom in sub_atoms:
            atom.position[2] += atom_append_h
            subs_to_add.append(copy.deepcopy(atom))

        calc.str_out                    = copy.deepcopy(calc.str_out[idx_to_remove])
        calc.str_out.positions[:,2]     -= h_sub_amount
        calc.str_out = ase.build.sort(calc.str_out)
        path_save = 'sub_'+calc.name_in
        save_atom(calc, path_save, calc.str_out)

        path_save = 'save_'+calc.name_in
        save_atom(calc, path_save, subs_to_add)
        calc.logger.print(f"{path_save} written")

    @staticmethod
    def get_slab_height(calc, ctype = 'in') -> float:
        """Get the height of the slab
        """
        if ctype == 'in':
            GraphBuilder.graph_call_in(calc)
            clusters = calc.clusters_in
            structure = calc.str_in
        elif ctype == 'out':
            GraphBuilder.graph_call_out(calc)
            clusters = calc.clusters_out
            structure = calc.str_out
        largest_cluster = max(clusters, key=len)
        largest_cluster_indices = list(largest_cluster)
        slab_z_pos = structure.positions[largest_cluster_indices, 2]
        slab_height = np.percentile(slab_z_pos, 95)
        return slab_height


class AtomNotMatchError(Exception):
    def __init__(self):
        print("Error occured : the atoms of slab is not in bulk, check the files")

class NatomsExceedError(Exception):
    def __init__(self):
        line = (
                "Error: slab_add is needed, but the number of atoms exceeds the limit.\n"
                "Please adjust n_atoms_limit or penetration height"
                )
        print(line)
