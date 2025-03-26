import os, sys
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
        calc.slab_height = SlabModifier.get_slab_height(calc, ctype = 'in')
        ## If condition is met, add or substract slab
        h_min = calc.slab_min_height
        h_max = calc.slab_max_height
        h = calc.slab_height
        # Case 1 : Add bulk to the slab structure without any additional structure
        case1 = h < h_min and not os.path.exists(saved_file)
        # Case 2 : Add bulk to the slab structure from the saved file & if the  saved file is not enough, add the atoms from the bulk
        case2 = h < h_min and os.path.exists(saved_file)
        # Case 3 : Substract the atoms from the slab structure to fit the slab height to the slab_center_height
        case3 = h > h_max
        if case1 or case2:
            calc.logger.print(f"Adding bulk to the slab structure: h ({h:.2f} A) < h_min ({h_min} A)")
        elif case3:
            calc.logger.print(f"Substracting bulk to the slab structure: h ({h:.2f} A) > h_max ({h_max} A)")
        else:
            calc.logger.print(f"Slab height is in the range:   h_min ({h_min} A) < h ({h:.2f} A) < h_max ({h_max} A)")
        if case1:
            SlabModifier._add_slab(calc)
            calc.str_out = ase.build.sort(calc.str_out)
            path_save = 'add_'+calc.name_in
            save_atom(calc, path_save, calc.str_out)
            calc.logger.print(f"{path_save} written")
            is_anneal_needed = True
        elif case2:
            required_add_bulk, sub_atoms =\
                SlabModifier._add_slab_from_saved(calc)
            path_save = 'add_'+calc.name_in
            save_atom(calc, path_save, calc.str_out)
            if required_add_bulk:
                calc.slab_height = SlabModifier.get_slab_height(calc, ctype='out')
                ## If saved file is not enough, add the atoms from the bulk
                SlabModifier._add_slab(calc)
                calc.str_out = ase.build.sort(calc.str_out)
                ## the structure is saved as lammps-data format which is the default format of ase
            else:
                path_save = 'save_'+calc.name_in
                save_atom(calc, path_save, sub_atoms)
            path_save = 'add_'+calc.name_in
            save_atom(calc, path_save, calc.str_out)
            calc.logger.print(f"{path_save} written")
            is_anneal_needed = True

        elif case3:
            sub_atoms = SlabModifier._substract_slab(calc)
            calc.str_out = ase.build.sort(calc.str_out)
            path_save = 'sub_'+calc.name_in
            save_atom(calc, path_save, calc.str_out)
            path_save = 'save_'+calc.name_in
            save_atom(calc, path_save, sub_atoms)
            calc.logger.print(f"{path_save} written")
            is_anneal_needed = False

        else:
            if os.path.exists(saved_file):
                calc.logger.print(f"Transfer the saved file {saved_file} to {'save_'+calc.name_in}")
                shutil.copy(saved_file, 'save_'+calc.name_in)
            is_anneal_needed = False

        return is_anneal_needed

    @staticmethod
    def _get_position_in_PBC(atoms :ase.Atoms):
        cell = atoms.cell
        positions = atoms.positions
        for i in range(3):
            positions[:,i] = positions[:,i] % cell[i,i]
        return positions

    @staticmethod
    def _add_slab_from_saved(calc):
        '''
        Update the slab height by adding the atoms from the bulk to self.str_out
        Fetch information
        '''
        saved_file = 'save_'+calc.fname_in(calc.iteration-calc.run_params[calc.KEY.SLAB_UPDATE_ITERATION])
        calc.logger.print(f"Adding bulk to the slab structure from the saved file {saved_file}")
        saved_atom = load_atom(saved_file, calc.convert_dict)
        h_saved = saved_atom.cell[2,2]
        h_slab   = SlabModifier.get_slab_height(calc)
        h_center = calc.slab_center_height
        h_bulk_add = h_center - h_slab
        if h_saved < h_bulk_add:
            required_add_bulk = True
            h_bulk_add -= h_saved
        else:
            required_add_bulk = False
        if required_add_bulk:
            calc.str_out.positions[:,2] += h_saved
            for atom in saved_atom:
                calc.str_out.append(copy.deepcopy(atom))
        else:
            append_saved_atom_index = (saved_atom.positions[:,2] > h_saved - h_bulk_add)
            calc.str_out.positions[:,2] += h_bulk_add
            for atom in saved_atom[append_saved_atom_index]:
                atom = copy.deepcopy(atom)
                atom.position[2] -= (h_saved - h_bulk_add)
                calc.str_out.append(atom)
            del saved_atom[append_saved_atom_index]
            saved_atom.cell[2,2] -= h_bulk_add
        calc.str_out = ase.build.sort(calc.str_out)
        return required_add_bulk, saved_atom

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
        slab_h_origin = None

        pos_bulk = torch.tensor(SlabModifier._get_position_in_PBC(calc.atom_bulk)).to(calc.device)
        for idx in range(lowest_match):
            print(sel_pos[idx])
            is_atom = torch.all(torch.abs(pos_bulk[:,:2] - sel_pos[:,:2][idx]) < err, dim=1)
            if is_atom.any():
                index           = torch.nonzero(is_atom).flatten()
                slab_h_origin = sel_pos[idx,2]
                break

        assert slab_h_origin != None , "Error occured : the atoms of slab is not in bulk, check the files"

        return slab_h_origin, pos_bulk, index

    @staticmethod
    def _put_bulk_atoms_into_slab(calc,
                                  slab_h_origin,
                                  pos_bulk,
                                  index,
                                  err):
        '''
        Add atoms to the slab within PBC cell (1st addition), the aboult h_bulk_add is added to the slab
        '''
        h_slab   = calc.slab_height
        h_center = calc.slab_center_height
        h_bulk_add = h_center - h_slab
        calc.str_out.positions[:, 2] += h_bulk_add
        slab_h_shift = slab_h_origin + h_bulk_add
        bulk_h_origin = pos_bulk[index, 2]

        idx_append_in_cell = torch.nonzero((pos_bulk[:,2] < bulk_h_origin - err) & (pos_bulk[:,2] > bulk_h_origin - h_bulk_add - err)).flatten()
        for idx in idx_append_in_cell:
            atom = copy.deepcopy(calc.atom_bulk[int(idx)])
            atom.position[2] += (slab_h_shift -bulk_h_origin)
            calc.str_out.append(atom)

        return h_bulk_add, bulk_h_origin
    @staticmethod
    def _put_bulk_atoms_into_slab_overPBC(calc,
                                          slab_h_origin,
                                          pos_bulk,
                                          h_bulk_add,
                                          bulk_h_origin):
        '''
        Check the addition over the PBC cell is required : if the selected atoms are at low Z position taht
        This is the residual Z positions to be append which updated after each addition
        The height of lowest atom after 1st addition
        The slab addition if the addition is required over the PBC cell
        If the addition is too large that the all cell of atoms is required, add all the atoms to the slab at once
        Add the atoms to the slab within PBC cell
        '''
        bulk_z_cell   = calc.atom_bulk.cell[2, 2]
        height_res_to_add = h_bulk_add - bulk_h_origin
        PBC_need = height_res_to_add > 0 # If the addition is required over the PBC cell
        if not PBC_need:
            return

        slab_height = slab_h_origin + h_bulk_add - bulk_h_origin
        while PBC_need:
            PBC_q  = height_res_to_add // bulk_z_cell
            # PBC_r = height_res_to_add % bulk_z_cell
            if PBC_q > 0: # All of Bulk atoms are added to the slab
                for idx in range(len(calc.atom_bulk)):
                    atom = copy.deepcopy(calc.atom_bulk[idx])
                    atom.position[2] += (slab_height - bulk_z_cell)
                    calc.str_out.append(atom)
                slab_height         -= bulk_z_cell
                height_res_to_add   -=  bulk_z_cell
            elif PBC_q == 0:
                idx_append_res = torch.nonzero((pos_bulk[:,2] > bulk_z_cell - height_res_to_add )).flatten()
                for idx in idx_append_res:
                    atom = copy.deepcopy(calc.atom_bulk[int(idx)])
                    atom.position[2] += (slab_height - bulk_z_cell)
                    calc.str_out.append(atom)
                break

    @staticmethod
    def _add_slab(calc):
        lowest_match = 6  # At least (lowest_math) atoms should be matched
        err  = 5e-4
        slab_h_origin, pos_bulk, index =\
                SlabModifier._find_slab_h_origin(calc,
                                                 lowest_match,
                                                 err)

        h_bulk_add, bulk_h_origin =\
                SlabModifier._put_bulk_atoms_into_slab(calc,
                                                       slab_h_origin,
                                                       pos_bulk,
                                                       index,
                                                       err)
        SlabModifier._put_bulk_atoms_into_slab_overPBC(calc,
                                                       slab_h_origin,
                                                       pos_bulk,
                                                       h_bulk_add,
                                                       bulk_h_origin)


    @staticmethod
    def _substract_slab(calc) -> ase.Atoms:
        """
        Extract the atoms from the slab to fit the slab height to the slab_center_height
        Remove all of the atoms which substracted from the slab
        h > h_crit : saved into file with prefix 'sub_'
        h < h_crit : saved into file with prefix 'save_'
        """
        h_crit = calc.slab_height - calc.slab_center_height
        print(f"Slab to substract : h < {h_crit:.2f}")
        sub_atoms = copy.deepcopy(calc.str_out)
        idx_to_remove = sub_atoms.positions[:,2] > h_crit
        del sub_atoms[idx_to_remove]
        saved_file = 'save_'+calc.fname_in(calc.iteration-calc.run_params[calc.KEY.SLAB_UPDATE_ITERATION])
        if os.path.exists(saved_file):
            calc.logger.print(f"The file {saved_file} exists, append the atoms from the file")
            subs_to_add         = load_atom(saved_file, calc.convert_dict)
            atom_append_h       = subs_to_add.cell[2,2]
            pos_before          = subs_to_add.positions
            subs_to_add.cell[2,2] += h_crit
            subs_to_add.positions = pos_before
        else:
            subs_to_add      = ase.Atoms(cell=sub_atoms.cell)
            atom_append_h    = 0
            subs_to_add.cell[2,2] = h_crit

        for atom in sub_atoms:
            atom.position[2] += atom_append_h
            subs_to_add.append(copy.deepcopy(atom))

        calc.str_out                    = copy.deepcopy(calc.str_out[idx_to_remove])
        calc.str_out.positions[:,2]     -= h_crit

        return subs_to_add

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
