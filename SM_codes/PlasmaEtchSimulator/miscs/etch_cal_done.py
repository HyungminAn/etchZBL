#!/bin/python
import copy
import pickle
import ase
import ase.io
import os, sys
import matplotlib.pyplot as plt
## From sepcific directory, find all calculation and check calculation is done from vasp (OUTCAR, POSCAR)
import pathos.multiprocessing as mp
import numpy as np
import typing
def find_calculations(directory : str) -> list:
    calcs = []
    to_do = []
    for root, dirs, files in os.walk(directory):
        if 'POSCAR' in files:
            to_do.append(root)

        if 'OUTCAR' in files:
            calcs.append(root)

    print(f"The calulations are {len(calcs)} {len(to_do)} {len(calcs)/len(to_do)*100:.2f}")
    return calcs

def parse_energy(directory : str, additional_parse : str = None) -> dict:
    out_dict = {}
    for root, dirs, files in os.walk(directory):
        if 'POSCAR' in files:
            ctype  = root.split('/')[-4]
            nshoot = int(root.split('/')[-2])
            nstr   = int(root.split('/')[-1])
            if ctype not in out_dict:
                out_dict[ctype] = {}
            if nshoot not in out_dict[ctype]:
                out_dict[ctype][nshoot] = {}
            if nstr not in out_dict[ctype][nshoot]:
                out_dict[ctype][nshoot][nstr] = {
                    'dft'       : None,
                    '7net'      : None,
                    'natom'     : None,
                    'dft_atom'  : None,
                    '7net_atom' : None,
            }
            if 'OUTCAR' in files:
                try: 
                    atom = ase.io.read(f"{root}/OUTCAR",format='vasp-out')
                    out_dict[ctype][nshoot][nstr]['dft'] = atom.get_potential_energy()
                    out_dict[ctype][nshoot][nstr]['dft_atom'] = atom
                except:
                    print(f"Error in reading {root}/OUTCAR")
                    
            if 'out.extxyz' in files:
                atom = ase.io.read(f"{root}/out.extxyz",format='extxyz')
                out_dict[ctype][nshoot][nstr]['7net']       = atom.get_potential_energy()
                out_dict[ctype][nshoot][nstr]['7net_atom']  = atom
                out_dict[ctype][nshoot][nstr]['natom']      = len(atom)
            print(f"Done {root}")

            # Create data label
    return out_dict

def process_data(data : dict) -> dict:
    """Chage parse data to a specific format : list of DFT and 7net using orering

    Args:
        data (dict): Dictionary of parsed data

    Returns:
        dict: Dictionary of list of DFT and 7net
    """
    out_dict = {}
    atom_list = {
        'dft' : [],
        '7net': [],
    }
    for dkey in data.keys():
        sel_data = data[dkey]
        all_keys = []
        for nshoot in sel_data.keys():
            for nstr in sel_data[nshoot].keys():
                all_keys.append((nshoot,nstr))
        ## Sort  by nshoot and nstr
        all_keys = sorted(all_keys, key = lambda x: (x[0],x[1]))
        dft_en = []
        net_en = []
        num_atom = []
        for key in all_keys:
            if sel_data[key[0]][key[1]]['dft'] is None or sel_data[key[0]][key[1]]['7net'] is None:
                print('cut')
                continue
            dft_en.append(sel_data[key[0]][key[1]]['dft'])
            net_en.append(sel_data[key[0]][key[1]]['7net'])
            num_atom.append(sel_data[key[0]][key[1]]['natom'])
            atom_list['dft'].append(sel_data[key[0]][key[1]]['dft_atom'])
            atom_list['7net'].append(sel_data[key[0]][key[1]]['7net_atom'])
        out_dict[dkey] = {
            'dft' : dft_en,
            '7net': net_en,
            'natom': num_atom
        }

    return out_dict, atom_list 

def plot_energy(data : dict, fname : str) -> None:
    xsize = 5
    ysize = 2

    
    caltype = sorted(list(data.keys()), key = lambda x: (x.split('_')[-2], x.split('_')[-1]))
    ntype   = len(caltype)

    figsize = (xsize,ysize * ntype)
    ## subplots with ntype rows

    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(ntype, 1,  figsize=figsize)

    for calidx, cal in enumerate(caltype):
        ax = fig.axes[calidx]
        sel_data = data[cal]
        ## Sort data by nshoot and nstr which (1,0) ,(1,1) ,(1,2) ... (2,0) (2,1) ... ordering

        ax.plot(sel_data['dft'],    label = 'DFT', color = 'red', linestyle = '--', linewidth = 1)
        ax.plot(sel_data['7net'],   label = '7net', color = 'blue', linestyle = '-', linewidth = 1)
        if calidx == 0:
            ax.legend(loc='upper right')
        ax.set_title(f"{cal}")
        ax.set_xlabel('Structure index')        
        ax.set_ylabel('Energy (eV)')
    plt.tight_layout()
    plt.savefig(fname)


def plot_scatter(data : dict, fname : str, label : list = None) -> None:
    '''Plot scatter plot of DFT energy/natom vs 7net energy/natom at once'''
    size = 3
    figsize = (size, size)
    ## subplots with ntype rows
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)

    ## Collct data from label
    key_to_col = list(data.keys()) if label == None else label
    dft_data = np.array([])
    net_data = np.array([])
    for key in key_to_col:
        dft_data = np.concatenate((dft_data, np.array(data[key]['dft'])/np.array(data[key]['natom'])))
        net_data = np.concatenate((net_data, np.array(data[key]['7net'])/np.array(data[key]['natom'])))
    ## Range setting for x and y which are same
    
    
    ## Density coloring
    ## Color by density to plot ax.scatter
    import scipy.stats as stats

    ## Density calculation for color
    xy = np.vstack([dft_data, net_data])
    z = stats.gaussian_kde(xy)(xy)
    ## Sort by density
    idx = z.argsort()
    dft_data = dft_data[idx]
    net_data = net_data[idx]
    z = z[idx]
    ax.scatter(dft_data, net_data, s = 2, c = z, edgecolors='none', alpha = 0.9, zorder = 1, cmap='viridis')
    ax_min = min(dft_data.min(), net_data.min()) - 0.1
    ax_max = max(dft_data.max(), net_data.max()) + 0.1
    ax_max = -6.3
    ## Plot diagonal line
    ax.plot([ax_min, ax_max], [ax_min, ax_max], color = 'black', linestyle = '--', linewidth = 1, zorder = 0)

    plt.tight_layout()
    print(f"ax_min {ax_min} ax_max {ax_max}")
    # Tick is per 0.0 -0.5 -1.0 ... whithin the range  
    ticks = np.arange(0.5 * ax_min // 0.5 , 0 ,0.5)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
    
    
    plt.xlabel('DFT eV/atom')
    plt.ylabel('7net eV/atom')
    ax.set_aspect('equal', 'box')
    ## Show RMSE and MAE to scatter plot using ax.text
    mae = np.mean(np.abs(dft_data - net_data))
    rmse = np.sqrt(np.mean((dft_data - net_data)**2))
    ax.text(ax_min + 0.1, ax_max - 0.2, f"MAE  : {1e3*mae:.1f} meV/atom\nRMSE : {1e3*rmse:.1f} meV/atom",     fontsize = 8)

    plt.savefig(fname)



def plot_force_scatter(data : dict, fix : float, fname : str, label : list = None, nproc : int = 0, select_HF : bool = False) -> None:
    '''Plot scatter plot of DFT energy/natom vs 7net energy/natom at once'''
    size = 3
    figsize = (size, size)
    ## subplots with ntype rows
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)

    ## Collct data from label
    key_to_col = list(data.keys()) if label == None else label
    dft_data = np.array([])
    net_data = np.array([])
    for  datom, natom in zip(data['dft'], data['7net']):
        if select_HF:
            ## Select HF molecule from ase get distance 
            dist = datom.get_all_distances(mic = True)
            ## Get symbol from atom
            symbol = datom.get_chemical_symbols()
            H_idx = np.where(np.array(symbol) == 'H')[0]
            F_idx = np.where(np.array(symbol) == 'F')[0]
            ## check HF distance 
            select_idx = []
            for hidx in H_idx:
                for fidx in F_idx:
                    if dist[hidx][fidx] < 1.9:
                        print(f"Found HF distance {dist[hidx][fidx]}")
                        select_idx.append(hidx)
                        select_idx.append(fidx)
            select_idx = np.array(select_idx)
        else:        
            ## Get force array from atom object without fix layer from bottom
            select_idx = np.where(datom.get_positions()[:,2] > fix)[0]
        if len(select_idx) == 0:
            continue
        dfc = datom.get_forces()[select_idx]
        nfc = natom.get_forces()[select_idx]
        dft_data = np.concatenate((dft_data, dfc.reshape(-1))) 
        net_data = np.concatenate((net_data, nfc.reshape(-1)))

    ## Density coloring
    if False:
        print("Scipy calculation")
        import scipy.stats as stats
        ## Density calculation for color
        xy = np.vstack([dft_data, net_data])
        z = stats.gaussian_kde(xy)(xy)
        ## Sort by density
        idx = z.argsort()
    else:
        print("Torch calculation")
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Example data
        data = torch.tensor(np.vstack([dft_data, net_data]).T, dtype=torch.float32).to(device)
        n = data.size(0)
        batch_size = 1000  # Adjust batch size based on available memory
        bandwidth = 0.1
        # Placeholder for density values
        torch.no_grad()
        density = torch.zeros(n, device=device)
        # Calculate density in batches to reduce memory usage
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_data = data[i:end_i].unsqueeze(1)  # Shape: (batch_size, 1, 2)
            diff = batch_data - data.unsqueeze(0)    # Shape: (batch_size, n, 2)
            distances = torch.sqrt(torch.sum(diff**2, dim=-1))  # Pairwise distances, Shape: (batch_size, n)
            density[i:end_i] = torch.exp(-(distances / bandwidth) ** 2).sum(dim=1)  # Sum along the last dimension
            print(f"Done {i} to {end_i}")
        z = density.cpu().numpy()
        idx = z.argsort()

    dft_data = dft_data[idx]
    net_data = net_data[idx]
    z = z[idx]
    ax.scatter(dft_data, net_data, s = 2, c = z, edgecolors='none', alpha = 0.9, zorder = 1, cmap='viridis')

    fmax = 15
    ## Plot diagonal line
    ax.plot([-fmax, fmax], [-fmax, fmax], color = 'black', linestyle = '--', linewidth = 1, zorder = 0)

    plt.tight_layout()
    # Tick is per 0.0 -0.5 -1.0 ... whithin the range  
    ticks = [-30+5*idx for  idx in range(13)]
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(-fmax, fmax)
    plt.ylim(-fmax, fmax)
    
    plt.title(f"Force parity plot")    
    plt.xlabel('DFT  (eV/atom)')
    plt.ylabel('7net (eV/atom)')
    ax.set_aspect('equal', 'box')
    ## Show RMSE and MAE to scatter plot using ax.text
    mae = np.mean(np.abs(dft_data - net_data))
    rmse = np.sqrt(np.mean((dft_data - net_data)**2))
    ax.text(-fmax * 0.9, fmax * 0.75, f"MAE  : {mae:.1f} eV/$\AA$\nRMSE : {rmse:.1f} eV/$\AA$",     fontsize = 8)

    plt.savefig(fname)


def find_force_idx(data : dict, fix : float, absFmax : float = None, absFmin : float = None, gradmax : float = None, gradmin : float = None, fname : str = None) -> None:
    '''This function find specific region of force (for exapmble |F|>5, the gradient (7net F/DFT F) over 1.5 ...  )
    and get the index of structure & index of atom ....  
    '''
    dft_data = np.array([])
    net_data = np.array([])
        
    index_to_atom = []

    for  struct_idx, (datom, natom) in enumerate(zip(data['dft'], data['7net'])):
        select_idx = np.where(datom.get_positions()[:,2] > fix)[0]
        if len(select_idx) == 0:
            continue

        dfc = datom.get_forces()[select_idx]
        nfc = natom.get_forces()[select_idx]
        
        ## Save the number of atom index
        dft_data = np.concatenate((dft_data, dfc.reshape(-1))) 
        net_data = np.concatenate((net_data, nfc.reshape(-1)))

        ## Save the index of structure and atom which I want to extract it from condition that calcualted later
        for atom_idx in select_idx:
            index_to_atom += [(struct_idx, atom_idx)] * 3    
    

    # Define a helper function to apply a condition and store results
    def apply_condition(conin, condition):
        if condition is not None:
            conout = np.logical_and(conin, condition)
        return conout
    select_list = [True]*len(dft_data)
    # Apply conditions and store results in matching_indices
    select_list = apply_condition(select_list, np.abs(dft_data) < absFmax)
    select_list = apply_condition(select_list, np.abs(dft_data) > absFmin)
    select_list = apply_condition(select_list, np.abs(net_data / dft_data) < gradmax)
    select_list = apply_condition(select_list, np.abs(net_data / dft_data) > gradmin)

 
    dft_data = dft_data[select_list]
    net_data = net_data[select_list]
 
    if fname != None:
        size = 3
        figsize = (size, size)
        ## subplots with ntype rows
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots(figsize=figsize)
        ## This is test plot for force scatter plot
        ax.scatter(dft_data, net_data, s = 1, edgecolors='none', alpha = 0.5, zorder = 1)
        fmax = 15
        ## Plot diagonal line
        ax.plot([-fmax, fmax], [-fmax, fmax], color = 'black', linestyle = '--', linewidth = 1, zorder = 0)

        plt.tight_layout()
        # Tick is per 0.0 -0.5 -1.0 ... whithin the range  
        ticks = [-30+5*idx for  idx in range(13)]
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.xlim(-fmax, fmax)
        plt.ylim(-fmax, fmax)
        
        plt.title(f"Force parity plot")    
        plt.xlabel('DFT  (eV/atom)')
        plt.ylabel('7net (eV/atom)')
        ax.set_aspect('equal', 'box')
        plt.savefig(fname)

    ## SAVE selected index to ase POSCAR with converted symbol 
    sym_conv = {
        'Si': 'Ge',
        'N': 'P',
        'C': 'B',
        'H': 'Li',
        'F': 'Cl',
    }
    stat = {
        'Si': [],
        'N': [],
        'C': [],
        'H': [],
        'F': [],
    }

    for  index, idx_bool in enumerate(select_list):
        # Get the index of structure and atom
        if idx_bool:
            struct_idx, atom_idx = index_to_atom[index]
            # Get the atom object
            atom = copy.deepcopy(data['dft'][struct_idx])
            ## Print the force condition is met 
            vdfc = data['dft'][struct_idx].get_forces()[atom_idx]
            vnfc = data['7net'][struct_idx].get_forces()[atom_idx]
            print(f"Force condition is met {struct_idx} {atom_idx} {vdfc} {vnfc} {vnfc/vdfc}")

            
            # Change the chemical symbol from atom_idx
            if (struct_idx, atom_idx) in index_to_atom:
                stat[atom[atom_idx].symbol].append((struct_idx, atom_idx))
            
            atom[atom_idx].symbol = sym_conv[atom[atom_idx].symbol]            
            # Save the structure
            ase.io.write(f"sel/POSCAR_{index}", atom, format='vasp')
            print(f"Saved {struct_idx} {atom_idx} {index}")

    for key in stat.keys():
        print(f"{key} {len(stat[key])}")

if __name__ == '__main__':
    loc = '/data2/gasplant63/etch_gnn/7_SiNOCHF/15_small_cell_etch/5_dft_comp'
    calcs = find_calculations(loc)

    if os.path.exists(f'{loc}/energy.pkl') and True:
        calcs = pickle.load(open(f'{loc}/energy.pkl','rb'))
    else:
        calcs = parse_energy(loc)
        pickle.dump(calcs,open(f'{loc}/energy.pkl','wb'))

    pcalcs, atom_list = process_data(calcs)
    # plot_energy(pcalcs,f'{loc}/energy.png')
    # plot_scatter(pcalcs,f'{loc}/scatter.png')
    # plot_force_scatter(atom_list, 4, f'{loc}/force_scatter.png')
    # plot_force_scatter(atom_list, 4, f'{loc}/force_scatter_HF.png', select_HF = True)
    find_force_idx(atom_list, 4, absFmax = 15, absFmin = 5, gradmin= 0.5, gradmax= 0.7, fname = 'test_plot.png')