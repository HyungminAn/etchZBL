
import ase,os,sys
import numpy as np
from ase.neighborlist import neighbor_list
import ase.io,ase.build
import pickle
from graph_tool import Graph
from graph_tool.topology import label_components
from graph_similarity_save_pickle import atom_image

import ase, os
import numpy as np
from ase.neighborlist import neighbor_list
import ase.build
from ase.io import read, write
from ase.constraints import FixAtoms
import pickle
from graph_tool import Graph
from graph_tool.topology import label_components, similarity
from graph_similarity_save_pickle import atom_image


def get_graph_info(images, bond_length_dict):
    n_images = len(images)
    graphs = [Graph(directed = False) for _ in range(n_images)]
    vprops = []
    clusters = []
    hists = []

    for i in range(n_images):
        graphs[i].add_vertex(len(images[i]))
        NN_list = neighbor_list('ij', images[i], bond_length_dict)
        vprop = graphs[i].new_vertex_property("short",vals=images[i].get_atomic_numbers())
        vprops.append(vprop)
        graphs[i].add_edge_list(zip(NN_list[0],NN_list[1]))
        cluster, hist = label_components(graphs[i])
        clusters.append(cluster)
        hists.append(hist)

    return graphs, vprops, clusters, hists


def is_two_graph_same(graph1, graph2):
    return similarity(graph1, graph2, norm=False, distance=True, asymmetric=True) == 0.0


class new_atom_image(atom_image):
    def __init__(self):
        super().__init__()

    def get_images(self,images,MD_idx):
        self.images=images
        for i in self.images:
            i.set_pbc((True,True,False))
        self.MD_idx=MD_idx

        graphs, vprops, clusters, hists = \
            get_graph_info(self.images, self.bond_length_dict)
        self.graphs = graphs
        self.vprops = vprops
        self.clusters = clusters
        self.hists = hists

    def get_only_bulk(self):
        self.only_bulks=[]

        gas_crit = 4.5
        delete_crit = 18
        n_incidence = self.MD_idx.shape[0]
        n_atom_types = 5

        for idx in range(n_incidence):
            slab_idx = np.argmax(self.hists[idx])
            bulk_idx=np.argwhere(self.clusters[idx].a==slab_idx)
            bulk_height=np.max(self.images[idx].positions[bulk_idx,2])

            cluster_idx= [i for i in range(len(self.hists[idx]))]
            cluster_idx.pop(slab_idx)

            to_delete_idx=np.array([],dtype=int)

            for gas_idx in cluster_idx:
                atom_in_gas_idx = np.argwhere(self.clusters[idx].a==gas_idx)
                gas_height=np.mean(self.images[idx].positions[atom_in_gas_idx,2],axis=0)
                is_this_gas = gas_height >= bulk_height + gas_crit

                if is_this_gas:
                    to_delete_idx=np.concatenate((to_delete_idx, atom_in_gas_idx.flatten()))

            for atom_idx in range(len(self.images[idx])):
                height = self.images[idx].positions[atom_idx,2]
                is_gas_far_from_slab = height >= delete_crit
                if is_gas_far_from_slab:
                    to_delete_idx=np.concatenate((to_delete_idx,[atom_idx]))

            image_bulk = self.images[idx].copy()
            del image_bulk[to_delete_idx]
            image_bulk = ase.build.sort(image_bulk, tags=image_bulk.get_array('id'))
            self.only_bulks.append(image_bulk)

        #remove duplicated bulk configuration
        graphs, _, _, _ = get_graph_info(self.only_bulks, self.bond_length_dict)

        to_delete_image_idx=[]
        for i in range(1,len(self.only_bulks)):
            if is_two_graph_same(graphs[i-1],graphs[i]):
                to_delete_image_idx.append(i)
        self.unique_bulks=[]
        self.unique_image_idx=[]
        n_unique = len(self.only_bulks)-len(to_delete_image_idx)
        self.unique_MD_idx = np.zeros((n_unique, 2),dtype=int)
        self.unique_compositions = np.zeros((n_unique, n_atom_types),dtype=int)
        N = 0

        for i in range(len(self.only_bulks)):
            if i not in to_delete_image_idx:
                self.unique_bulks.append(self.only_bulks[i])
                self.unique_MD_idx[N]=self.MD_idx[i]
                self.unique_image_idx.append(i)
                self.unique_compositions[N]=np.array([np.sum(self.only_bulks[i].get_atomic_numbers()==j) for j in range(1, n_atom_types+1)])
                N+=1

        dst = "post_process_bulk_gas"
        os.makedirs(dst,exist_ok=True)
        matcher={
            1: 14,
            2: 8,
            3: 6,
            4: 1,
            5: 9,
        }
        write_options = {
                'format': 'lammps-data',
                'atom_style': 'atomic',
                'specorder': ["Si", "O", "C", "H", "F"],
                }
        fix_h = 2.0
        for i in range(len(self.unique_bulks)):
            incidence = self.MD_idx[self.unique_image_idx[i],0]
            snapshot_idx = self.MD_idx[self.unique_image_idx[i],1]
            coo_path = f"{dst}/{incidence}/{incidence}_{snapshot_idx}"

            os.makedirs(coo_path,exist_ok=True)
            write(f"{coo_path}/coo", self.unique_bulks[i], **write_options)

            tmp_atoms=self.unique_bulks[i].copy()
            # tmp_atoms.set_atomic_numbers([matcher[j] for j in tmp_atoms.get_atomic_numbers()])
            # arg_sort=tmp_atoms.numbers.argsort()
            s_flag = FixAtoms(indices=[atom.index for atom in tmp_atoms if atom.position[2] < fix_h])
            tmp_atoms.set_constraint(s_flag)
            # write(f"{coo_path}/POSCAR",tmp_atoms[arg_sort],format='vasp')
            write(f"{coo_path}/POSCAR", tmp_atoms, format='vasp', sort=True)

        with open("unique_bulk.dat",'w') as O:
            O.write("#incidence,snapshot_idx\n")
            for i in self.unique_MD_idx:
                O.write(f"{i[0]},{i[1]}\n")

        new_coos=[i.copy() for i in self.unique_bulks]
        # matcher={1:14,2:7,3:1,4:9}
        # for i in range(len(new_coos)):
        #     new_coos[i].set_atomic_numbers( [  matcher[j] for j in new_coos[i].get_atomic_numbers()])
        write("unique_bulk.extxyz", new_coos, format='extxyz')


def get_MD_idx(path):
    with open(path, 'r') as O:
        save_list=[[int(j) for j in i.split()[1:]] for i in O.readlines()[1:]]

    N=0
    for i in save_list:
        for j in i:
            N+=1

    MD_idx=np.zeros((N,2),dtype=int)
    N=0

    for e,i in enumerate(save_list):
        for j in sorted(i):
            MD_idx[N,:]=e+1,j
            N+=1

    return MD_idx


def main():
    path = "./to_rlx.dat"
    MD_idx = get_MD_idx(path)

    with open("bulk_idx.dat",'w') as O:
        O.write("#incidence,snapshot_idx\n")
        for i in MD_idx:
            O.write(f"{i[0]},{i[1]}\n")
    with open(f"nnp_rlx.pickle",'rb') as O:
        coos=pickle.load(O)

    Image = new_atom_image()
    Image.get_images(coos,MD_idx)
    Image.get_only_bulk()


if __name__ == '__main__':
    main()


def get_graph_info(image, bond_length_dict):
    graph = Graph(directed = False)
    graph.add_vertex(len(image))
    NN_list = neighbor_list('ij', image, bond_length_dict)
    vprop = graph.new_vertex_property("short",vals=image.get_atomic_numbers())
    graph.add_edge_list(zip(NN_list[0],NN_list[1]))
    cluster, hist = label_components(graph)
    return graph, vprop, cluster, hist


def write_gas(image, atom_idx_in_gas, image_idx, desorbed_gas_ids):
    write_options = {
        'format': 'lammps-data',
        'atom_style': 'atomic',
        'specorder': ["Si", "O", "C", "H", "F"],
    }
    vector2other_atoms = image.get_distances(atom_idx_in_gas[0],atom_idx_in_gas[1:],mic=True,vector=True)
    tmp_dump = image.copy()
    tmp_dump.positions[atom_idx_in_gas[1:]]=tmp_dump.positions[atom_idx_in_gas[0]]+vector2other_atoms
    delete_atoms=np.setdiff1d(np.arange(len(tmp_dump)),atom_idx_in_gas)
    del tmp_dump[delete_atoms]
    tmp_dump.set_cell([20,21,22])
    dst = f"post_process_bulk_gas/gas/{image_idx}/{image_idx}_{len(desorbed_gas_ids)-1}/"
    os.makedirs(dst, exist_ok=True)
    ase.io.write(f"{dst}/coo", tmp_dump, **write_options)


def get_removed_atom(image_init, image_final, bond_length_dict):
    initial_id = image_init.get_array('id')
    final_id = image_final.get_array('id')

    removed_atom = np.setdiff1d(initial_id,final_id)
    graph, _, cluster, hist = \
        get_graph_info(image_final, bond_length_dict)

    if len(hist) == 1:
        # No gas
        return removed_atom

    h_crit = 4.5
    slab_idx=np.argmax(hist)
    slab_atom_idx=np.argwhere(cluster.a==slab_idx)
    bulk_height=np.max(image_final.positions[slab_atom_idx,2])
    atoms_above_bulk_idx = np.argwhere(image_final.positions[:,2]>bulk_height+h_crit).flatten()
    removed_atom=np.concatenate((removed_atom, image_final.get_array('id')[atoms_above_bulk_idx].flatten()))

    return removed_atom


class new_atom_image(atom_image):
    def __init__(self):
        super().__init__()

    def get_images(self,image_path):
        with open(image_path,'rb') as O:
            self.dump=pickle.load(O)


def get_desorbed_gas_ids(Image, removed_atom):
    confirmed_removed_atom_id=[]
    desorbed_gas_ids=[]
    h_crit = 16.0

    for atom_id in removed_atom:
        if atom_id in confirmed_removed_atom_id:
            continue

        for i in reversed(range(100, len(Image.dump))):
            image = Image.dump[i]
            atom_ids = image.get_array('id')
            cond1 = atom_id in atom_ids
            if not cond1:
                continue

            gas_atom_idx = np.nonzero(atom_ids==atom_id)[0][0]
            cond2 = image.positions[gas_atom_idx, 2] <= h_crit
            if not cond2:
                continue

            graph, _, cluster, hist = get_graph_info(image, Image.bond_length_dict)

            atom_idx=np.nonzero(atom_ids==atom_id)[0][0]
            gas_idx=cluster.a[atom_idx]
            atom_idx_in_gas=np.nonzero(cluster.a==gas_idx)[0]
            atom_id_in_gas=Image.dump[i].get_array('id')[atom_idx_in_gas].flatten()
            desorbed_gas_ids.append(atom_id_in_gas)

            for id in atom_id_in_gas:
                confirmed_removed_atom_id.append(id)

            if len(atom_idx_in_gas) > 1:
                write_gas(image, atom_idx_in_gas, i, desorbed_gas_ids)

            break

    return desorbed_gas_ids


def get_gas(Image, image_idx):
    image_init, image_final = Image.dump[1], Image.dump[-1]
    removed_atom = get_removed_atom(
        image_init, image_final, Image.bond_length_dict)

    desorbed_gas_ids = get_desorbed_gas_ids(Image, removed_atom)

    with open("desorbed_gas_id.dat",'a') as O:
        n_species = 5
        image_init = Image.dump[1]

        map_dict = {
            'Si': 0,
            'O': 1,
            'C': 2,
            'H': 3,
            'F': 4,
        }

        for e, i in enumerate(desorbed_gas_ids):
            composition=np.zeros((n_species),dtype=int)
            for j in i:
                atom_idx=np.nonzero(image_init.get_array('id')==j)[0][0]
                composition[map_dict[image_init[atom_idx].symbol]] += 1
            line = f"{image_idx}/{e}/"
            line += ",".join(map(str, composition))
            line += "/"
            O.write(line)
            for j in i:
               O.write(f"{j},")
            O.write("\n")


def main():
    src = sys.argv[1]
    Image = new_atom_image()
    with open("desorbed_gas_id.dat",'w') as O:
        O.write(f"#incidence/gas_idx/composition/ids\n")

    # n_incidence = 50
    n_incidence = 20

    for i in range(1, n_incidence+1):
        with open(f"{src}/graph_{i}.pickle", 'rb') as O:
            tmp=pickle.load(O)

        Image.dump=tmp.dump
        get_gas(Image, i)


if __name__ == '__main__':
    main()
