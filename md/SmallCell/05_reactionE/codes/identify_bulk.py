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
