import ase,os,sys
import numpy as np
from ase.neighborlist import neighbor_list
import ase.io,ase.build
import pickle
from graph_tool import Graph
from graph_tool.topology import label_components
from graph_similarity_save_pickle import atom_image


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
