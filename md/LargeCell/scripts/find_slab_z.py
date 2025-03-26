import sys

import numpy as np

from graph_tool.topology import label_components
from AtomImage import AtomImage
"""
This calculates max z-coordinate of the slab.

Original version : Hyungmin An (2023. 11. 08)
"""


def get_max_z(atomimage):
    atomimage.draw_graph()
    cluster, hist = label_components(atomimage.graph)
    slab_idx = np.argmax(hist)
    idx_slab_atoms = np.argwhere(cluster.a == slab_idx)
    pos_z = atomimage.atoms.get_positions()[idx_slab_atoms, 2]
    max_z = np.max(pos_z)
    print(max_z)


def main():
    path = sys.argv[1]
    Image = AtomImage()
    Image.read_atom_image(path)
    Image.find_NN()
    get_max_z(Image)


if __name__ == '__main__':
    main()
