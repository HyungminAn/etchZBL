def main():
    with open('desorption_graph.dat', 'r') as f:
        lines = f.readlines()

    deleted_atom_set = set()
    new_dict = {}
    atom_dict = {
        'Si': 0,
        'O': 1,
        'C': 2,
        'H': 3,
        'F': 4,
    }
    for line in lines:
        if 'reset' in line:
            deleted_atom_set = set()
            continue

        idx, comp, cluster, atom_types = line.strip('\n').split(' / ')
        comp = [int(i) for i in comp.split()]
        cluster = [int(i) for i in cluster.split()]
        atom_types = atom_types.split()

        cluster_new, atom_types_new = [], []
        for i in range(len(cluster)):
            atom_idx = cluster[i]
            if atom_idx in deleted_atom_set:
                comp[atom_dict[atom_types[i]]] -= 1
            else:
                deleted_atom_set.add(atom_idx)
                cluster_new.append(cluster[i])
                atom_types_new.append(atom_types[i])

        comp = " ".join([str(i) for i in comp])
        cluster_new = " ".join([str(i) for i in cluster_new])
        atom_types_new = " ".join(atom_types_new)
        line_new = f"{idx} / {comp} / {cluster_new} / {atom_types_new}\n"
        with open('new_desorption_graph.dat', 'a') as f:
            f.write(line_new)


main()
