def main():
    with open('new_desorption_graph.dat', 'r') as f:
        lines = f.readlines()

    tot_dict = {}
    for line in lines:
        idx, comp, cluster, atom_types = line.strip('\n').split(' / ')
        n_Si = int(comp.split()[0])
        if n_Si > 0:
            if tot_dict.get(comp) is not None:
                tot_dict[comp] += 1
            else:
                tot_dict[comp] = 1

    for key, value in tot_dict.items():
        print(key, value)


main()
