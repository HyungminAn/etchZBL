import matplotlib.pyplot as plt


def main():
    with open('outcars_nonzero_spins.dat', 'r') as f:
        lines = f.readlines()

    spin_dict = {}

    for line in lines:
        path, *_, spin = line.split()
        struct_type = path.split('/')[0]

        if not spin_dict.get(struct_type):
            spin_dict[struct_type] = []

        spin_dict[struct_type].append(float(spin))

    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_1d = axes.reshape(-1)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for ax, (struct_type, spin_list) in zip(axes_1d, spin_dict.items()):
        ax.hist(spin_list, bins=20, range=(0, 2))

        textbox = f"{len(spin_list)} structures"
        ax.text(0.95, 0.95, textbox, fontsize=18,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes, bbox=props)

        title = f"{struct_type}"
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig('spin_mag.png')


main()
