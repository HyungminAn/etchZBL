import matplotlib.pyplot as plt


def main():
    with open('time.dat', 'r') as f:
        lines = f.readlines()

    time_dict = {}

    for line in lines:
        path, *_, time = line.split()

        path = path.split('/')[1]
        if not time_dict.get(path):
            time_dict[path] = []

        time_dict[path].append(float(time))

    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_1d = axes.reshape(-1)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for ax, (path, time_list) in zip(axes_1d, time_dict.items()):
        ax.hist(time_list, bins=50, range=(0, 5000))

        time_sum = sum(time_list)
        time_avg = time_sum / len(time_list)
        time_sum = round(time_sum)
        time_avg = round(time_avg)

        textbox = f"total : {time_sum} (s)\navg : {time_avg} (s)"
        ax.text(0.95, 0.95, textbox, fontsize=18,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes, bbox=props)

        title = f"{path} ({len(time_list)} structures)"
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig('hist_time.png')


main()
