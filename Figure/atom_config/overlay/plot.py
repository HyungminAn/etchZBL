import pickle
import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, axis_width=6.744, axis_height=4.709, bar_width=0.555):
        # Dimensions in inches
        self.axis_width = axis_width
        self.axis_height = axis_height
        self.bar_width = bar_width

    def run(self, dict1, dict2, dict3, dict4):
        # Shared sorted keys
        keys = sorted(dict1.keys())
        n = len(keys)

        # Extract values
        h1 = np.array([dict1[k] for k in keys], dtype=float)
        h2 = np.array([dict2[k] for k in keys], dtype=float)
        z3_min = np.array([dict3[k][0] for k in keys], dtype=float)
        z3_max = np.array([dict3[k][1] for k in keys], dtype=float)
        z4_min = np.array([dict4[k][0] for k in keys], dtype=float)
        z4_max = np.array([dict4[k][1] for k in keys], dtype=float)

        # Shift relative to first h2 value
        shift = h2 - h2[0]

        # Compute mixed and carbon layer ranges
        mixed_bottom = shift - (h1 - z3_min)
        mixed_top = shift - (h1 - z3_max)
        carbon_bottom = shift - (h1 - z4_min)
        carbon_top = shift - (h1 - z4_max)

        for i in range(len(mixed_bottom)):
            if np.isnan(mixed_bottom[i]) and np.isnan(mixed_top[i]):
                mixed_bottom[i] = 0
                mixed_top[i] = 0

        for i in range(len(carbon_bottom)):
            if np.isnan(carbon_bottom[i]) and np.isnan(carbon_top[i]):
                carbon_bottom[i] = mixed_top[i]
                carbon_top[i] = mixed_top[i]

        # Determine first bar height for y-limits
        idx_min = np.argmin(h2)
        first_bar_height = h1[idx_min] + h2[0] - h2[idx_min]

        # Compute bar positions in data units
        bw = self.bar_width
        spacing = (self.axis_width - n * bw) / (n - 1)
        centers = bw / 2 + np.arange(n) * (bw + spacing)

        # Create figure and axis with transparent background
        fig = plt.figure(figsize=(self.axis_width, self.axis_height))
        ax = fig.add_axes([0, 0, 1, 1])
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # Connect top and bottom endpoints with dashed lines
        x_left = [x-self.bar_width/2 for x in centers]
        x_right = [x+self.bar_width/2 for x in centers]
        substrate_top = mixed_bottom.copy()
        substrate_bottom = np.zeros_like(mixed_bottom)
        data_to_plot = {
            'mixed_top': mixed_top,
            'mixed_bottom': mixed_bottom,
            'carbon_top': carbon_top,
            'carbon_bottom': carbon_bottom,
            'substrate_top': substrate_top,
            'substrate_bottom': substrate_bottom
        }
        color_dict = {
            'mixed_top': 'red',
            'mixed_bottom': 'red',
            'carbon_top': 'black',
            'carbon_bottom': 'black',
            'substrate_top': 'blue',
            'substrate_bottom': 'blue',
        }
        shift_dict = {
            'mixed_top': -0.5,
            'mixed_bottom': 0.5,
            'carbon_top': 0,
            'carbon_bottom': 0.5,
            'substrate_top': -0.5,
            'substrate_bottom': 0.5,
        }

        for y_type, y in data_to_plot.items():
            s = shift_dict[y_type]
            c = color_dict[y_type]

            xls = x_left.copy()
            xrs = x_right.copy()
            ys = y.copy()

            if y_type == 'carbon_top' or y_type == 'carbon_bottom':
                xls = xls[3:]
                xrs = xrs[3:]
                ys = ys[3:]

            points_left = []
            points_right = []
            # Draw solid lines for mixed and carbon layers
            for xl, xr, yi in zip(xls, xrs, ys):
                yi += s
                ax.plot([xl, xr], [yi, yi], linestyle='-', color=c)
                points_left.append((xl, yi))
                points_right.append((xr, yi))
            # Draw dashed lines connecting left and right points
            for i in range(len(points_left) - 1):
                x0, y0 = points_right[i]
                x1, y1 = points_left[i+1]
                ax.plot([x0, x1], [y0, y1], linestyle='--', color=c)

        # Set axis limits
        ax.set_xlim(centers[0] - bw/2, centers[-1] + bw/2)
        ax.set_ylim(-first_bar_height, 0)

        # X-axis ticks labeled by keys
        ax.set_xticks(centers)
        ax.set_xticklabels(keys)

        # Y-axis ticks and label
        ax.set_yticks([0, -first_bar_height])
        ax.set_yticklabels(['0', f'-{first_bar_height:.2f}'])
        ax.set_ylabel('Height change')

        # # Add bidirectional arrow for scale
        # # arrow_x = centers[0] - bw
        # arrow_x = centers[0]
        # ax.annotate('', xy=(arrow_x, 0), xytext=(arrow_x, -first_bar_height),
        #             arrowprops=dict(arrowstyle='<->', color='black'))

                # Add scale ruler (25 nm total, 5 nm ticks) next to the first bar
        ruler_x = centers[0]  # Position to the left of first bar
        ruler_top = 0
        y_max = first_bar_height
        ruler_bottom = -y_max

        # Draw main ruler line
        ax.plot([ruler_x, ruler_x], [ruler_bottom, ruler_top], color='black', linewidth=1.2)

        # Draw 5 nm ticks
        tick_interval = y_max / 5
        for i in range(6):  # 0, 5, ..., 25
            y_tick = -i * tick_interval
            ax.plot([ruler_x - 0.1, ruler_x + 0.1], [y_tick, y_tick], color='black', linewidth=1)

            # # Label every 10 nm tick
            # if i % 2 == 0:
            #     ax.text(ruler_x - 0.2, y_tick, f'{-y_tick:.0f}', va='center', ha='right', fontsize=8)

        fig.savefig('result.png', dpi=200)
        fig.savefig('result.eps')
        fig.savefig('result.pdf')
        return fig, ax

class DataLoader:
    def run(self):
        src = "../05_RegimeAnalysis/SiO2"
        path_images = f"{src}/CF_300_images.pkl"
        path_z_shift = f"{src}/CF_300_shifted_height.txt"
        path_z_mixed = f"{src}/CF_300_z_mixed.txt"
        path_z_film = f"{src}/CF_300_z_film.txt"
        keys = [10] + [i for i in range(1350, 13501, 1350)]
        result = {
            'z': self.extract_z(keys, path_images),
            'z_shift': self.extract_z_shift(keys, path_z_shift),
            'z_mixed': self.extract_z_mixed(keys, path_z_mixed),
            'z_film': self.extract_z_carbon(keys, path_z_film)
            }
        return result

    def extract_z(self, keys, path_images):
        with open(path_images, "rb") as f:
            images = pickle.load(f)
        result = {}
        for key in keys:
            result[key] = np.max(images[key].get_positions()[:, 2])
        print(f"Extracted heights for keys: {keys}")
        return result

    def extract_z_shift(self, keys, path_data):
        data = np.loadtxt(path_data, dtype=float, skiprows=1)
        result = {}
        for x, y in data:
            result[int(x)] = y
        print(f"Extracted shifted heights for keys: {keys}")
        return result

    def extract_z_mixed(self, keys, path_data):
        data = np.loadtxt(path_data, dtype=float, skiprows=1)
        result = {}
        for x, z_min, z_max, h in data:
            result[int(x)] = (z_min, z_max, h)
        print(f"Extracted mixed heights for keys: {keys}")
        return result

    def extract_z_carbon(self, keys, path_data):
        data = np.loadtxt(path_data, dtype=float, skiprows=1)
        result = {}
        for x, z_min, z_max, h in data:
            result[int(x)] = (z_min, z_max)
        print(f"Extracted carbon heights for keys: {keys}")
        return result


def main():
    dl = DataLoader()
    data = dl.run()

    pl = Plotter()
    pl.run(
        data['z'],
        data['z_shift'],
        data['z_mixed'],
        data['z_film']
        )


if __name__ == "__main__":
    main()
