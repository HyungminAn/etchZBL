import pickle

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import timeit

class SankeyDiagram:
    """
    A class for creating and saving a Sankey diagram from a nested dictionary
    of the form total_dict[struct_idx][global_idx] = state_C.
    """

    def __init__(self, total_dict):
        """
        Initialize with a given nested dictionary.

        total_dict: A nested dictionary such that total_dict[struct_idx][global_idx] = state_C.
        """
        self.total_dict = total_dict

    def get_unique_states(self, struct_idx):
        """
        Return a sorted list of unique state values for the given struct_idx.
        """
        return sorted(set(self.total_dict[struct_idx].values()))

    def compute_transitions(self, struct_idx0, struct_idx1):
        """
        Compute how many times a state at struct_idx0 transitions
        to a state at struct_idx1.

        Returns a dictionary {(state0, state1): count}.
        """
        transitions = {}
        for g_idx in self.total_dict[struct_idx0].keys():
            s0 = self.total_dict[struct_idx0][g_idx]
            s1 = self.total_dict[struct_idx1][g_idx]
            transitions[(s0, s1)] = transitions.get((s0, s1), 0) + 1
        return transitions

    def build_sankey_data(self, struct_idx0, struct_idx1):
        """
        (Two-step version)
        Build and return the data (labels, source, target, value) needed
        to create a Sankey diagram for the states at struct_idx0
        transitioning to states at struct_idx1.
        """
        labels_0 = self.get_unique_states(struct_idx0)
        labels_1 = self.get_unique_states(struct_idx1)
        transitions = self.compute_transitions(struct_idx0, struct_idx1)

        source = []
        target = []
        value = []

        for (s0, s1), count in transitions.items():
            i0 = labels_0.index(s0)
            i1 = labels_1.index(s1) + len(labels_0)
            source.append(i0)
            target.append(i1)
            value.append(count)

        labels = labels_0 + labels_1

        return labels, source, target, value

    def build_multi_step_sankey_data(self, struct_idx_list):
        """
        (Multi-step version)
        Build and return the data (labels, source, target, value) needed
        to create a Sankey diagram for multiple struct_idx steps, e.g. [0, 500, 1000].

        struct_idx_list: A list of struct indices in chronological or logical order.
                         Example: [0, 500, 1000].

        1) Extract the unique states for each struct_idx.
        2) Concatenate these states into a single labels list,
           keeping track of each step's offset.
        3) For each consecutive pair (i, i+1), compute transitions and fill source/target/value.
        4) Return (labels, source, target, value).
        """
        step_label_lists = []
        offsets = []
        total_labels = []
        current_offset = 0

        # 1) Gather states and compute offsets
        for s_idx in struct_idx_list:
            unique_states = self.get_unique_states(s_idx)

            step_label_lists.append(unique_states)
            offsets.append(current_offset)
            current_offset += len(unique_states)
            total_labels.extend(unique_states)

        source = []
        target = []
        value = []

        # 2) For each pair of steps, compute transitions
        for i in range(len(struct_idx_list) - 1):
            struct_idx0 = struct_idx_list[i]
            struct_idx1 = struct_idx_list[i+1]
            transitions = self.compute_transitions(struct_idx0, struct_idx1)

            label_0 = step_label_lists[i]
            label_1 = step_label_lists[i+1]
            offset_0 = offsets[i]
            offset_1 = offsets[i+1]

            # Fill source, target, value
            for (s0, s1), count in transitions.items():
                # Find the index in step i's labels
                i0 = offset_0 + label_0.index(s0)
                # Find the index in step i+1's labels
                i1 = offset_1 + label_1.index(s1)
                source.append(i0)
                target.append(i1)
                value.append(count)

        return total_labels, source, target, value

    @staticmethod
    def get_node_colors(labels):
        preferred_colors = {
                'NOT_CREATED': 'white',
                'C3': 'red',
                'SiC_cluster': 'yellow',
                'C2': 'blue',
                'Fluorocarbon': 'green',
                'BYPRODUCT': 'black',
                }
        default_color = 'lightgray'
        color_cycle = px.colors.qualitative.Set2
        cycle_idx = 0
        node_colors = []
        for lab in labels:
            if lab in preferred_colors:
                node_colors.append(preferred_colors[lab])
            else:
                # node_colors.append(color_cycle[cycle_idx % len(color_cycle)])
                node_colors.append(default_color)
                cycle_idx += 1
        return node_colors

    @staticmethod
    def get_edge_colors(labels, source, target):
        link_colors = []
        def is_valid(label1,
                     label2,
                     left_side=False,
                     right_side=False,
                     both_sides=False):
            target_labels = [ 'C3' ]
            if (label1 == label2):
                return False

            if left_side and label1 in target_labels:
                return True

            if right_side and label2 in target_labels:
                return True

            if both_sides and label1 in target_labels and label2 in target_labels:
                return True

            return False

        for s, t in zip(source, target):
            label_s = labels[s]
            label_t = labels[t]
            # if is_valid(label_s, label_t, both_sides=True):
            #     link_colors.append("rgba(200,200,200,0.5)")
            if is_valid(label_s, label_t, right_side=True):
                link_colors.append("rgba(0,0,0,0.5)")
            elif is_valid(label_s, label_t, left_side=True):
                link_colors.append("rgba(200,200,200,0.5)")
            else:
                link_colors.append("rgba(255,255,255,0)")
        return link_colors

    @staticmethod
    def get_labels_with_numbers(labels, source, value):
        node_out_flow = [0]*len(labels)
        for s_idx, v in zip(source, value):
            node_out_flow[s_idx] += v

        #    예: "A_0 (120)"
        node_labels = []
        for i, lab in enumerate(labels):
            node_labels.append(f"{lab} ({node_out_flow[i]})")

        link_labels = [str(v) for v in value]
        return node_labels, link_labels

    def create_sankey_figure(self, labels, source, target, value, title="State Transitions"):
        """
        Create and return a Plotly Sankey figure given labels, source, target, and value lists.

        title: The title for the diagram.
        """
        node_colors = self.get_node_colors(labels)
        edge_colors = self.get_edge_colors(labels, source, target)
        node_labels, link_labels = self.get_labels_with_numbers(labels, source, value)
        fig = go.Figure(data=[go.Sankey(
            # arrangement="fixed",
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=15,
                line=dict(color="black", width=0.5),
                # label=labels,
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=edge_colors,
            )
        )])
        fig.update_layout(title_text=title, font_size=10)
        return fig

    def save_sankey_diagram(self, struct_idx0, struct_idx1, filename, width=1200, height=800, scale=1.0):
        """
        (Two-step version)
        Build and save the Sankey diagram for the specified struct indices to an image file.

        struct_idx0: The first struct index (e.g., 0).
        struct_idx1: The second struct index (e.g., 1000).
        filename: Output file path (e.g., 'sankey_diagram.png').
        width, height: Dimensions of the image in pixels.
        scale: Resolution scale factor.
        """
        labels, source, target, value = self.build_sankey_data(struct_idx0, struct_idx1)
        fig = self.create_sankey_figure(
            labels, source, target, value,
            title=f"struct_idx={struct_idx0} → struct_idx={struct_idx1} State Transitions"
        )
        fig.write_image(filename, width=width, height=height, scale=scale)
        print(f"Sankey diagram saved to {filename} (size: {width}x{height}, scale={scale}).")

    @timeit
    def save_multi_step_sankey_diagram(self, struct_idx_list, filename, width=1200, height=800, scale=1.0):
        """
        (Multi-step version)
        Build and save a Sankey diagram showing transitions across multiple struct_idx steps.

        struct_idx_list: A list of struct indices in chronological/logical order (e.g., [0, 500, 1000]).
        filename: Output file path (e.g., 'multi_step_sankey.png').
        width, height: Dimensions of the image in pixels.
        scale: Resolution scale factor.
        """
        labels, source, target, value = self.build_multi_step_sankey_data(struct_idx_list)
        print(f'labels ({len(labels)}):', labels)
        print(f'source ({len(source)}):', source)
        print(f'target ({len(target)}):', target)
        print(f'value ({len(value)}):', value)
        # Make a nice title from the struct_idx_list
        title_str = " → ".join(str(idx) for idx in struct_idx_list)
        fig = self.create_sankey_figure(
            labels, source, target, value,
            title=f"Multi-step Transitions: {title_str}"
        )
        fig.write_image(filename, width=width, height=height, scale=scale)
        print(f"Multi-step Sankey diagram saved to {filename} (size: {width}x{height}, scale={scale}).")


@timeit
def reconstruct_total_dict(df):
    total_dict = {}
    for row in df.itertuples(index=False):
        struct_idx = int(row.struct_idx)
        global_idx = int(row.global_idx)
        state_C = row.state_C

        if struct_idx not in total_dict:
            total_dict[struct_idx] = {}
        total_dict[struct_idx][global_idx] = state_C
    return total_dict


@timeit
def load_pdf():
    return pd.read_hdf('total_dict.h5', key='df')


def main():
    # 1) Load the pickled total_dict
    df = load_pdf()
    total_dict = reconstruct_total_dict(df)
    # total_dict = {}
    # for row in df.itertuples(index=False):
    #     struct_idx = int(row.struct_idx)
    #     global_idx = int(row.global_idx)
    #     state_C = row.state_C

    #     if struct_idx not in total_dict:
    #         total_dict[struct_idx] = {}
    #     total_dict[struct_idx][global_idx] = state_C

    # 2) Create an instance of the SankeyDiagram class
    diagram = SankeyDiagram(total_dict)

    # [A] Two-step usage example (from struct_idx=0 to struct_idx=1000)
    # diagram.save_sankey_diagram(0, 1999, filename="two_step_sankey.png")

    # [B] Multi-step usage example (from struct_idx=0 to 500 to 1000)
    # struct_indices = [300*i for i in range(14)] + [3999]
    struct_indices = [300*i for i in range(14)] + [3999]
    # struct_indices = [0, 300, 600, 900, 999]

    diagram.save_multi_step_sankey_diagram(
        struct_idx_list=struct_indices,
        filename="multi_step_sankey.png",
        width=1600,
        height=900,
        scale=2.0
    )

if __name__ == "__main__":
    main()

