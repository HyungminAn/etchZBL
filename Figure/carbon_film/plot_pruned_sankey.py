import time
from collections import Counter

import pandas as pd
import numpy as np
from functools import wraps

import plotly.graph_objects as go
import plotly.express as px


def timeit(function):
    '''
    Wrapper function to measure the execution time of a function.
    '''
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(f'{function.__name__:40s} took {end - start:10.4f} seconds')
        return result
    return wrapper


class SankeyDiagram:
    def __init__(self, total_dict):
        self.total_dict = total_dict

    def compute_transitions(self, struct_idx0, struct_idx1):
        """ 기존 2-step transition 계산 """
        transitions = {}
        if struct_idx0 not in self.total_dict or struct_idx1 not in self.total_dict:
            return transitions
        g_common = set(self.total_dict[struct_idx0].keys()).intersection(self.total_dict[struct_idx1].keys())
        for g_idx in g_common:
            s0 = self.total_dict[struct_idx0][g_idx]
            s1 = self.total_dict[struct_idx1][g_idx]
            transitions[(s0, s1)] = transitions.get((s0, s1), 0) + 1
        return transitions

    # --------------------------------------------------
    # 1) PRUNED_DICT 생성: make_pruned_total_dict
    # --------------------------------------------------
    def make_pruned_total_dict(self, struct_idx_list):
        """
        Pruned 로직 예시:
         - 처음 NOT_CREATED가 아닌 시점부터 추적
         - BYPRODUCT 상태가 되면 그 순간까지만 추적
         - 끝까지 NOT_CREATED면 아예 제외
        """
        states_per_global = {}
        for s_idx in struct_idx_list:
            if s_idx not in self.total_dict:
                continue
            for g_idx, st in self.total_dict[s_idx].items():
                if g_idx not in states_per_global:
                    states_per_global[g_idx] = []
                states_per_global[g_idx].append((s_idx, st))

        pruned_total_dict = {s_idx:{} for s_idx in struct_idx_list}

        for g_idx, step_state_list in states_per_global.items():
            # step 순으로 정렬
            step_state_list.sort(key=lambda x: x[0])

            # 끝까지 NOT_CREATED면 제외
            all_states = [st for (_, st) in step_state_list]
            if all_states.count("NOT_CREATED") == len(all_states):
                continue

            # # 처음 NOT_CREATED 아닌 시점
            # start_idx = 0
            # for i, (stp, st) in enumerate(step_state_list):
            #     if st != "NOT_CREATED":
            #         start_idx = i
            #         break

            # (기존) 처음 NOT_CREATED가 아닌 시점부터 추적
            # (수정) '처음으로 NOT_CREATED가 아닌 시점' 직전 스텝도 포함
            first_non_not_idx = -1
            for i, (stp, st) in enumerate(step_state_list):
                if st != "NOT_CREATED":
                    first_non_not_idx = i
                    break
            if first_non_not_idx < 0:
                # 실제로는 위에서 걸러졌지만 안전상
                continue

            # start_idx = first_non_not_idx
            # 마지막으로 NOT_CREATED였던 시점을 포함하기 위해
            # (한 스텝 앞) => first_non_not_idx - 1
            # 단, 음수가 되지 않도록 max(...,0)
            start_idx = max(first_non_not_idx - 1, 0)

            # BYPRODUCT가 나오면 그 시점까지만 추적
            end_idx = len(step_state_list)
            END_LABEL_LIST = ["BYPRODUCT",
                              "REFLECTED",
                              "REMOVED_DURING_MD"]
            for i in range(start_idx, len(step_state_list)):
                if step_state_list[i][1] in END_LABEL_LIST:
                    end_idx = i+1
                    break

            active_range = step_state_list[start_idx:end_idx]
            for (stp, st) in active_range:
                pruned_total_dict[stp][g_idx] = st

        return pruned_total_dict

    # --------------------------------------------------
    # 2) pruned_dict에 대해 build + (x,y) 배치
    # --------------------------------------------------
    def build_multi_step_sankey_data_with_xy_from_dict(self, dict_data, struct_idx_list):
        """
        주어진 dict_data(=pruned_total_dict 등)를 사용하여,
        Sankey용 (labels, source, target, value) + (node_x, node_y) 구성.
        """
        # 먼저 transitions를 계산할 헬퍼 함수
        def compute_transitions_from_dict(dict_data, s0, s1):
            trans = {}
            if s0 not in dict_data or s1 not in dict_data:
                return trans
            g_common = set(dict_data[s0].keys()).intersection(dict_data[s1].keys())
            for g_idx in g_common:
                st0 = dict_data[s0][g_idx]
                st1 = dict_data[s1][g_idx]
                trans[(st0, st1)] = trans.get((st0, st1), 0) + 1
            return trans

        # (1) 각 step별 unique state 모으기
        step_label_lists = []
        offsets = []
        total_labels = []
        current_offset = 0

        for s_idx in struct_idx_list:
            if s_idx in dict_data:
                unique_states = sorted(set(dict_data[s_idx].values()))
            else:
                unique_states = []
            step_label_lists.append(unique_states)
            offsets.append(current_offset)
            current_offset += len(unique_states)
            total_labels.extend(unique_states)

        source = []
        target = []
        value = []

        # (2) consecutive pair 연결
        for i in range(len(struct_idx_list) - 1):
            s0 = struct_idx_list[i]
            s1 = struct_idx_list[i+1]
            transitions = compute_transitions_from_dict(dict_data, s0, s1)

            label_0 = step_label_lists[i]
            label_1 = step_label_lists[i+1]
            offset_0 = offsets[i]
            offset_1 = offsets[i+1]

            for (st0, st1), cnt in transitions.items():
                if st0 not in label_0 or st1 not in label_1:
                    continue
                i0 = offset_0 + label_0.index(st0)
                i1 = offset_1 + label_1.index(st1)
                source.append(i0)
                target.append(i1)
                value.append(cnt)

        # ---------------------------------------------
        # A) label별 "최대 사용량" 계산
        # ---------------------------------------------
        #   dict_data[s_idx][g_idx] = state
        #   => step i에서 state별 개수 카운팅

        label_max_usage = {}
        # 전체 label 집합 수집
        all_labels_set = set()
        for s_idx in struct_idx_list:
            if s_idx in dict_data:
                all_labels_set.update(dict_data[s_idx].values())

        # 초기화
        for lb in all_labels_set:
            label_max_usage[lb] = 0

        # 각 step에서 state별 개수를 세어, label_max_usage를 갱신
        for s_idx in struct_idx_list:
            if s_idx not in dict_data:
                continue
            step_counter = Counter(dict_data[s_idx].values())  # {state: count}
            for lb, cnt in step_counter.items():
                if cnt > label_max_usage[lb]:
                    label_max_usage[lb] = cnt

        # ---------------------------------------------
        # B) label 전역 순서 (fixed_order + 나머지)
        # ---------------------------------------------
        # 예: fixed_order 우선 배치, 그 뒤 label_max_usage 내림차순
        fixed_order = ["NOT_CREATED",
                       "BYPRODUCT",
                       "REFLECTED",
                       "REMOVED_DURING_MD",
                       "Fluorocarbon"]
        # 나머지 label은 max usage 기준 내림차순
        # (혹은 알파벳 순, 등등 원하는 방식)
        other_labels = sorted(
            [lb for lb in all_labels_set if lb not in fixed_order],
            key=lambda x: label_max_usage[x],
            reverse=True
        )

        final_label_order = []
        for fo in fixed_order:
            if fo in all_labels_set:
                final_label_order.append(fo)
        final_label_order.extend(other_labels)

        # label2slot: label -> index
        label2slot = {}
        for i, lb in enumerate(final_label_order):
            label2slot[lb] = i
        label2slot = self.compute_label_positions(label2slot,
                                                  label_max_usage,
                                                  scale=1.0)
        total_slots = len(label2slot)

        # ---------------------------------------------
        # C) node_counts (step i, label j)
        # ---------------------------------------------
        node_counts = [0]*len(total_labels)
        for i, s_idx in enumerate(struct_idx_list):
            if s_idx not in dict_data:
                continue
            step_counter = Counter(dict_data[s_idx].values())
            for j, st_label in enumerate(step_label_lists[i]):
                idx_in_total = offsets[i] + j
                node_counts[idx_in_total] = step_counter[st_label]

        # (3) 노드 x,y 배치
        node_x = [0]*len(total_labels)
        node_y = [0]*len(total_labels)
        num_steps = len(struct_idx_list)

        # ---------------------------------------------
        # D) 노드 배치 (x=step, y= label2slot)
        # ---------------------------------------------
        #   => 각 label은 항상 같은 y 위치!
        for step_idx in range(num_steps):
            step_x = (step_idx+1) / (num_steps+2) if num_steps>1 else 0.5
            labels_in_step = step_label_lists[step_idx]
            for j, st_label in enumerate(labels_in_step):
                idx_in_total = offsets[step_idx] + j
                node_x[idx_in_total] = step_x
                node_y[idx_in_total] = label2slot[st_label]

        return total_labels, source, target, value, node_x, node_y, node_counts

    # --------------------------------------------------
    # 3) 실제 figure 생성 (fixed) 함수
    # --------------------------------------------------
    def create_sankey_figure_fixed(self, labels, source, target, value, node_x, node_y,
                                   node_counts, title="Pruned Sankey"):

        node_colors = self.get_node_colors(labels)
        edge_colors = self.get_edge_colors(labels, source, target)

        node_labels = []
        LABEL_REMAP = {
                'NOT_CREATED': 'NC',
                'REMOVED_DURING_MD': 'RMD',
                'REFLECTED': 'REF',
                'Fluorocarbon': 'FC',
                'SiC_cluster': 'SiC',
                'BYPRODUCT': 'BP',
                }
        LABEL_COUNT = {}
        for i, lab in enumerate(labels):
            lab = LABEL_REMAP.get(lab, lab)
            if lab not in LABEL_COUNT:
                lab_final = f"{lab} ({node_counts[i]})"
                LABEL_COUNT[lab] = True
            else:
                lab_final = f"({node_counts[i]})"
            node_labels.append(lab_final)

        # link label(tooltip) 등은 필요하면 추가
        link_labels = [str(v) for v in value]

        fig = go.Figure(data=[go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=15,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                x=node_x,
                y=node_y
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

    # --------------------------------------------------
    # 4) 최종적으로 PRUNED + FIXED Sankey를 그려 저장하는 함수
    # --------------------------------------------------
    @timeit
    def save_fixed_pruned_multi_step_sankey_diagram(self, struct_idx_list,
                                                    filename="fixed_pruned_sankey",
                                                    width=1200, height=800, scale=1.0):
        """
        1) pruned_total_dict를 만든 뒤
        2) pruned_total_dict를 가지고 (x,y) 포함 Sankey 데이터 생성
        3) figure 만들고 저장
        """
        # 1) pruned_dict 생성
        pruned_dict = self.make_pruned_total_dict(struct_idx_list)

        # 2) (labels, source, target, value, x, y) 생성
        labels, source, target, value, node_x, node_y, node_counts = \
            self.build_multi_step_sankey_data_with_xy_from_dict(pruned_dict, struct_idx_list)

        # 3) figure 생성 & 저장
        title_str = " → ".join(map(str, struct_idx_list))
        fig = self.create_sankey_figure_fixed(
            labels, source, target, value, node_x, node_y,
            node_counts,
            title=f"PRUNED Sankey (Fixed Layout): {title_str}"
        )
        fig.write_image(f'{filename}.png', width=width, height=height, scale=scale)
        print(f"Saved pruned+fixed sankey to {filename}.png")
        fig.write_image(f'{filename}.pdf', width=width, height=height, scale=scale)
        print(f"Saved pruned+fixed sankey to {filename}.pdf")
        # fig.write_image(f'{filename}.eps', width=width, height=height, scale=scale)
        # print(f"Saved pruned+fixed sankey to {filename}.eps")

    # 이하, node/edge 색상 함수 등(기존 코드 재사용)...
    def get_node_colors(self, labels):
        preferred_colors = {
            'NOT_CREATED': 'white',
            'C3': 'red',
            'SiC_cluster': 'yellow',
            'C2': 'blue',
            'Fluorocarbon': 'green',
            'BYPRODUCT': 'black',
        }
        default_color = 'lightgray'
        node_colors = []
        for lab in labels:
            if lab in preferred_colors:
                node_colors.append(preferred_colors[lab])
            else:
                node_colors.append(default_color)
        return node_colors

    def get_edge_colors(self, labels, source, target):
        link_colors = []
        COLOR_DICT = {
                # 'C3': 'rgba(100, 100, 100, 0.2)'
                'SiC_cluster': 'rgba(100, 100, 100, 0.2)'
                }
        COL_WHITE = 'rgba(255, 255, 255, 0)'
        for (s, t) in zip(source, target):
            lab_s = labels[s]
            lab_t = labels[t]
            if lab_s == lab_t:
                link_colors.append(COL_WHITE)
            elif lab_s in COLOR_DICT:
                link_colors.append(COLOR_DICT[lab_s])
            elif lab_t in COLOR_DICT:
                link_colors.append(COLOR_DICT[lab_t])
            else:
                link_colors.append(COL_WHITE)
        return link_colors

    def get_labels_with_numbers(self, labels, source, value):
        node_out_flow = [0]*len(labels)
        for s_idx, v in zip(source, value):
            node_out_flow[s_idx] += v

        node_labels = []
        for i, lab in enumerate(labels):
            node_labels.append(f"{lab} ({node_out_flow[i]})")

        link_labels = [str(v) for v in value]
        return node_labels, link_labels

    @staticmethod
    def compute_label_positions(label2slot, label_max_usage, scale=1.0):
        labels = list(label2slot.keys())
        values = np.array([label_max_usage[l] for l in labels], dtype=float)

        # Prevent division by zero
        values = np.maximum(values, 1e-8)

        # Apply scaling if needed
        values = np.log1p(values)  # log(1 + x) for stability
        values = values ** scale

        # Normalize lengths
        lengths = values / values.sum()

        # Compute cumulative centers
        centers = np.zeros_like(lengths)
        start = 0.0
        for i in range(len(lengths)):
            centers[i] = start + lengths[i] / 2
            start += lengths[i]

        # Rescale to [0.05, 0.95]
        centers = (centers - centers[0]) / (centers[-1] - centers[0]) * 0.9 + 0.05

        return dict(zip(labels, centers))

@timeit
def get_total_dict():
    df = pd.read_hdf('total_dict.h5', key='df')
    total_dict = {}
    for row in df.itertuples(index=False):
        s_idx = int(row.struct_idx)
        g_idx = int(row.global_idx)
        state_C = row.state_C
        if s_idx not in total_dict:
            total_dict[s_idx] = {}
        total_dict[s_idx][g_idx] = state_C
    return total_dict

def main():

    # 1) total_dict 로드
    total_dict = get_total_dict()

    # 2) SankeyDiagram 인스턴스 생성
    diagram = SankeyDiagram(total_dict)

    # 3) 스텝 리스트 설정
    # struct_indices = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900,]
    struct_indices = [i for i in range(0, 9001, 300)]

    # 4) PRUNED + FIXED Sankey 생성
    diagram.save_fixed_pruned_multi_step_sankey_diagram(
        struct_idx_list=struct_indices,
        filename="fixed_pruned_sankey",
        width=1600,
        height=900,
        scale=2.0
    )

if __name__ == "__main__":
    main()
