import os
import sys
import time
import math
import pickle

import numpy as np
from braceexpand import braceexpand
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap
from sklearn.decomposition import PCA

from get_coord_vec import CoordVecGenerator, PARAMS, timing, print_timing_summary

# Global timing storage
_time_records = {}

@timing
def parse_structure_list(filename):
    """
    Parse a file of the form:

        [class_name]
        /path/to/POSCAR_{0..9960..80}/OUTCAR :
        /another/brace_{100..500..100}/FILE :
        ...

    and return a dict: { class_name: [expanded_paths, ...], ... }
    """
    result = {}
    current = None
    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # header lines: [class]
            if line.startswith('[') and line.endswith(']'):
                current = line[1:-1]
                result[current] = []
                continue
            if current is None:
                continue

            # split off trailing ':' (and optional slice text)
            if ':' in line:
                tpl, slice_txt = line.rsplit(':', 1)
                tpl = tpl.strip()
                # if slice_txt is empty, use full slice
                slc = slice(None)
                if slice_txt.strip():
                    # e.g. "10:50:5" → slice(10,50,5)
                    slc = eval(f"slice({slice_txt})")
            else:
                tpl, slc = line, slice(None)

            # expand braces and apply slice
            expanded = list(braceexpand(tpl))
            result[current].extend(expanded[slc])

    return result

def process_tag(tag, paths, cutoff_file, cvg, params):
    """
    For a given tag, load per‐element files if they exist; compute and save
    only missing elements. Returns dict el→array.
    """
    elem_list = params.ELEM_LIST
    out_prefix = params.OUTPUT_PREFIX

    total_result = {}
    to_compute = []

    # 1) Try loading each element’s file
    for el in elem_list:
        fname = f"{out_prefix}_{tag}_{el}.txt"
        if os.path.exists(fname):
            print(f"  [LOAD] {fname}")
            total_result[el] = np.loadtxt(fname)
        else:
            to_compute.append(el)

    # 2) If none missing, done
    if len(to_compute) < len(elem_list):
        print(f"[SKIP] All elements for tag '{tag}' loaded from disk.")
        return total_result

    # 3) Otherwise compute only missing elements
    print(f"[COMPUTE] Missing elements {to_compute} for tag '{tag}'")
    # accumulator for just those el
    accum = {el: [] for el in to_compute}

    for path in paths:
        print(f"    → {path}", end="")
        try:
            result = cvg.compute(path, cutoff_file, format='vasp')
        except:
            print(f" [FAIL] {path}")
            continue
        print(" ✓")
        for el in to_compute:
            mat = result.get(el)
            if mat is not None and mat.size:
                accum[el].append(mat)

    # 4) Concatenate & save each missing element
    for el in to_compute:
        mats = accum[el]
        if not mats:
            print(f"  [WARN] No data for {el} under tag '{tag}'")
            total = np.empty((0, cvg.nbins_total))
        else:
            total = np.vstack(mats)

        total_result[el] = total
        fname = f"{out_prefix}_{tag}_{el}.txt"
        # fmt='%g' prints zeros as '0' and compacts floats
        np.savetxt(fname, total, fmt='%g')
        print(f"  [SAVE] {fname} (fmt='%g')")

        # — OR, for even smaller compressed files, you could do:
        # np.savez_compressed(f"{out_prefix}_{tag}_{el}.npz", total=total)

    return total_result


class UmapProcessor:
    @timing
    def run(self,
        result_dict,
        element,
        pca_dim=20,
        umap_neighbors=15,
        umap_min_dist=0.1,
        palette='tab10'):
        """
        For a single `element`, either load a cached embedding or:
          1) Prepare X, y, tag_names
          2) PCA-reduce → UMAP → plot
        """
        # define cache path
        cache = f"{PARAMS.OUTPUT_PREFIX}_UMAP_{element}.pkl"
        if os.path.exists(cache):
            print(f"[LOAD] cached embedding for {element}")
            with open(cache, 'rb') as f:
                y, emb, tag_names = pickle.load(f)
        else:
            X, y, tag_names = self.prepare_data(result_dict, element)
            if X is None:
                return None, None
            X_red = self.run_pca(X, pca_dim)
            emb, _ = self.run_umap(X_red, umap_neighbors, umap_min_dist)
            # cache it
            with open(cache, 'wb') as f:
                pickle.dump((y, emb, tag_names), f)
            print(f"[SAVE] cached embedding to {cache}")

        tag_dict = {tag: i for i, tag in enumerate(result_dict.keys())}
        # always plot (even if loaded)
        self.plot(y, emb, tag_names, element, pca_dim, palette, tag_dict)
        return emb, None

    @timing
    def prepare_data(self, result_dict, element):
        mats, labels, tag_names = [], [], []
        for ti, (tag, sub) in enumerate(result_dict.items()):
            mat = sub.get(element)
            if mat is None or mat.size == 0:
                continue
            mats.append(mat)
            labels.append(np.full(mat.shape[0], ti, dtype=int))
            tag_names.append(tag)

        if not mats:
            print(f"No data for element {element}")
            return None, None, None

        X = np.vstack(mats)                  # (N_total, 135)
        y = np.concatenate(labels)           # (N_total,)
        return X, y, tag_names

    @timing
    def run_pca(self, X, pca_dim):
        # 2) PCA pre-reduction
        if X.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=42)
            X_red = pca.fit_transform(X)
        else:
            X_red = X
        return X_red

    @timing
    def run_umap(self, X_red, umap_neighbors, umap_min_dist):
        reducer = umap.UMAP(
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric='euclidean',
            # random_state=42,
            random_state=None,
            n_jobs=-1,
        )
        emb = reducer.fit_transform(X_red)
        return emb, reducer

    @timing
    def plot(self, y, emb, tag_names, element, pca_dim, palette, tag_dict):
        """
        Scatter UMAP embeddings colored by tag, with discrete colors.
        """
        # # discrete colormap + norm
        # n_tags = len(tag_names)
        # cmap = plt.get_cmap(palette, n_tags)
        # norm = mcolors.BoundaryNorm(boundaries=np.arange(n_tags+1)-0.5, ncolors=n_tags)

        # fig, ax = plt.subplots(figsize=(10, 10))
        # sc = ax.scatter(
        #     emb[:, 0], emb[:, 1],
        #     c=y,
        #     cmap=cmap,
        #     norm=norm,
        #     s=1,
        #     alpha=0.3,
        # )
        # cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(n_tags))
        # cbar.ax.set_yticklabels(tag_names)

        # ax.set_title(f"UMAP of {element} environments (PCA→{pca_dim} dims)")
        # ax.set_xlabel("UMAP 1")
        # ax.set_ylabel("UMAP 2")

        # outname = f"{PARAMS.OUTPUT_PREFIX}_UMAP_{element}.png"
        # fig.tight_layout()
        # fig.savefig(outname, dpi=300)

        # -------------------------------------------------------------------

        n_tags = len(tag_names)
        cmap = plt.get_cmap(palette, n_tags)

        ncols = math.ceil(math.sqrt(n_tags))
        nrows = math.ceil(n_tags / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten()

        for ax in axes[:n_tags]:
            ax.scatter(
                emb[:, 0], emb[:, 1],
                color='lightgray',
                s=1,
                alpha=0.1,
            )

        for i, tag in enumerate(tag_names):
            ax = axes[i]
            mask = (y == tag_dict[tag])
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                color=cmap(i),
                s=2,
                alpha=0.6,
                label=tag
            )
            ax.set_title(tag)
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in axes[n_tags:]:
            fig.delaxes(ax)

        fig.suptitle(f"UMAP of {element} environments (PCA→{pca_dim} dims)")
        fig.tight_layout()

        outname = f"{PARAMS.OUTPUT_PREFIX}_UMAP_{element}.png"
        fig.savefig(outname, dpi=300)


def main():
    if len(sys.argv) != 3:
        print("Usage: load_structures.py <structure_list_file> <cutoff.npy>")
        sys.exit(1)

    struct_file = sys.argv[1]
    cutoff_file = sys.argv[2]
    mapping = parse_structure_list(struct_file)
    params = PARAMS()
    cvg = CoordVecGenerator(params)

    result = {}
    for tag, paths in mapping.items():
        print(f"\n=== Processing tag: {tag} ===")
        result_sub = process_tag(tag, paths, cutoff_file, cvg, params)
        result[tag] = result_sub

    print("\nAll tags done.")

    up = UmapProcessor()
    for elem in params.ELEM_LIST:
        t_start = time.perf_counter()
        up.run(
            result,
            element=elem,
            pca_dim=20,           # tune lower/higher
            umap_neighbors=10,
            umap_min_dist=0.05
        )
        t_end = time.perf_counter()
        print(f"UMAP for {elem} took {t_end - t_start:.2f}s")
        break

    print_timing_summary()


if __name__ == "__main__":
    main()
