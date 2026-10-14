"""
PostProcessor: per-iteration structure post-processing pipeline.

Replaces the former StrProcess class. Each instance handles exactly one
MD iteration: load → remove byproducts → modify slab → save.
"""
import os

import ase

from .io import save_atom, load_atom, expand_box, make_convert_dict
from .graph import get_builder
from .byproduct import ByProductRemover
from .slab import get_modifier
from .slab._core import get_slab_height


def _default_fname_in(it: int) -> str:
    return f'str_shoot_{it}.coo'


def _default_fname_out(it: int) -> str:
    return f'str_shoot_{it}_after_mod.coo'


class PostProcessor:
    """
    Encapsulates all post-MD processing for a single ion incidence iteration.

    Shared resources (bulk, byproduct remover, graph builder, cutoff matrix)
    are injected at construction time to avoid reloading them on every iteration.
    """

    def __init__(
        self,
        iteration: int,
        config: dict,
        bulk: ase.Atoms,
        cutoff_matrix,
        byproduct_remover: ByProductRemover,
        graph_builder,
        logger=None,
    ) -> None:
        self.iteration = iteration
        self.config = config
        self.bulk = bulk
        self.cutoff_matrix = cutoff_matrix
        self.byproduct_remover = byproduct_remover
        self.graph_builder = graph_builder
        self.logger = logger

        ep = config['etch_params']
        rp = config['run_params']
        self.elmlist: list = config['pot_params']['elmlist']
        self.convert_dict: dict = make_convert_dict(self.elmlist)
        self.etch_params = ep
        self.box_height: float = rp['box_height']
        self.slab_update_iteration: int = rp['slab_update_iteration']

        self.fname_in = _default_fname_in
        self.fname_out = _default_fname_out
        self.name_in: str = self.fname_in(iteration)
        self.name_out: str = self.fname_out(iteration)

        self.str_in: 'ase.Atoms | None' = None
        self.str_out: 'ase.Atoms | None' = None
        self.clusters_in: 'list[set] | None' = None
        self.byproduct_removed: bool = False

    def calc_done(self) -> bool:
        return os.path.exists(self.name_out)

    def has_input(self) -> bool:
        return os.path.exists(self.name_in)

    def load(self) -> None:
        if os.path.exists(self.name_in) and self.str_in is None:
            self.str_in = load_atom(self.name_in, self.convert_dict)

        if os.path.exists(self.name_out) and self.str_out is None:
            self.str_out = load_atom(self.name_out, self.convert_dict)

        if self.str_in is not None and self.str_out is None:
            self.str_out = expand_box(self.str_in, self.box_height)

    def build_graph(self) -> None:
        assert self.str_in is not None, f'str_in not loaded for iteration {self.iteration}'
        self.clusters_in = self.graph_builder.build(self.str_in, self.cutoff_matrix)

    def remove_byproduct(self) -> None:
        if self.clusters_in is None:
            self.build_graph()
        self.byproduct_remover.run(self)

    def modify_slab(self) -> bool:
        """Apply the configured slab modification strategy.
        Returns True if annealing is needed."""
        if self.clusters_in is None:
            self.build_graph()
        strategy = self.config['run_params']['slab_modify_crit']
        modifier = get_modifier(strategy)
        return modifier.run(self)

    def save(self) -> None:
        if self.calc_done():
            if self.logger:
                self.logger.print(f'{self.name_out} already exists')
            return
        assert self.str_out is not None, 'str_out is None; cannot save'
        save_atom(self.str_out, self.name_out, self.elmlist)
        if self.logger:
            self.logger.print(f'{self.name_out} written')

    def slab_height(self) -> 'float | None':
        """Return slab height from str_out clusters, or None if unavailable."""
        if self.str_out is None:
            return None
        clusters = self.graph_builder.build(self.str_out, self.cutoff_matrix)
        return get_slab_height(clusters, self.str_out)
