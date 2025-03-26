import os
import copy

import ase
import ase.io
import ase.build

from PlasmaEtchSimulator.lmp.etch import LAMMPSetch


class LAMMPSanneal(LAMMPSetch):
    """This class run annearling process from slab to use etching simulation"""
    def __init__(self,
                 pot_params : dict,
                 etch_params : dict,
                 etc_params : dict,
                 bulk : str,
                 slab_height : float,
                 logger = None) -> None:
        super().__init__(pot_params,
                         etch_params,
                         etc_params,
                         logger)
        self.lmpname    = 'lammps_slab.in'
        self.slabname   = 'slab.data'
        self.bulk       = bulk
        self.slab_height = slab_height

    def _slab_header(self):
        body = f'''
thermo_style    custom   step pe temp ke etotal fmax press cpu tpcpu spcpu
thermo          10

region      rMove       block   EDGE EDGE   EDGE EDGE EDGE EDGE
group       gMove       region  rMove

region  rFixed  block   INF INF INF INF 0.0 {self.fix}
group gBottom region rFixed
velocity gBottom set 0.0 0.0 0.0
fix freeze_bottom gBottom setforce 0.0 0.0 0.0

fix         NVT_anneal      gMove    nvt     temp    {self.slab_temperature} {self.slab_temperature} 0.1
dump        dump_anneal     all custom/gz 10 dump_slab.gz &
                             id type element xu yu zu fx fy fz
dump_modify     dump_anneal     sort id     element Si N C H F  time yes format line "%d %d %s %.2f %.2f %.2f %.2f %.2f %.2f"
run         5000
unfix       NVT_anneal
undump      dump_anneal
write_data  ${{strout}}
'''
        return body

    def generate_lammps_input(self, lmpdir):
        location = lmpdir +'/'+self.lmpname
        header = self._pot_header()
        body   = self._slab_header()

        outlines = header + '\n' + body
        with open(location, 'w') as f:
            f.write(outlines)
            self.lmploc = location
            self.print(f"LAMMPS input for annealing {location} written")

    ### This class has additional function which create the slab !
    def initial_run(self,lmpdir : str) -> None:
        slabloc = lmpdir+'/'+self.slabname
        if os.path.exists(slabloc):
            self.print(f"Slab file {slabloc} already exists")
            return

        # Gen slab to anneal
        if 'OUTCAR' in self.bulk:
            atom = ase.io.read(self.bulk, format='vasp-out')
        else:
            atom = ase.io.read(self.bulk, format='lammps-data')

        ## Create
        oatom = copy.deepcopy(atom)

        ## Add vaccum and upshift from the params
        cell    = atom.cell
        pos     = atom.positions
        for i in range(3):
            pos[:,i] = pos[:,i] % cell[i,i]
        pos[:,2]     += self.upshift
        cell[2,2] += self.vaccum

        oatom.positions  = pos
        oatom.cell       = cell
        oatom = ase.build.sort(oatom)
        ase.io.write(slabloc, oatom, format='lammps-data', specorder = self.elmlist,)
        self.print(f"Slab file {slabloc} written")
