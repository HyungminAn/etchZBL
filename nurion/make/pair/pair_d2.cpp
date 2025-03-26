/* ----------------------------------------------------------------------
   Contributing author: 
   ------------------------------------------------------------------------- */

#include "pair_d2.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "math_extra.h"
#include "force.h"

#include <set>

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace LAMMPS_NS;

#define MAXLINE 50000
#define MINR 0.0001

/* ---------------------------------------------------------------------- */
// Constructor
PairD2::PairD2(LAMMPS *lmp) : Pair(lmp) {}

/* ---------------------------------------------------------------------- */
// Destructor
PairD2::~PairD2()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(c6);
    memory->destroy(Rvdw);

  }
}

/* ---------------------------------------------------------------------- */

void PairD2::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double   evdwl;
  // double rsq, r2inv, r6inv, rij, fexp, fdmp, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    double tmpf[3]={0.0,0.0,0.0};
    
    // #pragma omp simd  private(j,delx,dely,delz,rsq,r2inv,rij,jtype,r6inv,fexp,fdmp,fpair) safelen(4)
    // #pragma omp simd  private(j,delx,dely,delz,rsq,r2inv,rij,jtype,r6inv,fexp,fdmp,fpair) reduction(+:tmpf,eng_vdwl)
    // #pragma clang  loop vectorize(enable) interleave(enable)
    #pragma vector
    #pragma ivdep
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      // factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        const double  r2inv = 1.0 / rsq;
        const double rij = sqrt(rsq);
        const double r6inv = r2inv * r2inv * r2inv;
        const double fexp = exp(-d0*(rij/Rvdw[itype][jtype]-1));
        const double fdmp=1.0/(1.0+fexp);
        const double tmpE=-s6*c6[itype][jtype]*r6inv*fdmp;

        const double fpair = tmpE*(6.0/rij-fdmp*fexp*d0/Rvdw[itype][jtype]);
        tmpf[0]+=delx/rij * fpair;
        tmpf[1]+=dely/rij * fpair;
        tmpf[2]+=delz/rij * fpair;
        // f[i][0] += delx/rij * fpair;
        // f[i][1] += dely/rij * fpair;
        // f[i][2] += delz/rij * fpair;
        f[j][0] -= delx/rij * fpair;
        f[j][1] -= dely/rij * fpair;
        f[j][2] -= delz/rij * fpair;
        // if (newton_pair || j < nlocal) {
        //   f[j][0] -= delx/rij * fpair;
        //   f[j][1] -= dely/rij * fpair;
        //   f[j][2] -= delz/rij * fpair;
        // }
        // if (eflag_global) { eng_vdwl += tmpE; } // contribute to total energy

        if (eflag) {
          evdwl = -s6*c6[itype][jtype]*r6inv *fdmp;
          // evdwl *= factor_lj;
        }


        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
    f[i][0] += tmpf[0];
    f[i][1] += tmpf[1];
    f[i][2] += tmpf[2];
  }



  if (vflag_fdotr) virial_fdotr_compute();
}


void PairD2::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(cut, n, n, "pair:cut");
  memory->create(c6, n, n, "pair:c6");
  memory->create(Rvdw, n, n, "pair:Rvdw");


}

void PairD2::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");
  cut_global = utils::numeric(FLERR,arg[0],false,lmp);
  s6 = utils::numeric(FLERR,arg[1],false,lmp);
  d0 = utils::numeric(FLERR,arg[2],false,lmp);
    if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
    }
}


void PairD2::coeff(int narg, char **arg)
{  if (narg < 4 || narg > 5) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double c6_one = utils::numeric(FLERR, arg[2], false, lmp);
  double Rvdw_one = utils::numeric(FLERR, arg[3], false, lmp);
  double cut_one = cut_global;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      c6[i][j] = c6_one;
      Rvdw[i][j] = Rvdw_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");

}


double PairD2::init_one(int i, int j)
{
  
  if (setflag[i][j] == 0) {
    c6[i][j] = sqrt(c6[i][i] * c6[j][j]);
    Rvdw[i][j] = (Rvdw[i][i]+Rvdw[j][j])/2.0;
    }

  c6[j][i]=c6[i][j];
  Rvdw[j][i]=Rvdw[i][j];
  
  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairD2::write_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairD2::read_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
   ------------------------------------------------------------------------- */

void PairD2::write_restart_settings(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
   ------------------------------------------------------------------------- */

void PairD2::read_restart_settings(FILE *fp) {}
