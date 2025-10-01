

#ifdef PAIR_CLASS

PairStyle(d2,PairD2)

#else

#ifndef LMP_PAIR_D2
#define LMP_PAIR_D2

#include "pair.h"

namespace LAMMPS_NS {

  class PairD2 : public Pair {
    public:
      PairD2(class LAMMPS *);
      ~PairD2() override;
      void compute(int, int) override;
      void settings(int, char **) override;
      void coeff(int, char **) override;
      // void init_style() ;
      double init_one(int, int) override;
      void write_restart(FILE *) override;
      void read_restart(FILE *) override;
      void write_restart_settings(FILE *) override;
      void read_restart_settings(FILE *) override;
      // void write_data(FILE *) ;
      // void write_data_all(FILE *) ;
      // double single(int, int, int, int, double, double, double, double &) ;
      // void *extract(const char *, int &) override;

    protected:
      double cut_global;
      double **cut;
      double s6, d0;
      double **c6, **Rvdw;

      virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
