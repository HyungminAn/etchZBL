import os,ase,ase.io,ase.build,sys
# path_def="incidence"
matcher={1:14,2:7,3:1,4:9}
image=ase.io.read(sys.argv[1],format='lammps-data',index="0",style="atomic")
image=ase.build.sort(image,tags=image.arrays['id'])
image.set_atomic_numbers([matcher[i] for i in image.get_atomic_numbers()])
arg_sort=image.numbers.argsort()
ase.io.write(f'POSCAR',image[arg_sort],format='vasp')
