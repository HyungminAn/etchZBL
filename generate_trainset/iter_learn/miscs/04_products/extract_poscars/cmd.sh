# for i in {1..50};do
for i in 25;do
    py get_products.py ../02_NNP_RIE/CF/50/dump_${i}.lammps ${i}
done
