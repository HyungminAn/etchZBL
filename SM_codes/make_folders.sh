ions=( "CF3" "CF" "CHF2" )
energies=( "250 500 1000" )
celltypes=( "Large" "Normal" )
update_iter=10

src_inputyaml="/scratch/x3045a07/etchZBL/LargeCell/inputs/simplenn/Si3N4/03_md/input.yaml"
path_pot="/scratch/x3045a07/etchZBL/pot/SiN/pot_SiNCHF_old"
path_lmp="/scratch/x3045a07/lammps/build_avx2/lmp"

cwd=$(pwd)

for cell in ${celltypes[@]};do

    if [ ${cell} == "Large" ];then
        path_bulk="/scratch/x3045a07/etchZBL/LargeCell/inputs/simplenn/Si3N4/01_bulk/336cell/FINAL.coo"
        path_slab="/scratch/x3045a07/etchZBL/LargeCell/inputs/simplenn/Si3N4/02_slab/336cell/relaxed.coo"
        nshoot=9000
        slab_max_height=80
        slab_min_height=60
        slab_center_height=70
        box_height=180
        incident_height=110
        evapheight=115
    else
        path_bulk="/scratch/x3045a07/etchZBL/LargeCell/inputs/simplenn/Si3N4/01_bulk/223cell/FINAL.coo"
        path_slab="/scratch/x3045a07/etchZBL/LargeCell/inputs/simplenn/Si3N4/02_slab/223cell/relaxed.coo"
        nshoot=4000
        slab_max_height=50
        slab_min_height=30
        slab_center_height=40
        box_height=120
        incident_height=80
        evapheight=85
    fi

    for ion in ${ions[@]};do
        for energy in ${energies[@]};do
            dst="${ion}/${energy}eV"        
            if [ ${cell} == "Large" ];then
                dst="${dst}_Large"
            fi
            mkdir -p ${dst}

            # copy input.yaml
            sed "
                /  potloc/   s|  potloc: .*$|  potloc: ${path_pot}|g;
                /  ion/      s|  ion: .*$|  ion: ${ion}|g;
                /  energy/   s|  energy: .*$|  energy: ${energy}|g;
                /  bulk_loc/ s|  bulk_loc: .*$|  bulk_loc: ${path_bulk}|g;
                /  slab_loc/ s|  slab_loc: .*$|  slab_loc: ${path_slab}|g;
                /  lmp_loc/  s|  lmp_loc: .*$|  lmp_loc: ${path_lmp}|g;
                /  slab_update_iteration/ s|1|${update_iter}|g;
                /  nshoot/   s|  nshoot: .*$|  nshoot: ${nshoot}|g;
                /  slab_max_height/ s|  slab_max_height: .*$|  slab_max_height: ${slab_max_height}|g;
                /  slab_min_height/ s|  slab_min_height: .*$|  slab_min_height: ${slab_min_height}|g;
                /  slab_center_height/ s|  slab_center_height: .*$|  slab_center_height: ${slab_center_height}|g;
                /  box_height/ s|  box_height: .*$|  box_height: ${box_height}|g;
                /  incident_height/ s|  incident_height: .*$|  incident_height: ${incident_height}|g;
                /  evapheight/ s|  evapheight: .*$|  evapheight: ${evapheight}|g;
                " ${src_inputyaml} > ${dst}/input.yaml

            # copy batch.j
            sed "s/JOBTITLE/${ion}_${energy}_${cell}/g" batch.j > ${dst}/batch.j

            # copy str_shoot_0.coo
            cp ${path_slab} ${dst}/str_shoot_0.coo

            echo "${ion} ${energy} ${cell} Done"

            cd ${dst}
            qsub batch.j
            cd ${cwd}
        done  # energy
    done  # ion
done  # cell
