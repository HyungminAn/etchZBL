get_ion_mass() {
    local ion=$1

    local mass_C=12.011
    local mass_H=1.0089
    local mass_F=18.9984

    if [ ${ion} = "CF" ];then
        local ion_mass="${mass_C}+${mass_F}"
    elif [ ${ion} = "CF2" ];then
        local ion_mass="${mass_C}+2*${mass_F}"
    elif [ ${ion} = "CF3" ];then
        local ion_mass="${mass_C}+3*${mass_F}"
    elif [ ${ion} = "CHF" ];then
        local ion_mass="${mass_C}+${mass_H}+${mass_F}"
    elif [ ${ion} = "CHF2" ];then
        local ion_mass="${mass_C}+${mass_H}+2*${mass_F}"
    elif [ ${ion} = "CH2F" ];then
        local ion_mass="${mass_C}+2*${mass_H}+${mass_F}"
    fi

    local ion_mass=$(echo ${ion_mass} | bc)
    echo ${ion_mass}
}

main() {
    src_mol="/data2/andynn/ZBL_modify/zbl_chgTot/01_NNP_gen_slab/03_molecule_relax"
    src_files="/data2/andynn/ZBL_modify/zbl_chgTot/02_NNP_RIE"
    lmp_in="lammps_iterative_SiOCHF.in"

    # for mol_name in CF CF3 CH2F CHF2;do
    for mol_name in CF CF3 CH2F;do
        for ion_E in 20 50;do
            dst="${mol_name}/${ion_E}"
            mkdir -p ${dst}

            cp ${src_mol}/mol_${mol_name} ${dst}/

            for file in input.data;do
                cp ${src_files}/${file} ${dst}/
            done

            sed "s/JOBTITLE/small_${mol_name}_${ion_E}/g" ${src_files}/batch.j > ${dst}/batch.j

            ion_mass=$(get_ion_mass ${mol_name})

            sed "
                /PATH_MOL_ION/s/path_my_mol_path/mol_${mol_name}/g;
                /ION_MASS/s/my_ion_mass/${ion_mass}/g;
                /ION_KE/s/my_ion_kE/${ion_E}/g" ${src_files}/${lmp_in} > ${dst}/${lmp_in}
        done
    done
}

main
