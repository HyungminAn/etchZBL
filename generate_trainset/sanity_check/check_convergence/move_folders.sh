path=`echo ./01_bulk/01_elastic_deformation/{01..04}*/`
path="${path} `echo ./01_bulk/02_supercell_MD/{01..02}*/`"
path="${path} `echo ./02_bulkmqa/01_SiO2/{melt,quench,anneal}/`"
path="${path} `echo ./03_gas/01_diatomic/O{O,C,Si,H,F}/`"
path="${path} `echo ./03_gas/02_polyatomic/{COH,CO2,CHFO,OHF,CH2O,COF2,COF,OF2,H2O}`"
path="${path} `echo ./04_interstitial/{01,02}*/{01..05}*/`"
path="${path} `echo ./05_vacancy/{01,02}*/{01,02}*/{01,02}*/`"
path="${path} `echo ./06_vts/{01,02}*/{01..04}*/{melt,quench,anneal}/`"
path="${path} `echo ./07_slab/01_SiO2/02_SlabMelt/md/{anneal,melt,liquid}/`"
path="${path} `echo ./07_slab/01_SiO2/01_SlabMD/{01,02}*/{01,02}*/`"
path="${path} `echo ./08_channeling/{01,02}*/`"

for i in ${path};do
    if [ -e ${i}/poscars_continue ];then
        for j in ${i}/poscars_continue/POSCAR_*;do
            folder_name=`echo ${j} | xargs -n 1 basename`
            if [ -e ${i}/${folder_name} ];then
                src="${i}/${folder_name}"
                dst="${i}/outcar_undone"
                mkdir -p ${dst}
                mv ${src} ${dst}
                echo "${i}/${folder_name} removed. (Undone)"
            fi
        done
    fi

    if [ -e ${i}/poscars_unconverged ];then
        for j in ${i}/poscars_unconverged/POSCAR_*;do
            folder_name=`echo ${j} | xargs -n 1 basename`
            if [ -e ${i}/${folder_name} ];then
                src="${i}/${folder_name}"
                dst="${i}/outcar_unconverged"
                mkdir -p ${dst}
                mv ${src} ${dst}
                echo "${i}/${folder_name} removed. (Unconverged)"
            fi
        done
    fi
done
