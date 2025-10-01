src="/data2/andynn/ZBL_modify/SmallCell/05_reaction"
path_sh="/data2/andynn/ZBL_modify/codes/SmallCell/05_reactionE/codes/dft_in/make_folders.sh"
cwd=$(pwd)
for i in $(ls -d ${src}/C*/*/);do
    ion_E=$(basename $i)
    ion_name=$(basename $(dirname $i))
    dst="./${ion_name}/${ion_E}"
    echo ${dst}
    mkdir -p ${dst}
    cd ${dst}
    sh ${path_sh} ${i}/post_process_bulk_gas
    cd ${cwd}
done
