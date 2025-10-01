for src in $(ls -d data/*/*/);do
    ion=$(basename $(dirname $src))
    ion_E=$(basename $src)
    name="${ion}_${ion_E}"

    args=""
    for file in $(ls $src/log.txt_* | sort -V);do
        args="$args $file"
    done
    if [ -f ${src}/log.txt ];then
        args="$args ${src}/log.txt"
    fi

    py gen_mol_dict.py ${name} ${args}
    echo ${name}
done
