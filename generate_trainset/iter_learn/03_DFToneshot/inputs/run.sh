for i in run_{1..10};do
    cd ${i}
    qsub batch_nurion.j
    cd ..
done
