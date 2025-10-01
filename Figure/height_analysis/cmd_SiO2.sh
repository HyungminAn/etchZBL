# py main.py input_total.yaml SiO2
py main.py input_CF_300.yaml SiO2
for fmt in eps pdf png;do
    mv 3_2_1_SiO2_selected.${fmt} 3_2_1_CF_300.${fmt}
done
py main.py input_compare.yaml SiO2
for fmt in eps pdf png;do
    mv 3_2_1_SiO2_selected.${fmt} 3_2_1_CF_CH2F_compare.${fmt}
done
