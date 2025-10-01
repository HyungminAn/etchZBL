# Scripts used for drawing figures in the paper
## Main figures
### Figure 3
1. Data generation: `etch_yield/get_data/batch_run.py`, using the binary
   compiled from `etch_yield/get_data/fast_cpp/main.cpp` (use `compile.sh`
   within the folder). Also refer to `etch_yield/get_data/path.yaml`.
   (Or one may use `etch_yield/plot_profile.py` for generation with python.)
2. Plot `etch_yield/plot_with_experiments/plot.py`. Etch yield data should be
   written in `etch_yield/plot_with_experiments/dat.yaml`.

<div align="center">
    <img src="./etch_yield/plot_with_experiments/3_1_2_valid_etchyield.png"
    width="710" height="473"
    title="Figure 3"/>
</div>

### Figure 4
1. Generate merged structures using `atom_config/merge.py` with
   `atom_config/input.yaml`.
   (Example structures are in `atom_config/example/`)
2. Create overlay figure using `atom_config/overlay/plot.py`.

<div align="center">
    <img src="./atom_config/overlay/result.png"
    width="1348" height="941"
    title="Figure 4 (overlay)"
    style="transform: scale(0.5)"/>
</div>

### Figure 5
1. Data generation: use `height_analysis/main.py` with `height_analysis/input_SiO2.yaml`
   (usage in `height_analysis/cmd_SiO2.sh`.)
   The data used in Figure 5 are already in `height_change/SiO2/`.
2. Plot `height_change/SiO2/plot.py` using `height_change/SiO2/data_list.yaml`

<div align="center">
    <img src="./height_change/SiO2/3_1_3_valid_transient_SiO2.png"
    width="350" height="350"
    title="Figure 5"/>
</div>

### Figure 6
1. Data generation & plot: use `surface_composition/SiO2/plot.py` with
   `surface_composition/SiO2/input.yaml`. The data used in Figure 6 are already
   in `surface_composition/SiO2/`.

<div align="center">
    <img src="./surface_composition/SiO2/3_1_4_valid_surface_composition_SiO2.png"
    width="350" height="700"
    title="Figure 6"/>
</div>

### Figure 7
1. Refer to Figure 6.

<div align="center">
    <img src="./surface_composition/Si3N4/3_1_4_valid_surface_composition_Si3N4.png"
    width="350" height="700"
    title="Figure 7"/>
</div>

### Figure 8
1. Plot with `regime_overview/plot.py` using `regime_overview/data.yaml`.

<div align="center">
    <img src="./regime_overview/3_2_1_regime_overview.png"
    width="350" height="400"
    title="Figure 8"/>
</div>

### Figure 10
1. Data generation: refer to Figure 5.
2. Modify `height_analysis/main.py` to run `DataPlotterSelected` instead of
   `DataPlotter`. (`yaml` file should also be modified.)

<div align="center">
    <img src="./height_analysis/data/SiO2_CF_300/3_2_1_CF_300.png"
    width="350" height="350"
    title="Figure 10"/>
</div>

### Figure 12
1. Refer to Figure 10.

<div align="center">
    <img src="./height_analysis/data/Si3N4_CF_250/3_2_1_Si3N4_selected.png"
    width="350" height="350"
    title="Figure 12"/>
</div>

## Supplementary figures
### Figure S1
1. Run `coordNumAnalysis/plot.py` using `coordNumAnalysis/structure_list` and
   `coordNumAnalysis/cutoff_matrix.npy`.

<div align="center">
    <img src="./coordNumAnalysis/result.png"
    width="710" height="500"
    title="Figure S1"/>
</div>


### Figure S2
1. Run `train_results/parity_plot/plot.py` using
   `train_results/parity_plot/input.yaml`.

<div align="center">
    <img src="./train_results/parity_plot/result.png"
    width="800" height="800"
    title="Figure S2"/>
</div>

### Figure S3
1. Run `train_results/iter_learn/plot.py` using
   `train_results/iter_learn/input.yaml`.

<div align="center">
    <img src="./train_results/iter_learn/data/SiO2.png"
    width="400" height="880"
    title="Figure S3-(a)"/>
</div>

### Figure S4
1. Generate data using `rxnEnergy/get_data/main.py` and
   `rxnEnergy/get_data/input.yaml`. Refer to example: `rxnEnergy/get_data/cmd.sh`.
2. Plot using `rxnEnergy/plot_from_log.py`.
   (Data are in `rxnEnergy/data/`.)

<div align="center">
    <img src="./rxnEnergy/rxn_energy_metric.png"
    width="900" height="350"
    title="Figure S4"/>
</div>

### Figure S5
1. Data generation: refer to Figure 3.
2. Plot `etch_yield/plot_overview/plot_overview.py`. Paths of etch yield should
   be written in `etch_yield/plot_overview/path.yaml`.

<div align="center">
    <img src="./etch_yield/plot_overview/result.png"
    width="1065" height="1500"
    title="Figure S5"/>
</div>

### Figure S6
1. Data generation: refer to Figure 3.
2. Plot `etch_yield/plot_profile_single.py`. `etch_yield/CF_300.dat` was used.

<div align="center">
    <img src="./etch_yield/CF_300.png"
    width="350" height="350"
    title="Figure S6"/>
</div>

### Figure S7
1. Run `removal_height_check/check_removal_height.py` (refer to
   `removal_height_check/batch.j`).

<div align="center">
    <img src="./removal_height_check/removal_height.png"
    width="350" height="350"
    title="Figure S7"/>
</div>

### Figure S8
1. Data generation: refer to Figure 6.
   (Use data in `surface_composition/SiO2/CF_300/3_1_4_valid_surface_composition_SiO2.png`)

<div align="center">
    <img src="./removal_height_check/removal_height.png"
    width="350" height="700"
    title="Figure S8"/>
</div>

### Figure S9
1. Data generation: refer to Figure 5.
   (Use data in `height_change/Si3N4/`.)

<div align="center">
    <img src="./height_change/Si3N4/3_1_3_valid_transient_Si3N4.png"
    width="700" height="350"
    title="Figure S9"/>
</div>

### Figure S10
1. Data generation: `byproducts/get_data/gen_mol_dict.py`.
   (Data used are in `byproducts/get_data/data/`).
2. Plot: `byproducts/plot.py`.

<div align="center">
    <img src="byproducts/3_1_5_valid_byproduct_ratio.png"
    width="350" height="700"
    title="Figure S10"/>
</div>

### Figure S11
1. Use `height_analysis/profile_example/plot_profile.py`.

<div align="center">
    <img src="height_analysis/profile_example/figure_S11.png"
    width="525" height="710"
    title="Figure S11"/>
</div>

### Figure S12
1. Refer to Figure S11.

<div align="center">
    <img src="height_analysis/profile_example/figure_S12.png"
    width="525" height="710"
    title="Figure S12"/>
</div>

### Figure S13
1. Refer to Figure 10. (results are in
   `height_analysis/data/SiO2_CF_300/sensitivity_analysis/`.)

<div align="center">
    <img src="height_analysis/data/SiO2_CF_300/sensitivity_analysis/3_2_1_CF_300.png"
    width="350" height="700"
    title="Figure S13"/>
</div>

### Figure S14
1. Refer to Figure 5. (results are in `height_analysis/data/SiO2_total/`.)

<div align="center">
    <img src="height_analysis/data/SiO2_total/3_2_1_height_total_SiO2.png"
    width="710" height="639"
    title="Figure S14"/>
</div>

### Figure S15
1. Refer to Figure S10. (results are in `byproducts/get_data/data/SiO2/`.)

<div align="center">
    <img src="byproducts/get_data/data/SiO2/byproducts.png"
    width="1088" height="1088"
    title="Figure S15"/>
</div>

### Figure S16
1. Refer to Figure 5. (results are in `height_analysis/data/SiO2_CH2F_250/`.)

<div align="center">
    <img src="height_analysis/data/SiO2_CH2F_250/3_2_1_CF_CH2F_compare.png"
    width="350" height="350"
    title="Figure S16"/>
</div>

### Figure S17
1. Refer to Figure 5. (results are in `height_analysis/data/Si3N4_total/`.)

<div align="center">
    <img src="height_analysis/data/Si3N4_total/3_2_1_height_total_Si3N4.png"
    width="710" height="639"
    title="Figure S17"/>
</div>

### Figure S18
1. Refer to Figure S10. (results are in `byproducts/get_data/data/Si3N4/`.)

<div align="center">
    <img src="byproducts/get_data/data/Si3N4/byproducts.png"
    width="1088" height="1088"
    title="Figure S18"/>
</div>

### Figure S19
1. Refer to Figure 5. (results are in `height_analysis/data/Si3N4_CF2_CF3_250/`.)

<div align="center">
    <img src="height_analysis/data/Si3N4_CF2_CF3_250/3_2_2_F_effect.png"
    width="350" height="350"
    title="Figure S19"/>
</div>

### Figure S20
1. Refer to Figure 5. (results are in `height_analysis/data/Si3N4_CF_500_750_1000/`.)

<div align="center">
    <img src="height_analysis/data/Si3N4_CF_500_750_1000/3_2_2_energy_effect.png"
    width="350" height="350"
    title="Figure S20"/>
</div>

### Figure S21
1. Refer to Figure S10. (results are in `byproducts/statistics/`.)

<div align="center">
    <img src="byproducts/statistics/result_byproduct_stats.png"
    width="600" height="950"
    title="Figure S21"/>
</div>

### Figure S22
1. Refer to Figure 5. (results are in `height_analysis/data/Si3N4_CF3_1000/`.)

<div align="center">
    <img src="height_analysis/data/Si3N4_CF3_1000/CF3_1000eV_longer.png"
    width="350" height="350"
    title="Figure S22"/>
</div>

### Figure S23
1. Plot using `sputter_analysis/plot.py`.

<div align="center">
    <img src="sputter_analysis/result.png"
    width="710" height="350"
    title="Figure S23"/>
</div>

### Figure S24
1. Refer to Figure 5. (results are in `height_analysis/data/Si3N4_CH2F_250/`.)

<div align="center">
    <img src="height_analysis/data/Si3N4_CH2F_250/3_2_2_H_effect.png"
    width="350" height="350"
    title="Figure S24"/>
</div>

# Other scripts
See `etc/README.md`.
