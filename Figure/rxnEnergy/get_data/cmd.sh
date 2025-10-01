src_py="/data2/andynn/ZBL_modify/codes/Figure/rxnEnergy/main.py"
cwd=$(pwd)

is_nnp_relaxation_done=1
is_dft_calculation_done=1

# # If NNP relaxations are already done, link instead of running again
# for ion in CF CF3 CH2F;do
#     for energy in 20 50;do
#         src="/data2/andynn/ZBL_modify/SmallCell/02_NNP_RIE/new/${ion}/${energy}"
#         dst="${cwd}/${ion}/${energy}"
#         mkdir -p ${dst}
#         cd ${dst}
#         if (( $is_nnp_relaxation_done ));then
#             echo "Calculations for ${ion} at ${energy} eV are already done."
#             if [ -L "./01_incidences" ]; then
#                 echo "Removing old symlink ./01_incidences"
#                 rm -f ./01_incidences
#             fi
#             ln -s /data2/andynn/ZBL_modify/SmallCell/05_reaction/${ion}/${energy}/incidences ./01_incidences
#         fi
#         cd ${cwd}
#     done
# done

# # If DFT calculations are already done, link instead of running again
# for ion in CF CF3 CH2F;do
#     for energy in 20 50;do
#         dst="${cwd}/${ion}/${energy}"
#         cd ${dst}
#         if (( $is_dft_calculation_done ));then
#             if [ -L "./03_data_to_relax" ];then
#                 echo "Removing old symlink ./03_data_to_relax"
#                 rm -f ./03_data_to_relax
#             fi
#             ln -s /data2/andynn/ZBL_modify/SmallCell/05_reaction/${ion}/${energy}/post_process_bulk_gas ./03_data_to_relax
#         fi
#         cd ${cwd}
#     done
# done

# # Run
# for ion in CF CF3 CH2F;do
#     for energy in 20 50;do
#         src="/data2/andynn/ZBL_modify/SmallCell/02_NNP_RIE/new/${ion}/${energy}"
#         dst="${cwd}/${ion}/${energy}"
#         mkdir -p ${dst}
#         cd ${dst}
#         py ${src_py} ${src}
#         cd ${cwd}
#     done
# done

# Plotting
src_py_plot="/data2/andynn/ZBL_modify/codes/Figure/rxnEnergy/plotter.py"
path_yaml="./plot.yaml"
py ${src_py_plot} ${path_yaml}
