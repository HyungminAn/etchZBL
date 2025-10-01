file_list=(
  # "/data_etch/data_HM/nurion/set_1/CF_100_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_2/CF_250_coo.tar.gz"
  "/data_etch/data_HM/nurion/set_3/CF_300_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_3/CF_500_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CF_750_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CF_1000_coo.tar.gz"

  # "/data_etch/data_HM/nurion/set_3/CF2_25_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_2/CF2_50_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_2/CF2_100_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_1/CF2_250_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_1/CF2_500_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_3/CF2_750_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CF2_1000_coo.tar.gz"

  # "/data_etch/data_HM/nurion/set_2/CF3_10_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_2/CF3_25_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_2/CF3_50_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_1/CF3_100_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_1/CF3_250_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_1/CF3_500_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_3/CF3_750_coo.tar.gz"
  # "/data_etch/data_HM/nurion/set_3/CF3_1000_coo.tar.gz"

  # "/data2_1/andynn/Etch/data_nurion/set_3/CH2F_250_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_3/CH2F_500_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CH2F_750_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CH2F_1000_coo.tar.gz"

  # "/data_etch/data_HM/nurion/set_2/CHF2_250_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_3/CHF2_500_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_3/CHF2_750_coo.tar.gz"
  # "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CHF2_1000_coo.tar.gz"
)

# file_list=(
#   "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CF_750_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CF_1000_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CF2_1000_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CH2F_750_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CH2F_1000_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_uncomplete/CHF2_1000_log.tar.gz"
# )

# file_list=(
#   "/data_etch/data_HM/nurion/set_3/CF_300_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_3/CH2F_250_log.tar.gz"
#   "/data2_1/andynn/Etch/data_nurion/set_3/CH2F_500_log.tar.gz"
#     )


for file in ${file_list[@]};do
    sh mount.sh ${file}
done
