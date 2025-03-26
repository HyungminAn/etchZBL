def main():
    with open("rxn.dat",'r') as O:
        rxn_list=[i.split()[0].split("/") for i in O.readlines()[1:]]

    with open("gas_to_cal",'w') as O:
        for i in range(len(rxn_list)):
            structure_list=rxn_list[i][0].split(",")

            if len(structure_list)>1:
                for j in structure_list[1:]:
                    if "_" in j:
                        O.write("post_process_bulk_gas/gas/"+j.split("_")[0]+"/"+j+"\n")

            structure_list=rxn_list[i][1].split(",")
            if len(structure_list)>1:
                for j in structure_list[1:]:
                    if "_" in j:
                        O.write("post_process_bulk_gas/gas/"+j.split("_")[0]+"/"+j+"\n")


if __name__ == "__main__":
    main()
