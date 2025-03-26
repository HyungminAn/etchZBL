import numpy as np
import os
with open("rxn.dat",'r') as O:
    rxn_list=[i.split()[0].split("/") for i in O.readlines()[1:]]
trivial_list=os.listdir("post_process_bulk_gas/gas/trivial")

rxn_E=np.zeros((len(rxn_list),2)) #ref_rlx_E, NNP_rlx_E
for i in range(len(rxn_list)):
    rxn_list[i][0]=rxn_list[i][0].split(",")
    rxn_list[i][1]=rxn_list[i][1].split(",")
    

    #ref_rlx_e_final
    idx=rxn_list[i][1][0].split("_")
    
    rxn_E[i,0]+=np.loadtxt(f"post_process_bulk_gas/{idx[0]}/{rxn_list[i][1][0]}/dft/e")
    if len(rxn_list[i][1])>1:
        for j in rxn_list[i][1][1:]:
            if j in trivial_list:
                rxn_E[i,0]+=np.loadtxt(f"post_process_bulk_gas/gas/trivial/{j}/dft/e")
            else:
                idx=j.split("_")
                rxn_E[i,0]+=np.loadtxt(f"post_process_bulk_gas/gas/{idx[0]}/{j}/dft/e")

    #NNP_rlx_e_final
    idx=rxn_list[i][1][0].split("_")
    rxn_E[i,1]+=np.loadtxt(f"post_process_bulk_gas/{idx[0]}/{rxn_list[i][1][0]}/bpnn/e")
    if len(rxn_list[i][1])>1:
        for j in rxn_list[i][1][1:]:
            if j in trivial_list:
                rxn_E[i,1]+=np.loadtxt(f"post_process_bulk_gas/gas/trivial/{j}/bpnn/e")
            else:
                idx=j.split("_")
                rxn_E[i,1]+=np.loadtxt(f"post_process_bulk_gas/gas/{idx[0]}/{j}/bpnn/e")
    
    #ref_rlx_e_initial
    idx=rxn_list[i][0][0].split("_")
    rxn_E[i,0]-=np.loadtxt(f"post_process_bulk_gas/{idx[0]}/{rxn_list[i][0][0]}/dft/e")
    if len(rxn_list[i][0])>1:
        for j in rxn_list[i][0][1:]:
            if j in trivial_list:
                rxn_E[i,0]-=np.loadtxt(f"post_process_bulk_gas/gas/trivial/{j}/dft/e")
            else:
                idx=j.split("_")
                rxn_E[i,0]-=np.loadtxt(f"post_process_bulk_gas/gas/{idx[0]}/{j}/dft/e")
    #NNP_rlx_e_initial
    idx=rxn_list[i][0][0].split("_")
    rxn_E[i,1]-=np.loadtxt(f"post_process_bulk_gas/{idx[0]}/{rxn_list[i][0][0]}/bpnn/e")
    if len(rxn_list[i][0])>1:
        for j in rxn_list[i][0][1:]:
            if j in trivial_list:
                rxn_E[i,1]-=np.loadtxt(f"post_process_bulk_gas/gas/trivial/{j}/bpnn/e")
            else:
                idx=j.split("_")
                rxn_E[i,1]-=np.loadtxt(f"post_process_bulk_gas/gas/{idx[0]}/{j}/bpnn/e")

#arg_idx=np.argsort(np.abs(rxn_E[:,0]))
#for i in arg_idx:
#    print(f"line num:{i+1}",rxn_list[i],rxn_E[i,0],rxn_E[i,1])
# print(f"MAE: {np.mean(np.abs(rxn_E[:,0]-rxn_E[:,1])):.2f} eV, Pearson: {np.corrcoef(rxn_E[:,0],rxn_E[:,1])[0,1]:.4f},\
# R2: {1-np.sum(np.power(rxn_E[:,0]-rxn_E[:,1],2))/np.sum(np.power(rxn_E[:,0]-np.mean(rxn_E[:,0]),2)):.4f}")
error=rxn_E[:,1]-rxn_E[:,0]
print(f"error Pearson: {np.corrcoef(rxn_E[:,0],-error)[0,1]:.4f}")

print(f"R2: {1-np.sum(np.power(rxn_E[:,0]-error,2))/np.sum(np.power(rxn_E[:,0]-np.mean(rxn_E[:,0]),2)):.4f}")