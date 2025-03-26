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
    rxn_E[i,1]+=np.loadtxt(f"post_process_bulk_gas/{idx[0]}/{rxn_list[i][1][0]}/gnn/e")
    if len(rxn_list[i][1])>1:
        for j in rxn_list[i][1][1:]:
            if j in trivial_list:
                rxn_E[i,1]+=np.loadtxt(f"post_process_bulk_gas/gas/trivial/{j}/gnn/e")
            else:
                idx=j.split("_")
                rxn_E[i,1]+=np.loadtxt(f"post_process_bulk_gas/gas/{idx[0]}/{j}/gnn/e")
    
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
    rxn_E[i,1]-=np.loadtxt(f"post_process_bulk_gas/{idx[0]}/{rxn_list[i][0][0]}/gnn/e")
    if len(rxn_list[i][0])>1:
        for j in rxn_list[i][0][1:]:
            if j in trivial_list:
                rxn_E[i,1]-=np.loadtxt(f"post_process_bulk_gas/gas/trivial/{j}/gnn/e")
            else:
                idx=j.split("_")
                rxn_E[i,1]-=np.loadtxt(f"post_process_bulk_gas/gas/{idx[0]}/{j}/gnn/e")

#arg_idx=np.argsort(np.abs(rxn_E[:,0]))
#for i in arg_idx:
#    print(f"line num:{i+1}",rxn_list[i],rxn_E[i,0],rxn_E[i,1])
print(f"MAE: {np.mean(np.abs(rxn_E[:,0]-rxn_E[:,1])):.2f} eV, Pearson: {np.corrcoef(rxn_E[:,0],rxn_E[:,1])[0,1]:.4f},\
R2: {1-np.sum(np.power(rxn_E[:,0]-rxn_E[:,1],2))/np.sum(np.power(rxn_E[:,0]-np.mean(rxn_E[:,0]),2)):.4f}")
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
fig,ax=plt.subplots()
fig.set_size_inches(2.5,2)
ax.set_aspect('equal','box')
values=np.vstack((rxn_E[:,0],rxn_E[:,1]))
kernel = gaussian_kde(values)
kernel.set_bandwidth(bw_method=kernel.factor*3)
Z=kernel(values)
Z/=np.min(Z[np.nonzero(Z)])
idx = Z.argsort()
DFT_rxn, NNP_rxn, Z = rxn_E[idx,0], rxn_E[idx,1], Z[idx]
scatter_color = plt.cm.Blues
plt.scatter(DFT_rxn,NNP_rxn,c=Z+1,cmap=scatter_color,s=1,norm=matplotlib.colors.LogNorm(vmin=1),zorder=2)
cb=plt.colorbar()
cb.ax.tick_params()

box_range=[np.min(rxn_E),np.max(rxn_E)]
plt.plot(box_range,box_range,'k--',zorder=1)
# ticks=[-15,-5,5,15,25,35]
# plt.xticks(ticks)
# plt.yticks(ticks)
plt.ylabel(r"$E^{\rm{GNN}}_{\rm{rxn}}$ (eV)")
plt.xlabel(r"$E^{\rm{DFT}}_{\rm{rxn}}$ (eV)")
plt.savefig("rxn_E_DFT_GNN.png",dpi=400,bbox_inches='tight')
