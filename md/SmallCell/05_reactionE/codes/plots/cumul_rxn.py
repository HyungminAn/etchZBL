import numpy as np
with open("rxn_E.dat",'r') as O:
    a=np.array([[float(j) for j in i.split()] for i in O.readlines()[1:]])
b=np.cumsum(a,axis=0)
with open("cumul_rxn.dat",'w') as O:
    O.write("#ref_E gnn_E\n")
    for i in b:
        O.write(f"{i[0]} {i[1]}\n")
