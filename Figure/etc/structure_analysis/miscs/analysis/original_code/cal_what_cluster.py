import numpy as np
import sys

#Si SiF SiF2 SiF3 SiF4 N NH NH2 NH3 NH4 N2 SixNy
clusters=np.zeros(12,dtype=int)
start=int(sys.argv[2])
end=int(sys.argv[3])
with open(sys.argv[1],'r') as O:
    a=[i.split()[0] for i in O.readlines()]
b=[i.split("/") for i in a]
for i in range(len(b)):
    b[i][0]=int(b[i][0])
    b[i][1]=np.array([int(tmp) for tmp in b[i][1].split(",")[:-1]],dtype=int)
    b[i][2]=np.array([int(tmp) for tmp in b[i][2].split(",")[:-1]],dtype=int)
for i in b:
    if i[0]<start or i[0] > end:
        continue
    if i[1][0]==0:
        if i[1][1]==2:
            if np.sum(i[1])==2:
                clusters[10]+=1 #N2
        elif i[1][1]==1:
            clusters[4+np.sum(i[1])]+=1 #N--NH4
    elif i[1][0]==1:
        if i[1][1]==0:
            clusters[np.sum(i[1])-1]+=1 #Si--SiF4
        else:
            clusters[-1]+=1#SixNy
    else:
        clusters[-1]+=1#SixNy

print(clusters/np.sum(clusters))



