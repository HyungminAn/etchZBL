import numpy as np
import os,sys
import graph_tool.all as gt
import ase,ase.io,ase.neighborlist

if sys.argv[1][-1]=="/":
    target_path=sys.argv[1][:-1]
else:
    target_path=sys.argv[1]
with open(target_path+"/rxn.dat",'r') as O:
    rxn_list=[i.split()[0].split("/") for i in O.readlines()[1:]]
trivial_list=os.listdir(target_path+"/post_process_bulk_gas/gas/trivial")

rxn_E=np.zeros((len(rxn_list),2)) #ref_rlx_E, NNP_rlx_E
for i in range(len(rxn_list)):
    rxn_list[i][0]=rxn_list[i][0].split(",")
    rxn_list[i][1]=rxn_list[i][1].split(",")
    

    #ref_rlx_e_final
    idx=rxn_list[i][1][0].split("_")
    
    rxn_E[i,0]+=np.loadtxt(target_path+f"/post_process_bulk_gas/{idx[0]}/{rxn_list[i][1][0]}/dft/e")
    if len(rxn_list[i][1])>1:
        for j in rxn_list[i][1][1:]:
            if j in trivial_list:
                rxn_E[i,0]+=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/trivial/{j}/dft/e")
            else:
                idx=j.split("_")
                rxn_E[i,0]+=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/{idx[0]}/{j}/dft/e")

    #NNP_rlx_e_final
    idx=rxn_list[i][1][0].split("_")
    rxn_E[i,1]+=np.loadtxt(target_path+f"/post_process_bulk_gas/{idx[0]}/{rxn_list[i][1][0]}/gnn/e")
    if len(rxn_list[i][1])>1:
        for j in rxn_list[i][1][1:]:
            if j in trivial_list:
                rxn_E[i,1]+=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/trivial/{j}/gnn/e")
            else:
                idx=j.split("_")
                rxn_E[i,1]+=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/{idx[0]}/{j}/gnn/e")
    
    #ref_rlx_e_initial
    idx=rxn_list[i][0][0].split("_")
    rxn_E[i,0]-=np.loadtxt(target_path+f"/post_process_bulk_gas/{idx[0]}/{rxn_list[i][0][0]}/dft/e")
    if len(rxn_list[i][0])>1:
        for j in rxn_list[i][0][1:]:
            if j in trivial_list:
                rxn_E[i,0]-=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/trivial/{j}/dft/e")
            else:
                idx=j.split("_")
                rxn_E[i,0]-=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/{idx[0]}/{j}/dft/e")
    #NNP_rlx_e_initial
    idx=rxn_list[i][0][0].split("_")
    rxn_E[i,1]-=np.loadtxt(target_path+f"/post_process_bulk_gas/{idx[0]}/{rxn_list[i][0][0]}/gnn/e")
    if len(rxn_list[i][0])>1:
        for j in rxn_list[i][0][1:]:
            if j in trivial_list:
                rxn_E[i,1]-=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/trivial/{j}/gnn/e")
            else:
                idx=j.split("_")
                rxn_E[i,1]-=np.loadtxt(target_path+f"/post_process_bulk_gas/gas/{idx[0]}/{j}/gnn/e")

#arg_idx=np.argsort(np.abs(rxn_E[:,0]))
#for i in arg_idx:
#    print(f"line num:{i+1}",rxn_list[i],rxn_E[i,0],rxn_E[i,1])

class atom_image():
    def __init__(self,target_path='..'):
        self.target_path=target_path
        self.bond_length=np.zeros((4,4))
        #In lammps idx, Si:1 N:2 H:3 F:4
        self.bond_length[0,0]=0.0
        self.bond_length[0,1]=2.43
        self.bond_length[0,2]=1.92
        self.bond_length[0,3]=2.43
        self.bond_length[1,1]=1.85
        self.bond_length[1,2]=1.34
        self.bond_length[1,3]=1.85
        self.bond_length[2,2]=0.83
        self.bond_length[2,3]=1.3
        self.bond_length[3,3]=1.8
        for i in range(4):
            for j in range(i,4):
                self.bond_length[j,i]=self.bond_length[i,j]

        self.bond_length_dict={}
        for i in range(self.bond_length.shape[0]):
            for j in range(i,self.bond_length.shape[0]):
                self.bond_length_dict.update({(i+1,j+1):self.bond_length[i,j]})
        self.matcher={14:1,7:2,1:3,9:4}
        self.matcher_dft={1:14,2:7,3:1,4:9}
        # 'SiSi'  :    2.2  ,
        # 'SiN'   :    1.9  ,
        # 'SiH'   :    1.9  ,
        # 'SiF'   :    2.0  ,
        # 'NN'    :    1.8  ,
        # 'NH'    :    1.15 ,
        # 'NF'    :    1.7 ,
        # 'HH'    :    0.675 ,
        # 'HF'    :    0.864 ,
        # 'FF'    :    1.3   
        # self.deleting_molecules=np.array(((1,0,0,2),(1,0,0,4),(0,2,0,0),(0,1,3,0),(0,0,1,1)))
        # #SiF2,SiF4,N2,NH3
        # self.desorbed_ids_at_moment={}
        # self.deleting_moments=[]
        # self.NN_in_snapshot={}
        # self.deleted_cluster_by_id=[]
        # self.deleted_cluster_id_concat=[]

    # def read_coo(self,image_path,format):
    #     if format=='lammps-data':
    #         return ase.io.read(image_path,format='lammps-data',style="atomic",index="0")
    #     elif format=='vasp':
    #         return ase.io.read(image_path,format='vasp')
    
    def cal_Si_num_neigh(self,images):
        Si_num_neighs=np.zeros((len(images)),dtype=int)
        for e,i in enumerate(images):
            atomic_nums=i.get_atomic_numbers()
            NN_list=ase.neighborlist.neighbor_list('ij',i,self.bond_length_dict)
            Si_list=np.nonzero(atomic_nums==1)[0]
            
            for atom_idx in Si_list:
                Si_num_neighs[e]=np.sum(NN_list[0]==atom_idx)
        return Si_num_neighs
    

    def cal_test_num_neigh(self,images):
        num_neighs=np.zeros((len(images)),dtype=int) #the number of Si-Si, Si undercoordination, N-N
        for e,i in enumerate(images):
            atomic_nums=i.get_atomic_numbers()
            NN_list=ase.neighborlist.neighbor_list('ij',i,self.bond_length_dict)
            for j in range(len(NN_list[0])):
                if atomic_nums[NN_list[0][j]]==1 and atomic_nums[NN_list[1][j]]==1:
                    num_neighs[e]+=1
                elif atomic_nums[NN_list[0][j]]==2 and atomic_nums[NN_list[1][j]]==2:
                    num_neighs[e]+=1
            # Si_list=np.nonzero(atomic_nums==1)[0]
            
            # for atom_idx in Si_list:
            #     num_neighs[e]=np.sum(NN_list[0]==atom_idx)
        return num_neighs
    
    def get_image(self,initial_image_lists,final_image_lists):
        image_nums=initial_image_lists[0]
        image_path=self.target_path+"/post_process_bulk_gas/"+image_nums.split("_")[0]+"/"+image_nums+"/gnn/rlx.coo"
        initial_nn_rlx=ase.io.read(image_path,format='lammps-data',style="atomic",index="0")
        initial_nn_rlx=initial_nn_rlx[initial_nn_rlx.get_array('id').argsort()]
        initial_nn_rlx=initial_nn_rlx[np.array([self.matcher_dft[i] for i in initial_nn_rlx.get_atomic_numbers()]).argsort()]

        image_path=self.target_path+"/post_process_bulk_gas/"+image_nums.split("_")[0]+"/"+image_nums+"/dft/CONTCAR"
        initial_dft_rlx=ase.io.read(image_path,format='vasp')
        initial_dft_rlx.set_atomic_numbers([self.matcher[i] for i in initial_dft_rlx.get_atomic_numbers()])

        image_nums=final_image_lists[0]
        image_path=self.target_path+"/post_process_bulk_gas/"+image_nums.split("_")[0]+"/"+image_nums+"/gnn/rlx.coo"
        final_nn_rlx=ase.io.read(image_path,format='lammps-data',style="atomic",index="0")
        final_nn_rlx=final_nn_rlx[final_nn_rlx.get_array('id').argsort()]
        final_nn_rlx=final_nn_rlx[np.array([self.matcher_dft[i] for i in final_nn_rlx.get_atomic_numbers()]).argsort()]

        image_path=self.target_path+"/post_process_bulk_gas/"+image_nums.split("_")[0]+"/"+image_nums+"/dft/CONTCAR"
        final_dft_rlx=ase.io.read(image_path,format='vasp')
        final_dft_rlx.set_atomic_numbers([self.matcher[i] for i in final_dft_rlx.get_atomic_numbers()])

        self.images=(initial_nn_rlx,initial_dft_rlx,final_nn_rlx,final_dft_rlx)

        self.graphs =[]
        self.vprops=[]
        for i in range(len(self.images)):
            self.graphs.append(gt.Graph(directed = False))
            self.vprops.append([])  
        for i in range(len(self.images)):
            NN_list=ase.neighborlist.neighbor_list('ij',self.images[i],self.bond_length_dict)
            self.vprops[i]=self.graphs[i].new_vertex_property("short",vals=self.images[i].get_atomic_numbers())
            self.graphs[i].add_edge_list(zip(NN_list[0],NN_list[1]))
    
    def compare_bond_config(self):
        initial_diff=gt.similarity(self.graphs[0],self.graphs[1],norm=False,distance=True,asymmetric=True)
        final_diff=gt.similarity(self.graphs[2],self.graphs[3],norm=False,distance=True,asymmetric=True)
        return initial_diff,final_diff

    

a=atom_image(target_path)
print("#DFT_rxn_E gnn_rxn_E E_error initial_image_config final_image_config rxn_list")
# error=rxn_E[:,1]-rxn_E[:,0]

for i in rxn_E[:,0].argsort():
    a.get_image(rxn_list[i][0],rxn_list[i][0])
    print(rxn_E[i,0],rxn_E[i,1],rxn_E[i,1]-rxn_E[i,0],a.compare_bond_config(),rxn_list[i])
    
#     Si_coordination_change.append(a.cal_Si_coordination_change(i[0],i[1]))
# idx=np.argsort(rxn_E[:,1]-rxn_E[:,0])
# print("#DFT_rxn gnn_rxn error coord_change rxn")
# for i in idx:
#     print(rxn_E[i,0],rxn_E[i,1],rxn_E[i,1]-rxn_E[i,0],Si_coordination_change[i],rxn_list[i])

# i=90
# print(rxn_list[i],rxn_E[i,:],Si_coordination_change[i])
# i=126
# print(rxn_list[i],rxn_E[i,:],Si_coordination_change[i])

#total_coordination_change=[]
#for i in rxn_list[:]:
#    total_coordination_change.append(a.cal_test_coordination_change(i[0],i[1]))
#for i in range(len(rxn_list)):
#    print(rxn_E[i,1]-rxn_E[i,0],total_coordination_change[i])


# print("#error, Si_coordination_change")
# for i in range(len(rxn_list)):
#     print(rxn_E[i,1]-rxn_E[i,0],Si_coordination_change[i])
# print(rxn_E)
# print(rxn_list)

