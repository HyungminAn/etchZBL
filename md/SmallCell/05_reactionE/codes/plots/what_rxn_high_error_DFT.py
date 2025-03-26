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
import ase,ase.io,ase.neighborlist
class atom_image():
    def __init__(self):
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
        self.deleting_molecules=np.array(((1,0,0,2),(1,0,0,4),(0,2,0,0),(0,1,3,0),(0,0,1,1)))
        #SiF2,SiF4,N2,NH3
        self.desorbed_ids_at_moment={}
        self.deleting_moments=[]
        self.NN_in_snapshot={}
        self.deleted_cluster_by_id=[]
        self.deleted_cluster_id_concat=[]

    def read_coo(self,image_path):
        return ase.io.read(image_path,format='lammps-data',style="atomic",index="0")
    
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
    
    def cal_Si_coordination_change(self,initial_image_lists,final_image_lists):
        initial_images=[]
        for e,image_path in enumerate(initial_image_lists):
            if e==0:
                image_path="post_process_bulk_gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            elif "_" in image_path :
                image_path="post_process_bulk_gas/gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            else:
                image_path="post_process_bulk_gas/gas/trivial/"+image_path+"/gnn/rlx.coo"
            initial_images.append(self.read_coo(image_path))
        
        final_images=[]
        for e,image_path in enumerate(final_image_lists):
            if e==0:
                image_path="post_process_bulk_gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            elif "_" in image_path :
                image_path="post_process_bulk_gas/gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            else:
                image_path="post_process_bulk_gas/gas/trivial/"+image_path+"/gnn/rlx.coo"
            final_images.append(self.read_coo(image_path))
        
        coords_final_image=self.cal_Si_num_neigh(final_images[:1])
        coords_initial_image=self.cal_Si_num_neigh(initial_images[:1])
        Si_coordination_change=np.sum(coords_final_image-coords_initial_image)
        return Si_coordination_change

    def cal_test_coordination_change(self,initial_image_lists,final_image_lists):
        initial_images=[]
        for e,image_path in enumerate(initial_image_lists):
            if e==0:
                image_path="post_process_bulk_gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            elif "_" in image_path :
                image_path="post_process_bulk_gas/gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            else:
                image_path="post_process_bulk_gas/gas/trivial/"+image_path+"/gnn/rlx.coo"
            initial_images.append(self.read_coo(image_path))
        
        final_images=[]
        for e,image_path in enumerate(final_image_lists):
            if e==0:
                image_path="post_process_bulk_gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            elif "_" in image_path :
                image_path="post_process_bulk_gas/gas/"+image_path.split("_")[0]+"/"+image_path+"/gnn/rlx.coo"
            else:
                image_path="post_process_bulk_gas/gas/trivial/"+image_path+"/gnn/rlx.coo"
            final_images.append(self.read_coo(image_path))
        
        coords_final_image=self.cal_test_num_neigh(final_images)
        coords_initial_image=self.cal_test_num_neigh(initial_images)
        total_coordination_change=np.sum(coords_final_image-coords_initial_image)
        return total_coordination_change
    

a=atom_image()
Si_coordination_change=[]
for i in rxn_list[:]:
    Si_coordination_change.append(a.cal_Si_coordination_change(i[0],i[1]))
idx=np.argsort(rxn_E[:,1]-rxn_E[:,0])
print("#DFT_rxn gnn_rxn error coord_change rxn")
for i in idx:
    print(rxn_E[i,0],rxn_E[i,1],rxn_E[i,1]-rxn_E[i,0],Si_coordination_change[i],rxn_list[i])

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

