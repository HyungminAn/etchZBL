import time
import ase,os,sys
import numpy as np
import ase.neighborlist
import ase.io
from multiprocessing import Pool, cpu_count
# import graph_tool.all as gt

class atom_image():
    def __init__(self):
        self.bond_length=np.zeros((4,4))
        #In lammps idx, Si:1 N:2 H:3 F:4
        self.bond_length[0,0]=2.2
        self.bond_length[0,1]=2.1
        self.bond_length[0,2]=2.0
        self.bond_length[0,3]=2.0
        self.bond_length[1,1]=1.8
        self.bond_length[1,2]=1.4
        self.bond_length[1,3]=1.7
        self.bond_length[2,2]=0.81
        self.bond_length[2,3]=1.01
        self.bond_length[3,3]=1.54
        for i in range(4):
            for j in range(i,4):
                self.bond_length[j,i]=self.bond_length[i,j]
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

    def set_image_path(self,image_path):
        self.image_path=image_path
    def read_dump(self,i):
        image_path=self.image_path+f"/dump_{i}.lammps"
        self.dump=ase.io.read(image_path,format='lammps-dump-text',index=":")
    
    def read_initial_image(self,i):
        image_path=self.image_path+f"/HF_shoot_{i}.coo"
        self.initial_image=ase.io.read(image_path,format='lammps-data',index=0,style="atomic",sort_by_id=False)

    def read_final_image(self,i):
        image_path=self.image_path+f"/HF_shoot_{i}.coo"
        self.final_image=ase.io.read(image_path,format='lammps-data',index=0,style="atomic",sort_by_id=False)


    def mv_final2initial(self):
        self.initial_image=self.final_image
        del self.final_image
        

    def find_nearest_neighbors(self,i):
        """Find nearest neighbors for atom i within the cutoff_distance."""
        indices = np.arange(self.num_atoms)
        # atomic_N=self.atoms.get_atomic_numbers()
        distances = self.atoms.get_distances(i, indices, mic=True)
        # cutoff_list=np.zeros(len(atoms))
        # for j in indices:
        #     cutoff_list[j]=cutoff_distance[atomic_N[i]-1,atomic_N[j]-1]
        # neighbors_logical=distances < cutoff_list
        # neighbors_logical[i]=False

        neighbors_logical=np.full(self.num_atoms,False)
        for j in indices:
            neighbors_logical[j]= distances[j] < self.bond_length[self.atomic_numbers[i]-1,self.atomic_numbers[j]-1]
        neighbors_logical[i]=False
        neighbors = np.where(neighbors_logical)
        return (i, neighbors)
    
    def find_NN(self):
        # Create a multiprocessing Pool.
        pool = Pool(cpu_count())
        # Run the find_nearest_neighbors function for each atom.
        self.nearest_neighbor = pool.starmap(self.find_nearest_neighbors, [(i,) for i in range(Image.num_atoms)])
        pool.close()
        pool.join()

    def get_delete_atoms(self,current_incidence):
        self.graph = Graph(directed = False)
        self.graph.add_vertex(self.num_atoms)
        for i in self.nearest_neighbor:
            for j in i[1][0]:
                self.graph.add_edge(i[0],j)
        cluster, hist = label_components(self.graph)
        slab_idx = np.argmax(hist)
        # print(slab_idx)
        cluster_idx=list(range(len(hist)))
        cluster_idx.pop(slab_idx)
        to_delete_list=[]
        tot_N_deleting=0
        for i in cluster_idx:
            atom_in_cluster_idx=np.argwhere(cluster.a==i)
            stoichiometry=np.zeros(4,dtype=int)
            for j in atom_in_cluster_idx:
                stoichiometry[self.atomic_numbers[j]-1]+=1
            for to_compare_mol in self.deleting_molecules:
                if np.array_equal( stoichiometry,to_compare_mol):
                    tot_N_deleting+=np.sum(stoichiometry)
                    to_delete_list.append(atom_in_cluster_idx)
                    break
        if tot_N_deleting>0:
            self.delete_before_write=1
            with open("delete.log",'a') as O:
                for i in to_delete_list:
                    stoichiometry=np.zeros(4,dtype=int)
                    for j in i:
                        stoichiometry[self.atomic_numbers[i]-1]+=1
                    O.write(f"{current_incidence}")
                    O.write("/")
                    for j in stoichiometry:
                        O.write(f"{j},")
                    O.write(f"\n")
            n=0
            self.to_delete_idx=np.zeros(tot_N_deleting,dtype=int)
            for i in to_delete_list:
                for j in i:
                    self.to_delete_idx[n]=j
                    n+=1
        else:
            self.delete_before_write=0

    def write_final_image(self):
        # print(self.atoms[1])
        if self.delete_before_write==1:
            del self.atoms[self.to_delete_idx]        
        ase.io.write(self.image_path.replace(".coo","_after_removing.coo"),self.atoms,format='lammps-data',atom_style='atomic',velocities=True,specorder=["H","He","Li","Be"])

    def find_deleted_atoms(self):
        pool = Pool(cpu_count())
        
        

        result=pool.starmap(self.parallel_find_atoms, [(i,) for i in range(1, len(self.dump))])
        for i in result:
            if i!=[]:
                self.deleting_moments.append(i[0])
                self.desorbed_ids_at_moment[i[0]]=i[1]
        pool.close()
        pool.join()

    def parallel_find_atoms(self,snapshot):
        return_tmp=[]
        if len(self.dump[snapshot]) < len(self.dump[snapshot-1]):
            return_tmp.append(snapshot-1)
            id_diff= np.setdiff1d(self.dump[snapshot-1].arrays['id'],
            self.dump[snapshot].arrays['id'])
            return_tmp.append([])
            for diff in id_diff:
                return_tmp[1].append(diff)
        return return_tmp

    
    def find_NN_dump(self,dump_idx):
        # Create a multiprocessing Pool.
        pool = Pool(cpu_count())
        # Run the find_nearest_neighbors function for each atom.
        self.NN_in_snapshot[dump_idx] = pool.starmap(self.find_nearest_neighbors_dump, [(i,dump_idx) for i in range(len(Image.dump[dump_idx]))])
        pool.close()
        pool.join()
    

    def find_nearest_neighbors_dump(self,i,dump_idx):
        """Find nearest neighbors for atom i within the cutoff_distance."""
        indices = np.arange(len(self.dump[dump_idx]))
        distances = self.dump[dump_idx].get_distances(i, indices, mic=True)
        neighbors_logical=np.full(len(self.dump[dump_idx]),False)
        for j in indices:
            neighbors_logical[j]= distances[j] < self.bond_length[self.dump[dump_idx].numbers[i]-1,
                                                                  self.dump[dump_idx].numbers[j]-1]
        neighbors_logical[i]=False
        neighbors = np.where(neighbors_logical)
        return (i, neighbors)
    
    def find_desorbed_clusters(self,dump_idx):
        desorbed_idx_at_moment=np.zeros(len(self.desorbed_ids_at_moment[dump_idx]),dtype=int)
        for e,i in enumerate(self.desorbed_ids_at_moment[dump_idx]):
            desorbed_idx_at_moment[e]=  np.argwhere(self.dump[dump_idx].arrays['id']==i)[0][0]
        # print(desorbed_idx_at_moment)
        graph = gt.Graph(directed = False)
        graph.add_vertex(len(self.dump[dump_idx]))
        
        for i in self.NN_in_snapshot[dump_idx]:
            for j in i[1][0]:
                graph.add_edge(i[0],j)
        cluster, hist = gt.label_components(graph)
        slab_idx = np.argmax(hist)
        # print(slab_idx)
        cluster_idx=list(range(len(hist)))
        cluster_idx.pop(slab_idx)
        
        for deleted_atom_idx in desorbed_idx_at_moment:
            cluster_tmp=cluster.a[deleted_atom_idx]
            if  any(( self.dump[dump_idx].arrays['id'][deleted_atom_idx] in id_tmp for id_tmp in self.cluster_by_id)) \
                or self.dump[dump_idx].positions[deleted_atom_idx,2]<=2 or not cluster_tmp in cluster_idx:
                continue
            else:
                cluster_tmp=cluster.a[deleted_atom_idx]
                atom_current_idx=np.argwhere(cluster.a==cluster_tmp)
                self.cluster_by_id.append(self.dump[dump_idx].arrays['id'][atom_current_idx].reshape(-1))
                stoichio_tmp= self.dump[dump_idx].numbers[atom_current_idx]
                composition=np.zeros(4,dtype=int)
                for atom_number_idx in range(1,5):
                    composition[atom_number_idx-1]=np.count_nonzero(stoichio_tmp==atom_number_idx)
                self.cluster_stoichiometry.append(composition)





if __name__ == '__main__':
    import pickle
    # atoms=ase.io.read(sys.argv[1],format='lammps-data',index=0,style="atomic")
    import graph_tool.all as gt
    for i in range(1,801):
        with open(f"images/Images_{i}.bin",'rb') as O:
            Image=pickle.load(O)
        # Image=np.load(f"images/Images_{i}.npy",allow_pickle=True)
        Image.cluster_by_id=[]
        Image.cluster_stoichiometry=[]
        for moment_idx in Image.deleting_moments:
            Image.find_desorbed_clusters(moment_idx)
        for e,cluster in enumerate(Image.cluster_by_id):
            with open("desorption_graph.dat",'a') as O:
                O.write(f"{i}/")
                for composition in Image.cluster_stoichiometry[e]:
                    O.write(f"{composition},")
                O.write(f"/")
                for atom_id in cluster:
                    O.write(f"{atom_id},")
                O.write("\n")

        # print(Image.cluster_stoichiometry)
            
        # Image=atom_image()
        # Image.set_image_path(sys.argv[1])
        # Image.read_initial_image(i-1)
        # Image.read_dump(i)
        # Image.read_final_image(i)
        # Image.find_deleted_atoms()
        # for moment_idx in Image.deleting_moments:
        #     Image.find_NN_dump(moment_idx)
        # print(i,time.time()-start_time)
        # np.save(f"Images_{i}.npy",Image)
            
            
        
            # Image.mv_final2initial()
        
