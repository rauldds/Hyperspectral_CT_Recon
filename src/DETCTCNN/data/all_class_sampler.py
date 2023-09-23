'''
    This file contains the implementation of the custom sampler
    used in the project. This sampler helps to balance batches
    so that batches contain at least one sample of every class
'''
import numpy as np
import random
import time
from typing import Sized, Iterator
import torch
from torch.utils.data import Sampler, DataLoader
from src.DETCTCNN.data.music_2d_dataset import MUSIC2DDataset, JointTransform2D

class AllClassSampler(Sampler[int]):
    data_source: Sized
    batch_size: int

    def __init__(self, data_source: Sized, batch_size: int = 128, generator=None) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.generator = generator
        self.data_classes = self.data_source[:]["classes"]
    
    def __len__(self) -> int:
        num_samples = len(self.data_source)
        return num_samples
    
    def __iter__(self) -> Iterator[int]:
        # get the the classes contained by each sample of the dataset (as well as their respective indices)
        all_ids, all_classes = self.data_classes.nonzero(as_tuple=True)
        # defining random number generator that'll be used to generate the indices
        # corresponding to the order in which the dataset will be loaded.
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        
        # check the size of the dataset
        num_samples = len(self.data_source)
        # save the batch size
        n = self.batch_size
        # generate random indices for data loading (indices aren't repeated)
        random_indices = torch.randperm(num_samples, generator=generator)
        # list that'll contain all the indices that were replace by a index corresponding to a sample with the desired class
        all_replaced_ids = []
        # list that'll contain the final indices for data loading
        final_indices = []
        # iterate through the complete batches (batches that cotain 128 elements)
        for i in range(num_samples // n):
            #extract batch indices
            batch_indices = random_indices[i*n:i*n+n]
            # get the classes contained in each batch sample
            idx, classes_in_each_sample = self.data_classes[batch_indices].nonzero(as_tuple=True)
            # get the unique classes in the batch
            unique_classes, repetition_counter = classes_in_each_sample.unique(return_counts=True)
            missing_classes = []
            # check which classes are missing
            for i in range(16):
                if i not in unique_classes:
                    missing_classes.append(i)
            if len(missing_classes)>0:
                # list where the indices of samples that contain the missing classes will be stored
                list_ids_desired_classes = []
                for i in missing_classes:
                    # find the "local" indices of samples that contain the missing class
                    indices_desired_class = (all_classes == i).nonzero(as_tuple=True)[0]
                    # select one of the "local" indices that contain the missing class
                    random_idx_of_desired_class =random.randint(0,len(indices_desired_class)-1)
                    idx_desired_class = indices_desired_class[random_idx_of_desired_class]
                    # map the "local" index to a global index
                    global_idx_desired_class = all_ids[idx_desired_class]
                    # save global index
                    list_ids_desired_classes.append(global_idx_desired_class)
                all_classes_flag = True
                start_time = time.time()
                # replace indices of the batch with new indices, corresponding to missing classes
                while all_classes_flag:
                    new_batch_indices = batch_indices
                    # list to store the indices that have been replaced with new indices
                    replaced_ids = []
                    # replace indices
                    for i in list_ids_desired_classes:
                        replaced_id = random.randint(0,len(batch_indices)-1)
                        new_batch_indices[replaced_id] = i
                        replaced_ids.append(replaced_id)
                    _, new_classes_in_each_sample = self.data_classes[new_batch_indices].nonzero(as_tuple=True)
                    new_unique_classes = new_classes_in_each_sample.unique()
                    # check if after replacing indices the batch has the 16 classes or if a time limit has been reached
                    if (len(new_unique_classes) == 16) or ((time.time()-start_time)>3):
                        # stop while loop
                        all_classes_flag = False
                        batch_indices = new_batch_indices
                        # store replaced indices from this batch in a list containing the replaced indices from all the batches
                        all_replaced_ids.extend(replaced_ids)
                        # store the batch indices in a list containing all the indices of samples to be loaded 
                        # by the dataloader (in batch order)
                        final_indices.extend(batch_indices.tolist())

            else:
                # if no class was missing simply append indices of the batch
                final_indices.extend(batch_indices.tolist())
        list_random_indices = random_indices.tolist()
        # extract elements indices that didn't complete a batch of the desired size
        missing_random_indices = list_random_indices[-(num_samples%n):]
        # create new last batch containing both replaced indices and remaining indices
        last_batch = missing_random_indices
        last_batch.extend(all_replaced_ids)
        missing_classes = []
        # make sure that the last batch has all the classes
        for i in range(16):
            if i not in unique_classes:
                missing_classes.append(i)
        if len(missing_classes)>0:
            for i in missing_classes:
                indices_desired_class = (all_classes == i).nonzero(as_tuple=True)[0]
                random_idx_of_desired_class =random.randint(0,len(indices_desired_class)-1)
                idx_desired_class = indices_desired_class[random_idx_of_desired_class]
                global_idx_desired_class = all_ids[idx_desired_class]
                last_batch.append(global_idx_desired_class)
        final_indices.extend(last_batch)
        yield from final_indices

if __name__ == "__main__":
    batch_size = 128
    transform = JointTransform2D(crop=(40, 40), p_flip=0.5, color_jitter_params=None, long_mask=True,erosion=False)
    ds = MUSIC2DDataset(
        path2d="/media/rauldds/TOSHIBA EXT/MLMI/MUSIC2D_HDF5",
        path3d="/media/rauldds/TOSHIBA EXT/MLMI/MUSIC3D_HDF5",
        partition="train",
        spectrum="reducedSpectrum",
        transform=transform, 
        full_dataset=True,
        dim_red = "none",
        no_dim_red = 10,
        band_selection = None,
        include_nonthreat=True,
        oversample_2D=1,
        split_file=False)
    # DATASET FOR SAMPLER (NO TRANSFORM)
    ds_fs = MUSIC2DDataset(
        path2d="MUSIC2D_HDF5",
        path3d="MUSIC3D_HDF5",
        partition="train",
        spectrum="reducedSpectrum",
        transform=None, 
        full_dataset=True,
        dim_red = "none",
        no_dim_red = 10,
        band_selection = None,
        include_nonthreat=True,
        oversample_2D=1,
        split_file=False)
    our_sampler = AllClassSampler(data_source=ds_fs,batch_size=batch_size)
    dl = DataLoader(ds,batch_size=batch_size,sampler=our_sampler,drop_last=True)

    CLASSES_IN_BATCH_DICT = {
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0,
        10:0,
        11:0,
        12:0,
        13:0,
        14:0,
        15:0,
        16:0
    }
    epochs = 300
    for j in range (epochs):
        #print(f'epoch: {j}')
        for i, data in enumerate(dl, 0):
            classes = data["classes"]
            segs = data["segmentation"]
            dict_id = len(segs.argmax(1).unique())
            #print(f'patched batch classes: {len(segs.argmax(1).unique())}')
            CLASSES_IN_BATCH_DICT[dict_id]+=1
            # TODO: DROP LAST WOULD ELIMINATE THE INCOMPLETE DISTRIBUTION OF THE REMAINING SAMPLES
    
    total_batches=np.sum(np.asarray(list(CLASSES_IN_BATCH_DICT.values())))
    for i in CLASSES_IN_BATCH_DICT.keys():
        CLASSES_IN_BATCH_DICT[i]/=total_batches
        CLASSES_IN_BATCH_DICT[i]*=100
    print(f'batch distribution after {epochs} epochs: {CLASSES_IN_BATCH_DICT}')