import h5py
import numpy as np

with h5py.File('expert_demo_block_1.hdf5', 'r') as f1, h5py.File('expert_demo_block_2.hdf5', 'r') as f2:
    with h5py.File('expert_demo_combined.hdf5', 'w') as f3:
        for dataset_name in f1.keys():
            data1 = f1[dataset_name][:]
            data2 = f2[dataset_name][:]
            combined_data = np.concatenate((data1, data2), axis=0)
            f3.create_dataset(dataset_name, data=combined_data)