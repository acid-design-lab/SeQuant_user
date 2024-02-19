import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def make_monomer_descriptors(monomer_dict: dict[str, str]) -> pd.DataFrame:
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in monomer_dict.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(get_descriptors.ComputeProperties(molecule)).reshape((-1,num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)

    sc = MinMaxScaler(feature_range=(-1, 1))
    scaled_array = sc.fit_transform(descriptors_set)
    descriptors_set = pd.DataFrame(scaled_array, columns=descriptor_names, index=monomer_dict.keys())
    return descriptors_set


def seq_to_matrix(
    sequence: str,
    descriptors: pd.DataFrame,
    num: int
):
    rows = descriptors.shape[1]
    seq_matrix = np.empty((0, rows), float)  # shape (0,rows)
    for aa in sequence:
        descriptors_array = np.array(descriptors.loc[aa]).reshape((1,rows))  # shape (1,rows)
        seq_matrix = np.append(seq_matrix, descriptors_array, axis=0)
    seq_matrix = seq_matrix.T
    shape = seq_matrix.shape[1]
    if shape < num:
        add_matrix = np.pad(seq_matrix,
                            [(0, 0), (0, num-shape)],
                            mode='constant',
                            constant_values=0)
        #water = np.array(descriptors.loc['water']).reshape((rows,1))
        #water_padding = np.resize(a=water, new_shape=(rows,num-shape))
        #add_matrix = np.concatenate((seq_matrix,water_padding), axis=1)

        return add_matrix  # shape (rows,n)

    return seq_matrix


def encode_seqs(
    sequences_list: list[str],
    descriptors: dict[str, str],
    num: int
):
    lst = []
    i = 0
    for sequence in tqdm(sequences_list):
        seq_matrix = seq_to_matrix(sequence=sequence, descriptors=descriptors, num=num)
        lst.append(seq_matrix)
        i += 1
    encoded_seqs = np.dstack(lst)

    return encoded_seqs


def preprocess_input(peptides):
    peptides = peptides.reshape((peptides.shape[0], peptides.shape[1], peptides.shape[2], 1))
    return peptides


def train_test_split(peptides, train_data_ratio):
    indices = peptides.shape[0]
    n_samples = int(indices * train_data_ratio)
    indices = list(range(indices))
    idx_train = np.random.choice(indices, n_samples, replace=False)
    idx_test = list(set(indices) - set(idx_train))
    train_data = peptides[idx_train]
    test_data = peptides[idx_test]
    return train_data, test_data
