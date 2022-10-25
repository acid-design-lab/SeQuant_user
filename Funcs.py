from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model
import peptides

aa_dict = {'A': 'CC(C(=O)O)N', 'R': 'C(CC(C(=O)O)N)CN=C(N)N', 'N': 'C(C(C(=O)O)N)C(=O)N',
           'D': 'C(C(C(=O)O)N)C(=O)O', 'C': 'C(C(C(=O)O)N)S', 'Q': 'C(CC(=O)N)C(C(=O)O)N',
           'E': 'C(CC(=O)O)C(C(=O)O)N', 'G': 'C(C(=O)O)N', 'H': 'C1=C(NC=N1)CC(C(=O)O)N',
           'I': 'CCC(C)C(C(=O)O)N', 'L': 'CC(C)CC(C(=O)O)N', 'K': 'C(CCN)CC(C(=O)O)N',
           'M': 'CSCCC(C(=O)O)N', 'F': 'C1=CC=C(C=C1)CC(C(=O)O)N', 'P': 'C1CC(NC1)C(=O)O',
           'S': 'C(C(C(=O)O)N)O', 'T': 'CC(C(C(=O)O)N)O', 'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
           'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O', 'V': 'CC(C)C(C(=O)O)N', 'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
           'U': 'C(C(C(=O)O)N)[Se]'}


def generate_rdkit_descriptors(normalize: tuple = (-1, 1)):
    descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in aa_dict.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(get_descriptors.ComputeProperties(molecule)).reshape((-1, num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)

    sc = MinMaxScaler(feature_range=normalize)
    scaled_array = sc.fit_transform(descriptors_set)
    return pd.DataFrame(scaled_array, columns=descriptor_names, index=list(aa_dict.keys()))


def seq_to_matrix_(sequence, descriptors, num):

    rows = descriptors.shape[1]
    seq_matrix = tf.zeros(shape=[0, rows])  # shape (0,rows)
    for aa in sequence:
        aa_params = tf.constant(descriptors.loc[aa],
                                dtype=tf.float32)
        descriptors_array = tf.expand_dims(aa_params,
                                           axis=0)  # shape (1,rows)
        seq_matrix = tf.concat([seq_matrix, descriptors_array],
                               axis=0)
    seq_matrix = tf.transpose(seq_matrix)
    shape = seq_matrix.get_shape().as_list()[1]
    if shape < num:
        paddings = tf.constant([[0, 0], [0, num-shape]])
        add_matrix = tf.pad(seq_matrix,
                            paddings=paddings,
                            mode='CONSTANT',
                            constant_values=-1)

        return add_matrix  # shape (rows,n)

    return seq_matrix


def SeQuant_encoding(sequences_list, descriptors, num):

    container = []
    for i, sequence in enumerate(sequences_list):
        if i % 3200 == 0:
            print(i*100/len(sequences_list), ' %')

        seq_matrix = tf.expand_dims(seq_to_matrix_(sequence=sequence,
                                                   descriptors=descriptors,
                                                   num=num),
                                    axis=0)
        container.append(seq_matrix)

    encoded_seqs = tf.concat(container,
                             axis=0)

    return encoded_seqs


def generate_latent_representations(sequences_list,
                                    sequant_encoded_sequences,
                                    trained_model,
                                    add_peptide_descriptors=False):

    layer_name = 'Latent'
    intermediate_layer_model = Model(inputs=trained_model.input,
                                     outputs=trained_model.get_layer(layer_name).output)
    latent_representation = intermediate_layer_model.predict(sequant_encoded_sequences)

    if add_peptide_descriptors:
        full_descriptor_set, descriptor_names = add_peptide_generated_descriptors(sequences_list=sequences_list,
                                                                                  encoded_sequences_to_add_to=latent_representation)
        return full_descriptor_set, descriptor_names

    return latent_representation


def add_peptide_generated_descriptors(sequences_list, encoded_sequences_to_add_to):

    seq_prop_set = pd.DataFrame([peptides.Peptide(seq).descriptors() for seq in sequences_list])

    descriptor_names = list(seq_prop_set.columns)

    seq_prop_set = np.array(seq_prop_set)
    sc = MinMaxScaler(feature_range=(-1, 1))
    seq_prop_set = sc.fit_transform(seq_prop_set)

    return np.concatenate((encoded_sequences_to_add_to, seq_prop_set), axis=1), descriptor_names
