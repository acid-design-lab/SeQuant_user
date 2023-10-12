import os
import random

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Model

import peptides

AA_DICT = {
    'dA': r'O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)C[C@@H]3O',  # DNA
    'dT': r'CC1=CN(C(=O)NC1=O)C2CC(C(O2)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O',
    'dG': r'O=P(O)(O)OP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n1cnc2c1NC(=N/C2=O)\N)C[C@@H]3O',
    'dC': r'C1[C@@H]([C@H](O[C@H]1N2C=CC(=NC2=O)N)CO[P@@](=O)(O)O[P@@](=O)(O)OP(=O)(O)O)O',
    'rA': r'c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N',  # RNA
    'rU': r'C1=CN(C(=O)NC1=O)C2C(C(C(O2)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O',
    'rG': r'C1=NC2=C(N1C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N',
    'rC': r'c1cn(c(=O)nc1N)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O',
    'A': 'CC(C(=O)O)N',  # protein
    'R': 'C(CC(C(=O)O)N)CN=C(N)N',
    'N': 'C(C(C(=O)O)N)C(=O)N',  
    'D': 'C(C(C(=O)O)N)C(=O)O',
    'C': 'C(C(C(=O)O)N)S',
    'Q': 'C(CC(=O)N)C(C(=O)O)N',
    'E': 'C(CC(=O)O)C(C(=O)O)N',
    'G': 'C(C(=O)O)N',
    'H': 'C1=C(NC=N1)CC(C(=O)O)N',
    'I': 'CCC(C)C(C(=O)O)N',
    'L': 'CC(C)CC(C(=O)O)N',
    'K': 'C(CCN)CC(C(=O)O)N',
    'M': 'CSCCC(C(=O)O)N',
    'F': 'C1=CC=C(C=C1)CC(C(=O)O)N',
    'P': 'C1CC(NC1)C(=O)O',
    'S': 'C(C(C(=O)O)N)O',
    'T': 'CC(C(C(=O)O)N)O',
    'W': 'C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N',
    'Y': 'C1=CC(=CC=C1CC(C(=O)O)N)O',
    'V': 'CC(C)C(C(=O)O)N',
    'O': 'CC1CC=NC1C(=O)NCCCCC(C(=O)O)N',
    'U': 'C(C(C(=O)O)N)[Se]'
}


def generate_rdkit_descriptors(
    normalize: tuple[int, int] = (-1, 1),
    monomer_dict_: dict[str, str] = AA_DICT
) -> pd.DataFrame:
    descriptor_names: list[str] = list(
        rdMolDescriptors.Properties.GetAvailableProperties()
    )
    get_descriptors = rdMolDescriptors.Properties(descriptor_names)
    num_descriptors = len(descriptor_names)

    descriptors_set = np.empty((0, num_descriptors), float)

    for _, value in monomer_dict_.items():
        molecule = Chem.MolFromSmiles(value)
        descriptors = np.array(
            get_descriptors.ComputeProperties(molecule)
        ).reshape((-1, num_descriptors))

        descriptors_set = np.append(descriptors_set, descriptors, axis=0)
    if normalize is not None:
        sc = MinMaxScaler(feature_range=normalize)
        scaled_array = sc.fit_transform(descriptors_set)

        desc_df = pd.DataFrame(
            scaled_array,
            columns=descriptor_names,
            index=list(monomer_dict_.keys())
        )
    else:
        desc_df = pd.DataFrame(
            descriptors_set,
            columns=descriptor_names,
            index=list(monomer_dict_.keys())
        )

    return desc_df


def filter_sequences(
    sequences: list[str] | pd.DataFrame,
    max_length: int = 96,
    sequences_column_name: str = None,
    shuffle_seqs: bool = True,
    aa_dict_: dict[str, str] = AA_DICT
) -> list[str] | pd.DataFrame:
    if isinstance(sequences, list):
        all_seqs: list[str] = list({
            seq.upper() for seq in sequences if len(seq) <= max_length
        })

        filtered_seqs = [
            x for x in all_seqs if set(x).issubset(set(aa_dict_.keys()))
        ]
        if shuffle_seqs:
            filtered_seqs: list[str] = random.sample(
                filtered_seqs,
                len(filtered_seqs)
            )
        return filtered_seqs

    elif isinstance(sequences, pd.DataFrame):
        sequences[sequences_column_name] = sequences[
            sequences_column_name
        ].map(lambda x: x.replace(" ", ""))

        sequences[sequences_column_name] = sequences[
            sequences_column_name
        ].str.upper()

        sequences = sequences[
            sequences[sequences_column_name].apply(
                lambda x: len(x) <= max_length
            )
        ]
        peptide_subs = sequences[
            sequences[sequences_column_name].apply(
                lambda x: set(x).issubset(set(aa_dict_.keys()))
            )
        ]

        if shuffle_seqs:
            peptide_subs = peptide_subs.sample(frac=1)
        return peptide_subs


def seq_to_matrix_(
    sequence: str,
    polymer_type: str,
    descriptors: pd.DataFrame,
    num: int
) -> tf.Tensor:

    if polymer_type == 'protein':
        prefix = ''
    elif polymer_type == 'DNA':
        prefix = 'd'
    elif polymer_type == 'RNA':
        prefix = 'r'
    else:
        print('Wrong polymer type')
        return

    rows = descriptors.shape[1]
    seq_matrix = tf.zeros(shape=[0, rows])     # shape (0,rows)
    for aa in sequence:
        aa_params = tf.constant(
            descriptors.loc[prefix+aa],
            dtype=tf.float32
        )
        descriptors_array = tf.expand_dims(
            aa_params,
            axis=0                             # shape (1,rows)
        )
        seq_matrix = tf.concat(
            [seq_matrix, descriptors_array],
            axis=0
        )
    seq_matrix = tf.transpose(seq_matrix)
    shape = seq_matrix.get_shape().as_list()[1]
    if shape < num:
        paddings = tf.constant([[0, 0], [0, num - shape]])
        add_matrix = tf.pad(
            seq_matrix,
            paddings=paddings,
            mode='CONSTANT',
            constant_values=-1
        )

        return add_matrix                      # shape (rows,n)

    return seq_matrix


def SeQuant_encoding(
    sequences_list: list[str],
    polymer_type: str,
    descriptors: pd.DataFrame,
    num: int
) -> tf.Tensor:
    container = []
    for i, sequence in enumerate(sequences_list):
        if i % 3200 == 0:
            print(i * 100 / len(sequences_list), ' %')

        seq_matrix = tf.expand_dims(
            seq_to_matrix_(
                sequence=sequence,
                polymer_type=polymer_type,
                descriptors=descriptors,
                num=num
            ),
            axis=0
        )
        container.append(seq_matrix)

    encoded_seqs = tf.concat(container, axis=0)

    return encoded_seqs


def generate_latent_representations(
    sequences_list: list[str],
    sequant_encoded_sequences: tf.Tensor,
    polymer_type: str = 'protein',
    add_peptide_descriptors: bool = False,
    path_to_model_folder: str = ''
) -> tuple[np.ndarray, list[str]] | np.ndarray:

    trained_model = tf.keras.models.load_model(path_to_model_folder)

    layer_name = 'Latent'
    intermediate_layer_model = Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer(layer_name).output
    )
    latent_representation: np.ndarray = intermediate_layer_model.predict(
        sequant_encoded_sequences
    )

    if add_peptide_descriptors:
        (
            full_descriptor_set,
            descriptor_names
        ) = add_peptide_generated_descriptors(
            sequences_list=sequences_list,
            encoded_sequences_to_add_to=latent_representation
        )

        return full_descriptor_set, descriptor_names

    return latent_representation


def add_peptide_generated_descriptors(
    sequences_list: list[str],
    encoded_sequences_to_add_to: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    seq_prop_set = pd.DataFrame(
        [peptides.Peptide(seq).descriptors() for seq in sequences_list]
    )

    descriptor_names = list(seq_prop_set.columns)

    seq_prop_set = np.array(seq_prop_set)
    sc = MinMaxScaler(feature_range=(-1, 1))
    seq_prop_set = sc.fit_transform(seq_prop_set)

    concatenated_seq_props = np.concatenate(
        (encoded_sequences_to_add_to, seq_prop_set),
        axis=1
    )

    return concatenated_seq_props, descriptor_names
