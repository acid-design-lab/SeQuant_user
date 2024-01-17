import random
import numpy as np
import numpy.typing as npt
import pandas as pd

from rdkit import Chem
import peptides

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Model

from app.utils.conctants import particles_smiles


class SequantTools:
    """
    Class designed to process DNA/RNA/protein sequences with custom encoder
    """

    def __init__(self, **kwargs):
        self.desc_df: pd.DataFrame = pd.DataFrame()
        self.monomer_dict: dict[str, str] = particles_smiles
        self.generate_rdkit_descriptors()

    def generate_rdkit_descriptors(
            self,
            normalize: bool = True,
            feature_range: tuple[int, int] = (-1, 1)
    ):
        """
        Converts smiles into descriptors using rdkit
        :param normalize: flag to normalise descriptors
        :param feature_range: desired range of transformed data
        """
        descriptor_names: list[str] = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
        num_descriptors: int = len(descriptor_names)
        descriptors_set: npt.NDArray = np.empty((0, num_descriptors), float)

        get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)

        for _, value in self.monomer_dict.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = np.array(
                get_descriptors.ComputeProperties(molecule)
            ).reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        if normalize:
            scaler = MinMaxScaler(feature_range=feature_range)
            descriptors_set = scaler.fit_transform(descriptors_set)

        self.desc_df = pd.DataFrame(
            descriptors_set,
            columns=descriptor_names,
            index=list(self.monomer_dict.keys())
        )

    def filter_sequences(
            self,
            sequences: list[str] | pd.DataFrame,
            max_length: int = 96,
            sequences_column_name: str = None,
            shuffle_seqs: bool = True
    ) -> list[str] | pd.DataFrame:
        if isinstance(sequences, list):
            all_seqs: list[str] = list({
                seq.upper() for seq in sequences if len(seq) <= max_length
            })

            filtered_seqs = [
                x for x in all_seqs if set(x).issubset(set(self.monomer_dict.keys()))
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
                    lambda x: set(x).issubset(set(self.monomer_dict.keys()))
                )
            ]

            if shuffle_seqs:
                peptide_subs = peptide_subs.sample(frac=1)
            return peptide_subs