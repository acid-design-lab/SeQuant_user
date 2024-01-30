import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.typing as npt

import peptides
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import tensorflow as tf
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from app.utils.conctants import monomer_smiles


class SequantTools:
    """
    Class designed to process DNA/RNA/protein sequences with custom encoder using rdkit and peptide descriptors.
    """

    def __init__(
        self,
        sequences: list[str] = [],
        polymer_type: str = '',
        max_sequence_length: int = 96,
        model_folder_path: str = '',
        normalize: bool = True,
        feature_range: tuple[int, int] = (-1, 1),
        add_peptide_descriptors: bool = False
    ):
        """
        Initialisation.
        :param sequences: Enumeration of sequences for filtering.
        :param polymer_type: Polymers types. Possible values: 'protein', 'DNA', 'RNA'.
        :param max_sequence_length: The maximum number of characters in a sequence.
        :param model_folder_path: Path to encoder model folder.
        :param normalize: Set to True to transform values with MinMaxScaler.
        :param feature_range: Desired range of transformed data.
        :param add_peptide_descriptors: Set to True to add peptide descriptors.
        """
        self.descriptors: pd.DataFrame = pd.DataFrame()
        self.filtered_sequences: list[str] = []
        self.prefix: str = ''
        self.encoded_sequences: tf.Tensor = []
        self.model: Model = None #intermediate_layer_model
        self.peptide_descriptor_names: list[str] = []
        self.peptide_descriptors: npt.NDArray = []
        self.latent_representation: npt.NDArray = []

        self.sequences = sequences
        self.polymer_type = polymer_type
        self.max_length = max_sequence_length
        self.model_folder_path = model_folder_path
        self.normalize = normalize
        self.feature_range = feature_range
        self.add_peptide_descriptors = add_peptide_descriptors

        self.monomer_smiles_info: dict[str, str] = monomer_smiles
        self.scaler = MinMaxScaler(feature_range=self.feature_range)

        self.generate_rdkit_descriptors()
        self.filter_sequences()
        self.define_prefix()
        self.model_import()

    def generate_rdkit_descriptors(
            self
    ):
        """
        Generates descriptors for monomers in dict[monomer_name, smiles] using rdkit.
        """
        descriptor_names: list[str] = list(Chem.rdMolDescriptors.Properties.GetAvailableProperties())
        num_descriptors: int = len(descriptor_names)
        descriptors_set: npt.NDArray = np.empty((0, num_descriptors), float)

        get_descriptors = Chem.rdMolDescriptors.Properties(descriptor_names)

        for _, value in self.monomer_smiles_info.items():
            molecule = Chem.MolFromSmiles(value)
            descriptors = np.array(
                get_descriptors.ComputeProperties(molecule)
            ).reshape((-1, num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)

        if self.normalize:
            descriptors_set = self.scaler.fit_transform(descriptors_set)

        self.descriptors = pd.DataFrame(
            descriptors_set,
            columns=descriptor_names,
            index=list(self.monomer_smiles_info.keys())
        )

    def filter_sequences(
            self,
            shuffle: bool = True
    ) -> list[str]:
        """
        Filters sequences based on the maximum length and content of known monomers.
        :param shuffle: Set to True to shuffle list items.
        """
        all_sequences: list[str] = list({
            sequence.upper() for sequence in self.sequences if len(sequence) <= self.max_length
        })
        self.filtered_sequences = [
            sequence for sequence in all_sequences if set(sequence).issubset(set(self.monomer_smiles_info.keys()))
        ]
        if shuffle:
            self.filtered_sequences: list[str] = random.sample(
                self.filtered_sequences,
                len(self.filtered_sequences)
            )

    def define_prefix(self):
        """
        Formalizes the prefix depending on the polymer type.
        """
        assert self.polymer_type in ['protein', 'DNA', 'RNA'], "Possible values: 'protein', 'DNA', 'RNA'.\n"
        if self.polymer_type == 'protein':
            self.prefix = ''
        elif self.polymer_type == 'DNA':
            self.prefix = 'd'
        elif self.polymer_type == 'RNA':
            self.prefix = 'r'

    def model_import(self):
        """
        Initialise model
        """
        trained_model = tf.keras.models.load_model(self.model_folder_path)
        layer_name = 'Latent'
        self.model = Model(
            inputs=trained_model.input,
            outputs=trained_model.get_layer(layer_name).output
        )

    def sequence_to_descriptor_matrix(
            self,
            sequence: str
    ) -> tf.Tensor:
        """
        Сonverts a single sequence into a descriptor matrix.
        :param sequence: Alphanumeric sequence.
        :return: Tensor with shape (max_sequence_length, num_of_descriptors).
        """
        rows: int = self.descriptors.shape[1]
        sequence_matrix: tf.Tensor = tf.zeros(shape=[0, rows])  # shape (0,rows)
        for monomer in sequence:
            monomer_params = tf.constant(
                self.descriptors.loc[self.prefix + monomer],
                dtype=tf.float32
            )
            descriptors_array = tf.expand_dims(
                monomer_params,
                axis=0  # shape (1,rows)
            )
            sequence_matrix = tf.concat(
                [sequence_matrix, descriptors_array],
                axis=0
            )
        sequence_matrix = tf.transpose(sequence_matrix)
        shape = sequence_matrix.get_shape().as_list()[1]
        if shape < 96:
            paddings = tf.constant([[0, 0], [0, 96 - shape]])
            sequence_matrix = tf.pad(
                sequence_matrix,
                paddings=paddings,
                mode='CONSTANT',
                constant_values=-1
            )
        return sequence_matrix

    def encoding(
            self
    ) -> tf.Tensor:
        """
        Сonverts a list of sequences into a  sequences/descriptor tensor.
        :return: Sequences/descriptor tensor.
        """
        container = []
        for i, sequence in tqdm(enumerate(self.filtered_sequences)):
            seq_matrix = tf.expand_dims(
                self.sequence_to_descriptor_matrix(
                    sequence=sequence
                ),
                axis=0
            )
            container.append(seq_matrix)

        self.encoded_sequences = tf.concat(container, axis=0)
        return self.encoded_sequences

    def generate_latent_representations(
            self
    ) -> np.ndarray:
        """
        Processes the sequences/descriptor tensor using a model.
        :return: Ready-made features.
        """
        self.encoding()
        self.latent_representation: np.ndarray = self.model.predict(
            self.encoded_sequences
        )

        if self.add_peptide_descriptors:
            self.define_peptide_generated_descriptors()
            self.latent_representation = np.concatenate(
                (self.latent_representation, self.peptide_descriptors),
                axis=1
            )
        return self.latent_representation

    def define_peptide_generated_descriptors(
        self,
    ) -> np.ndarray:
        """
        Generates an array of descriptors using the peptides lib.
        :return: Peptide descriptors
        """
        peptide_descriptors = pd.DataFrame(
            [peptides.Peptide(seq).descriptors() for seq in self.filtered_sequences]
        )
        self.peptide_descriptor_names = list(peptide_descriptors.columns)
        self.peptide_descriptors = np.array(peptide_descriptors)

        if self.normalize:
            self.peptide_descriptors = self.scaler.fit_transform(self.peptide_descriptors)

        return self.peptide_descriptors
