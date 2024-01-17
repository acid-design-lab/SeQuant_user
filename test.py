import pandas as pd

from Funcs import (
    generate_rdkit_descriptors,
    SeQuant_encoding,
    generate_latent_representations,
    filter_sequences
)


max_peptide_length = 96
polymer_type = 'DNA'
seq_list = ['AT', 'GC']
seq_df = pd.DataFrame()

filtered_sequences = filter_sequences(sequences=seq_list,
                                      max_length=max_peptide_length,
                                      sequences_column_name='sequence',
                                      shuffle_seqs=True)

descriptors_set = generate_rdkit_descriptors()

encoded_sequences = SeQuant_encoding(sequences_list=filtered_sequences,
                                     polymer_type=polymer_type,
                                     descriptors=descriptors_set,
                                     num=max_peptide_length)

X = generate_latent_representations(sequences_list=seq_list,
                                    sequant_encoded_sequences=encoded_sequences,
                                    polymer_type=polymer_type,
                                    add_peptide_descriptors=False,
                                    path_to_model_folder=r'Models/nucleic_acids')
