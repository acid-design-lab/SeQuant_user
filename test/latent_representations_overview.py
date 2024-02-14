import pandas as pd
from app.sequant_tools import SequantTools

max_peptide_length = 96
polymer_type = 'protein'

# Processing labeled data (AMPs database)
labeled_data = pd.read_csv('../app/utils/data/AMP_ADAM2.txt', on_bad_lines='skip')
labeled_data = labeled_data.replace('+', 1)
labeled_data = labeled_data.fillna(0)
labeled_data = labeled_data.drop(labeled_data[labeled_data.SEQ.str.contains(r'[@#&$%+-/*BXZ]')].index)
labeled_data_seqs = labeled_data['SEQ'].to_list()

sqt = SequantTools(
    sequences=labeled_data_seqs,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    model_folder_path=r'../app/utils/models/proteins'
)

features = sqt.generate_latent_representations()

print(features[0])
print(len(features[0]))
