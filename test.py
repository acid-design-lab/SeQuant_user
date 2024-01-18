import pandas as pd
from app.sequant_tools import SequantTools

max_peptide_length = 84
polymer_type = 'DNA'
seq_list = ['Atgc', 'GC']
seq_df = pd.DataFrame()

sqt = SequantTools(
    sequences=seq_list,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    model_folder_path=r'app/utils/models/nucleic_acids'
)
X_ref = sqt.generate_latent_representations()
print(X_ref)
