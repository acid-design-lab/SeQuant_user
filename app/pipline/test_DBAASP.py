import pandas as pd
from app.sequant_tools import SequantTools
from app.utils.predict_utils import NovaPredictTools

df = pd.read_csv('../utils/data/DBAASP.csv')
df_seq = df['SEQ']

polymer_type = 'protein'
max_peptide_length = 96

sqt = SequantTools(
    sequences=df_seq,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    model_folder_path=r'../utils/models/proteins'
)

descriptors = sqt.generate_latent_representations()

df_filtered = df[df['SEQ'].isin(sqt.filtered_sequences)]
df_filtered = df_filtered.drop_duplicates(subset=['SEQ'])
targets = df_filtered['Size']

model = NovaPredictTools()
regressor = model.Lazyregressor_vae(
    features=descriptors,
    target=targets
)

print(regressor)
