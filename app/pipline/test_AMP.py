import pandas as pd
from app.sequant_tools import SequantTools
from app.utils.predict_utils import NovaPredictTools

df = pd.read_csv('../utils/data/AMP.csv')
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
targets = df['Antibacterial']

model = NovaPredictTools()
classifier = model.LazyClass_vae(
    features=descriptors,
    target=targets
)

print(classifier)