import pandas as pd
import peptides

peptide_1 = pd.read_csv('../../utils/data/peptides.csv')
peptide_2 = pd.read_csv('../../utils/data/peptides_2.csv')

# Uniting DataFrame
peptide = pd.concat([peptide_1, peptide_2], axis=0)
peptide.reset_index(drop=True, inplace=True)

# Keeping only required columns
needed_columns = ['SEQUENCE', 'TARGET ACTIVITY - TARGET SPECIES',
                  'TARGET ACTIVITY - ACTIVITY MEASURE VALUE',
                  'TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)']

peptides_cleaned = peptide.loc[:, needed_columns]
DBAASP_initial = peptides_cleaned.dropna(how='any')
DBAASP_initial.reset_index(drop=True, inplace=True)

# Renaming columns
DBAASP_initial.columns = ['SEQ', 'Target', 'Type', 'Activity (μg/ml)']

# Keeping rows containing MIC data
filtered_DBAASP = DBAASP_initial[DBAASP_initial['Type'] == 'MIC']
filtered_DBAASP = filtered_DBAASP.drop('Type', axis=1)
filtered_DBAASP.rename(columns={'Activity (μg/ml)': 'MIC (μg/ml)'}, inplace=True)

# Removing duplicates based on unique values from SEQ and Target columns
filtered_DBAASP = filtered_DBAASP.drop_duplicates(subset=['SEQ', 'Target'])
filtered_DBAASP.reset_index(drop=True, inplace=True)

# Getting descriptors from the peptides package
DBAASP_sequences = filtered_DBAASP["SEQ"]

DBAASP_peptide_descriptors = pd.DataFrame(
            [peptides.Peptide(seq).descriptors() for seq in DBAASP_sequences]
)

# Getting required descriptors
needed_descriptors = ["E1", "E2", "E3"]
DBAASP_peptide_descriptors_cleaned = DBAASP_peptide_descriptors.loc[:, needed_descriptors]

DBAASP_peptide_descriptors_cleaned.columns = ['Hydrophobicity', 'Size', 'Helical_propensity']

# Uniting datasets to obtain final dataframe with sequences and target values

DBAASP = pd.concat([filtered_DBAASP, DBAASP_peptide_descriptors_cleaned], axis=1)

DBAASP.to_csv('../../utils/data/DBAASP.csv', index=False)

