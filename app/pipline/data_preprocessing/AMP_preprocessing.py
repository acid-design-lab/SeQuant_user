import pandas as pd

max_peptide_length = 96
AMP_initial = pd.read_csv('../../utils/data/AMP_ADAM2.txt', on_bad_lines='skip')

labeled_data = AMP_initial.replace('+', 1)
labeled_data = labeled_data.fillna(0)
labeled_data = labeled_data.drop(labeled_data[labeled_data.SEQ.str.contains(r'[@#&$%+-/*BXZ]')].index)
labeled_data = labeled_data[labeled_data['SEQ'].apply(lambda x: len(x)) <= max_peptide_length]

labeled_data.to_csv('../../utils/data/AMP.csv', index=False)
