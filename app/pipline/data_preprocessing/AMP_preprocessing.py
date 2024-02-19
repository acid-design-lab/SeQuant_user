import pandas as pd

AMP_initial = pd.read_csv('../../utils/data/AMP_ADAM2.txt', on_bad_lines='skip')

labeled_data = AMP_initial.replace('+', 1)
labeled_data = labeled_data.fillna(0)
labeled_data = labeled_data.drop(labeled_data[labeled_data.SEQ.str.contains(r'[@#&$%+-/*BXZ]')].index)

labeled_data.to_csv('../../utils/data/AMP.csv', index=False)
