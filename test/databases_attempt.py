import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from app.sequant_tools import SequantTools

max_peptide_length = 96
polymer_type = 'protein'

# Processing labeled data (AMPs database)
labeled_data = pd.read_csv('../app/utils/data/AMP_ADAM2.txt', on_bad_lines='skip')
labeled_data = labeled_data.replace('+', 1)
labeled_data = labeled_data.fillna(0)
labeled_data = labeled_data.drop(labeled_data[labeled_data.SEQ.str.contains(r'[@#&$%+-/*BXZ]')].index)
labeled_data = labeled_data[labeled_data['SEQ'].apply(lambda x: len(x)) <= max_peptide_length]
labeled_data_seqs = labeled_data['SEQ'].to_list()

''' 
Можно добавить функцию для вывода отфильтрованных последовательностей,
иначе при обучении моделей пользователю придется самостоятельно задавать фильтры для датасета 
(чтобы совпали размеры x и у) - здесь фильтр на строке 17
'''

df = pd.DataFrame({'Seq': labeled_data_seqs, 'Label': labeled_data['Antibacterial']})

sqt = SequantTools(
    sequences=labeled_data_seqs,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    model_folder_path=r'../app/utils/models/proteins'
)

features = sqt.generate_latent_representations()
labels = np.array(labeled_data['Antibacterial'])

x = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model_rfc = RandomForestClassifier(n_estimators=1000, random_state=0)
model_rfc.fit(X_train, y_train)

y_pred = model_rfc.predict(X_test)

print(accuracy_score(y_test, y_pred))

