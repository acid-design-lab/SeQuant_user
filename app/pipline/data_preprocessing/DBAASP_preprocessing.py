import pandas as pd
from app.sequant_tools import SequantTools

peptides_1 = pd.read_csv('../../utils/data/peptides.csv')
peptides_2 = pd.read_csv('../../utils/data/peptides_2.csv')

"""
для DBAASP находим величины связанные с растворимостью (по типу растворимости) 
с помощью либы peptide + давай около трех величин выберешь сам, 
соответвенно они не должны быть входящими множествами с дескрипторы rdkit на которых мы учим.

5.1  Сбор сета для DBAASP:
Объедени оба сета, сброси дубликаты по сиквенсу. Потом идет поиск таргетных величин с помощью либы. 
Итоговый результат должен состоять из колонок сиквенс + n-таргетов. 
Далее его нужно сохранить и кинуть в отдельную папку на никитин диск.

Из анализа датасетов известно:
Среди последовательностейй есть дубликаты, так как активность пептидов измерялась на разных бактериях и разными величинами.
наиболее многочисленной группой является группа МИК, значит в первую очередь мы сбросим все строки, кроме содержащих МИК
Далее сгруппируем по таргету и последовательности, чтобы проверить на дубликаты, дубликаты сбросим

Итоговые колонки:
Секвенс
Таргет
Активность по базе данных
"""
# Uniting DataFrame
peptides = pd.concat([peptides_1, peptides_2], axis=0)
peptides.reset_index(drop=True, inplace=True)

# Keeping only the required columns
needed_columns = ['SEQUENCE', 'TARGET ACTIVITY - TARGET SPECIES',
                  'TARGET ACTIVITY - ACTIVITY MEASURE VALUE',
                  'TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)']

peptides_cleaned = peptides.loc[:, needed_columns]
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
polymer_type = 'protein'
max_peptide_length = 96

sqt = SequantTools(
    sequences=DBAASP_sequences,
    polymer_type=polymer_type,
    max_sequence_length=max_peptide_length,
    model_folder_path=r'../../utils/models/proteins',
    add_peptide_descriptors=True
)

DBAASP_peptide_descriptors = sqt.define_peptide_generated_descriptors()

DBAASP_peptide_descriptors = pd.DataFrame(DBAASP_peptide_descriptors)

# Getting required descriptors
needed_descriptors = [32, 33, 34]
DBAASP_peptide_descriptors_cleaned = DBAASP_peptide_descriptors.loc[:, needed_descriptors]

DBAASP_peptide_descriptors_cleaned.columns = ['Hydrophobicity', 'Size', 'Helical_propensity']

# Uniting datasets to obtain final dataframe with sequences and target values

DBAASP = pd.concat([filtered_DBAASP, DBAASP_peptide_descriptors_cleaned], axis=1)

DBAASP.to_csv('../../utils/data/DBAASP.csv', index=False)
