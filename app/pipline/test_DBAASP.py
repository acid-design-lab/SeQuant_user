import pandas as pd
from app.sequant_tools import SequantTools
from app.utils.predict_utils import NovaPredictTools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('../utils/data/DBAASP.csv')
df = df[df['Size'] >= 0].drop_duplicates(subset=['SEQ'])
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
sns.violinplot(x=df["Size"])
df_filtered = df[df['SEQ'].isin(sqt.filtered_sequences)]
df_filtered = df_filtered.drop_duplicates(subset=['SEQ'])
targets = df_filtered['Size'].to_numpy().reshape(-1, 1)

# model = NovaPredictTools()
# regressor = model.Lazyregressor_vae(
#     features=descriptors,
#     target=targets
# )
X_train, X_test, Y_train, Y_test = train_test_split(descriptors, targets, test_size=0.2, random_state=0)
scaler = MinMaxScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)

clf = RandomForestRegressor()
clf.fit(x_train, Y_train)
y_pred = clf.predict(x_test)
print("R2__score")
print(r2_score(Y_test, y_pred))
