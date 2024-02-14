# SeQuant
Пакет для генерации дескрипторов с помощью rdkit и авторской нейронной сети

## сборка и запуск
Установка pyenv и poetry - https://habr.com/ru/articles/599441/
Документация poetry - https://python-poetry.org/docs/#installing-with-the-official-installer

Инциализация виртуальной среды
```bash
poetry install
```

## Документация
Документация SeQuant


### Class SequantTools

#### Метод класса generate_rdkit_descriptors

Метод генерирует дескрипторы для мономеров, представленных в формате словаря {monomer_name: smiles}, используя библиотеку **RDKit**.
<br>
* Входные данные:
  * self - экземпляр класса.

* Логика работы:
  
  * Сначала определяется список имен доступных дескрипторов с помощью **Chem.rdMolDescriptors.Properties.GetAvailableProperties()**;
    
  * Далее создается пустой массив descriptors_set размером (0, num_descriptors), где num_descriptors - количество дескрипторов;
    
  * Затем для каждого значения в словаре **self.monomer_smiles_info**:
    - Создается “молекула" из SMILES-строки с помощью **Chem.MolFromSmiles()**;
    - Рассчитываются дескрипторы для данной молекулы с помощью **get_descriptors.ComputeProperties()**;
    - Полученные дескрипторы добавляются к descriptors_set.
  
  * Если параметр self.normalize равен True (дефолтное значение True), то дескрипторы нормализуются с использованием self.scaler.fit_transform(descriptors_set).
    - Используется **MinMaxScaler** из пакета sklearn 
(self.scaler = MinMaxScaler(feature_range=self.feature_range), feature_range: tuple[int, int] = (-1;1))
 к слову мы можем внести вариабельность в этом месте, но это потом однажды;

  * Полученные результаты сохраняются в виде DataFrame self.descriptors с колонками descriptor_names и индексами из ключей словаря **self.monomer_smiles_info**.
<br>

#### Метод класса filter_sequences

Метод фильтрует поданные при инициализации класса последовательности **self.sequences**, основываясь на максимальной длине последовательности **self.max_length** (дефолтное значение int = 96) и наличию мономеров последовательностей в словаре **self.monomer_smiles_info**.
<br>

* Входные данные:
  * self - экземпляр класса; 
  * shuffle - bool, дефолтное значение True: флаг для перемешивания списка.

* Логика работы:
  * Создается список all_sequences, содержащий уникальные последовательности в верхнем регистре из **self.sequences**, где длина последовательности не превышает **self.max_length**;
  * Далее фильтрация происходит путем проверки, что каждая последовательность из all_sequences содержит только мономеры из self.monomer_smiles_info.keys(). Отфильтрованные последовательности добавляются в список self.filtered_sequences;
  * Если параметр **ignore_unknown_monomer** False, и длина изначального и конечного списка не равны, то выходит ошибка;
  * Если параметр **shuffle** равен True, то отфильтрованный список случайно перемешиваются с помощью **random.sample**.

### Метод класса define_prefix

Метод определяет префикс для мономеров последовательностей в зависимости от типа полимера, указанного в **self.polymer_type**.
<br>
* Входные данные:
  * self - экземпляр класса.

* Логика работы:
  * Проверяется, что значение self.polymer_type находится в списке ['protein', 'DNA', 'RNA']. В противном случае вызывается исключение с сообщением о возможных значениях.
  * В соответствии с типом полимера присваивается значение атрибуту **self.prefix**:
    * Для типа 'protein' - префикс остается пустым.
    * Для типа 'DNA' - префикс устанавливается как 'd'.
    * Для типа 'RNA' - префикс устанавливается как 'r'.

### Метод класса model_import

Метод инициализирует предобученную модель автоэнкодера.
<br>
* Входные данные:
  * self - экземпляр класса.

* Логика работы:
  * Метод загружает предварительно обученную модель из указанного пути **self.model_folder_path**;
  * Устанавливается имя слоя, который будет использоваться для извлечения выходных данных модели  -  'Latent';
  * Создается новая модель **self.model**, которая принимает в качестве входов входы предварительно обученной модели и выводит выходной слой с именем 'Latent'.


### Метод класса sequence_to_descriptor_matrix

Метод преобразует одиночную последовательность в матрицу дескрипторов.
<br>
* Входные данные:
  * self - экземпляр класса;
  * sequence (str) - буквенная последовательность мономеров последовательностей.

* Логика работы:
  * Инициализируется переменная rows, равная количеству столбцов в **self.descriptors**;
  * Создается пустая матрица **sequence_matrix** с формой (0, rows);
  * Для каждого мономера в последовательности:
    - Получаются дескрипторы для мономера из **self.descriptors**;
    - Дескрипторы преобразуются в тензор и добавляются в sequence_matrix. (в названиях столбцов - имена дескрипторов, в строках - мономеры);
  * Матрица **sequence_matrix** транспонируется (в названиях столбцов - мономеры, в названиях строк - имена дескрипторов);
  * Если количество столбцов в **sequence_matrix** меньше 96, к матрице добавляются значения -1 до заполнения матрицы до 96.

* Возвращаемое значение:
  * Тензор с формой (max_sequence_length, num_of_descriptors).

### Метод класса encoding

Преобразует список последовательностей в тензор дескрипторов.
<br>
* Входные данные:
  * self - экземпляр класса.

* Логика работы:
  * Инициализируется пустой список container;
  * Для каждой последовательности в списке **self.filtered_sequences**.
создается матрица дескрипторов с помощью метода **sequence_to_descriptor_matrix()**
полученная матрица добавляется к списку container;
  * Атрибуту self.encoded_sequences присваивается итоговое значение списка container;   

* Возвращаемое значение:
  * Метод возвращает атрибут **encoded_sequences**.

### Метод класса generate_latent_representations

Метод создает скрытые представления для каждой последовательности, которые можно использовать как готовые дескрипторы для применения в ML.
<br>
* Входные данные:
  * self - экземпляр класса.
<br>
* Логика работы:
  * Применяется метод **encoding()**, с помощью которого создается список матриц дескрипторов каждой последовательности (для каждой последовательности своя матрица, где столбцы - названия мономеров, строки - имена дескрипторов);
  * С помощью метода **predict()** атрибута **self.model** (модель, созданная методом **model_import()** ) предсказываются скрытые представления (значения выходного слоя - “Latent”, модели);
  * Если параметр **self.add_peptide_descriptors** принимает значение True (дефолтное значение - False): к скрытым представлениям добавляются дескрипторы пептидов, сформированные методом define_peptide_generated_descriptors().
<br>
* Возвращаемое значение:
  * Скрытые представления - готовые к применению в ML фичи.

<br>

### Метод класса define_peptide_generated_descriptors

Создает дескрипторы пептидов, используя библиотеку Peptides.
<br>
* Входные данные:
  * self - экземпляр класса.
<br>
* Логика работы:
  * Создается DataFrame peptide_descriptors;
  * Для каждой последовательности seq в self.filtered_sequences:
    - Генерируются дескрипторы для пептида, представленного  последовательностью seq с помощью peptides.Peptide(seq).descriptors();
    - Результат добавляется в peptide_descriptors;
  * Имена столбцов DataFrame сохраняются в self.peptide_descriptor_names;
  * DataFrame преобразуется в numpy array и сохраняется в self.peptide_descriptors;
  * Если параметр normalize равен True:
    - Дескрипторы нормализуются с использованием self.scaler.
<br>
* Возвращаемое значение:
   numpy array сгенерированных пептидных дескрипторов - self.peptide_descriptors.

### Метод класса add_monomers

Позволяет добавить новые мономеры в словарь стандартных мономеров.
<br>
* Входные данные:
  * self - экземпляр класса;
<br>
* Необходимые данные при инициализации класса:
  * self.new_monomers: list[dict] = [] - список словарей с новыми мономерами в формате 
  {'name': 'str', 'class': 'protein/DNA/RNA', 'smiles': 'str'};
  * self.ignore_unknown_monomer: bool = False - флаг игнорирования неизвестных мономеров;
<br>
* Логика работы:
  * Если флаг ignore_unknown_monomer равен False:
    * Для каждого элемента item в списке self.new_monomers:
      * Извлекается имя мономера name и класс class;
      * Устанавливается префикс 'r' для класса 'RNA' и 'd' для класса 'DNA';
      * Формируется имя с префиксом;
      * Если имя отсутствует в словаре self.monomer_smiles_info:
        * Добавляется запись в словарь: self.monomer_smiles_info[name] = item['smiles'].
<br>

### Пример использования метода **generate_latent_representations()**.

```python
import pandas as pd
from app.sequant_tools import SequantTools

max_peptide_length = 84
polymer_type = 'protein'
seq_list = ['Atgc', 'GC', 'YYGT']
seq_df = pd.DataFrame()

mon1 = {'name': 'A', 'class': 'protein', 'smiles': ''}
mon2 = {'name': 'Y', 'class': 'DNA', 'smiles': 'CCC(C(=O)O)'}
mon3 = {'name': 'I', 'class': 'protein', 'smiles': 'C'}

new_monomers = [mon1, mon2, mon3]

sqt = SequantTools(
   sequences=seq_list,
   polymer_type=polymer_type,
   max_sequence_length=max_peptide_length,
   model_folder_path=r'../app/utils/models/nucleic_acids',
   new_monomers=new_monomers
)

X = sqt.generate_latent_representations()
print(X)
```

Результат:

```python
[[ 0.01442203, ..., 0.05054794]
 [-0.01018154, ..., 0.07343792]
 [ 0.01396738, ..., 0.0524387 ]]
```
### Пример использования класса для решения задачи классификации (разделение пептидов на обладающие антибактериальным действием и не обладающие таковым):

```python
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from app.sequant_tools import SequantTools

max_peptide_length = 96
polymer_type = 'protein'

labeled_data = pd.read_csv('../Data/AMP_ADAM2.txt', on_bad_lines='skip')
labeled_data_seqs = labeled_data['SEQ'].to_list()
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
```
