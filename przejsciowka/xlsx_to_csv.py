#skrypt pomagajacy w konwersji plików XLSX na CSV
import pandas as pd

df  = pd.read_excel('data_modified.xlsx')
#skonwertowane do csv 

#usun puste wiersze:
df = df.dropna(how='all')

df.to_csv('data_csv.csv', index=False)

