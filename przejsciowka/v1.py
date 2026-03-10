#dodatkwoe wejscia dodac ktore maja byc wstepnie puste!
#budowa prostego skryptu sieci neuronowej

#1) zrozumiec kod
#2) stworzyc petle for dla kazdego z elementow filtorwanych
#3) stworzyc df tylko dla wodoru i tylko dla metanolu + h2o
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_csv.csv')

#celem do przewidzenia jest 'Stack voltage'
target = 'Stack voltage'
#usuniecie wierszy bez targetu:
df = df.dropna(subset = [target]).copy()

#pozbycie sie y i stacked power jako zbednch kolumn: 
drop_cols = [target] + ['Stack power']

feature_cols = []
for c in df.columns:
    if c not in drop_cols:
        feature_cols.append(c)

#mapowanie x i y:
x = df[feature_cols]
y = df[target]

print(f'test naglowkow x:\n{x.head()}')
print (f'test naglowkow y:\n{y.head()}')

#podzial na train i test: 
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# #generowanie petli dla kazdego compound: 
# for compound in df['compound'].unique():
#testowanie: 

