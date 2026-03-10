import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Wczytaj dane
# Zmien 'twoje_dane.csv' na właściwą ścieżkę
df = pd.read_csv("data_csv.csv")

# 2) Ustal nazwy kolumn
TARGET = "Stack voltage"

# 3) Usuń wiersze bez targetu (jeśli są)
df = df.dropna(subset=[TARGET]).copy()
#usuniecie koluj

# 4) Zdefiniuj listę cech wejściowych
#    - Usuwamy target
#    - Usuwamy "Stack power" jeśli zawiera V*I (wyciek informacji)
drop_cols = [TARGET]
if "Stack power" in df.columns:
    drop_cols.append("Stack power")

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df[TARGET]

# 5) Podział na train/test
#    Na start robimy prosty podział losowy.
#    Potem pokażę Ci, jak zrobić podział "po serii" (bardziej poprawny dla eksperymentów).
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 6) Pipeline: uzupełnianie braków -> skalowanie -> model
#    Używamy Ridge (regresja liniowa + L2), bo zwykła LinearRegression bywa niestabilna,
#    gdy cechy są skorelowane (a przepływy zwykle są).
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", Ridge(alpha=1.0))
])

# 7) Trening
model.fit(X_train, y_train)

# 8) Predykcja i metryki
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("=== Wyniki na TEST ===")
print(f"MAE  = {mae:.4f} V")
print(f"RMSE = {rmse:.4f} V")
print(f"R^2  = {r2:.4f}")

# 9) Interpretacja: współczynniki regresji
#    Po skalowaniu współczynniki mówią, jak silnie cecha wpływa na V "w skali odchyleń standardowych".
ridge = model.named_steps["regressor"]
coefs = pd.Series(ridge.coef_, index=feature_cols).sort_values(key=lambda s: s.abs(), ascending=False)

print("\n=== Najważniejsze cechy (wg |współczynnika|) ===")
print(coefs.head(15))

# 10) Szybki sanity check: 10 przykładowych predykcji
preview = pd.DataFrame({
    "y_true": y_test.values[:10],
    "y_pred": y_pred[:10],
    "error": (y_pred[:10] - y_test.values[:10])
})
print("\n=== Podgląd 10 predykcji ===")
print(preview)