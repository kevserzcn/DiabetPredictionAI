# 1) Gerekli kütüphaneleri yükleyin
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

# 2) Veriyi yükleyin
df = pd.read_csv("diabetes.csv")

# 3) Aykırı değerleri temizleyin
def detect_outliers_iqr(df):
    outlier_indices = []
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices.extend(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index)
    return list(set(outlier_indices))

outlier_indices = detect_outliers_iqr(df)
df_cleaned = df.drop(outlier_indices).reset_index(drop=True)

# 4) Veriyi bölme
X = df_cleaned.drop(["Outcome"], axis=1)
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 5) Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6) Modeli eğitme (Grid Search CV ile en iyi parametreleri bulma)
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dt = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train_scaled, y_train)

best_dt_model = grid_search.best_estimator_

# 7) Yeni veriyi test etme
new_data = np.array([[6, 148, 71, 35, 1, 34.6, 0.627, 51]])
new_data_scaled = scaler.transform(new_data)
new_prediction = best_dt_model.predict(new_data_scaled)

print("Yeni verinin tahmini:", "Diyabetli" if new_prediction[0] == 1 else "Diyabetli Değil")
