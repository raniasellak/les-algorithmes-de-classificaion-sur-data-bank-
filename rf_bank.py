import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score,
                             recall_score, classification_report)

# 1. Charger les données
df = pd.read_csv("bank.csv", sep=";")

# 2. Encodage : texte -> nombres
df["y"] = df["y"].map({"no": 0, "yes": 1})
df = pd.get_dummies(df, drop_first=True)

# 3. Séparer X (features) et y (cible)
X = df.drop("y", axis=1)
y = df["y"]

# 4. Train / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Normalisation (facultative pour RF, mais on garde le même pipeline)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Entraîner le Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Prédiction
y_pred = model.predict(X_test)

# 8. Évaluation
print("Accuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall    :", recall_score(y_test, y_pred))
print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))

# 9. Afficher la matrice de confusion
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["no", "yes"], cmap="Oranges"
)
plt.title("Matrice de confusion - Random Forest")
plt.show()