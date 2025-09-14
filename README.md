<img width="3506" height="2481" alt="diplom" src="https://github.com/user-attachments/assets/0e61e83d-cc9e-4351-acdf-6b7afa17bb73" />
# Cellule 1 — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, ConfusionMatrixDisplay)

%matplotlib inline
sns.set()
np.random.seed(42)

# Cellule 2 — Génération du dataset fictif
n = 500

data = pd.DataFrame({
    "Age": np.random.randint(18, 70, n),
    "Sexe": np.random.choice([0,1], n),  # 0 = femme, 1 = homme
    "DureeConsommation": np.random.randint(0, 20, n),  # années
    "FrequenceParSemaine": np.random.randint(0, 14, n),
    "NbCigarettesParJour": np.random.randint(0, 20, n),
    "ExpositionPassive": np.random.choice([0,1], n, p=[0.6,0.4]),
    "Sport": np.random.choice([0,1], n, p=[0.4,0.6])
})

# Calcul d'un score de risque synthétique + bruit
risk = (
    0.03*data["Age"] +
    0.5*data["DureeConsommation"] +
    0.8*data["FrequenceParSemaine"] +
    0.4*data["NbCigarettesParJour"] +
    5*data["ExpositionPassive"] -
    4*data["Sport"] +
    np.random.normal(0, 5, n)
)

# Seuil : top 40% = malade (ajustable)
data["MaladiePulmonaire"] = (risk > np.percentile(risk, 60)).astype(int)

print("Taille du dataset :", data.shape)
data.head()

# Cellule 3 — EDA basique
print(data.info())
print("\nStatistiques descriptives :")
display(data.describe())

print("\nDistribution cible (MaladiePulmonaire) :")
print(data["MaladiePulmonaire"].value_counts())

# Countplot
plt.figure(figsize=(5,4))
sns.countplot(x="MaladiePulmonaire", data=data)
plt.title("Distribution: 0 = sain, 1 = malade")
plt.show()

# Cellule 4 — Visualisations
# Histogrammes
data.hist(figsize=(12,8), bins=15)
plt.tight_layout()
plt.show()

# Heatmap corrélations
plt.figure(figsize=(9,7))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# Cellule 5 — Prétraitement
X = data.drop("MaladiePulmonaire", axis=1)
y = data["MaladiePulmonaire"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (80/20) stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Train/test sizes:", X_train.shape[0], X_test.shape[0])

# Cellule 6 — Modélisation
# 1) Régression logistique
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("=== Régression Logistique ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification report:\n", classification_report(y_test, y_pred_lr))

# 2) Arbre de décision (contrôlé)
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Arbre de décision ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification report:\n", classification_report(y_test, y_pred_dt))

# Cellule 7 — Confusion matrices plots
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, cmap="Blues", display_labels=["Sain","Malade"], ax=plt.gca())
plt.title("Logistic Regression")

plt.subplot(1,2,2)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, cmap="Oranges", display_labels=["Sain","Malade"], ax=plt.gca())
plt.title("Decision Tree")

plt.tight_layout()
plt.show()


# Cellule 8 — ROC & AUC
y_proba_lr = lr.predict_proba(X_test)[:,1]
y_proba_dt = dt.predict_proba(X_test)[:,1]

auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_dt = roc_auc_score(y_test, y_proba_dt)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)

plt.figure(figsize=(6,6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic (AUC={auc_lr:.2f})")
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={auc_dt:.2f})")
plt.plot([0,1],[0,1],"--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbes ROC")
plt.legend()
plt.show()

print(f"AUC Logistic: {auc_lr:.3f}   |   AUC Decision Tree: {auc_dt:.3f}")


# Cellule 9 — Importance features
# Arbre de décision (feature_importances_)
importances = dt.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("Importance (Decision Tree) :")
display(feat_imp)

plt.figure(figsize=(8,4))
feat_imp.plot(kind="bar")
plt.title("Importance des variables (Decision Tree)")
plt.ylabel("Importance")
plt.show()

# Coeffs logistic (signes et magnitude)
coef = pd.Series(lr.coef_[0], index=features).sort_values(key=abs, ascending=False)
print("Coefficients (Logistic Regression) :")
display(coef)


# Cellule 10 — Cross-validation (Stratified K-Fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Accuracy CV pour logistic
cv_acc_lr = cross_val_score(lr, X_scaled, y, cv=skf, scoring="accuracy")
cv_auc_lr = cross_val_score(lr, X_scaled, y, cv=skf, scoring="roc_auc")

print("Logistic - CV Accuracy: ", np.round(cv_acc_lr,3), " mean:", np.round(cv_acc_lr.mean(),3))
print("Logistic - CV AUC:      ", np.round(cv_auc_lr,3), " mean:", np.round(cv_auc_lr.mean(),3))

# Pour RandomForest (exemple d'un modèle plus puissant)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_acc_rf = cross_val_score(rf, X_scaled, y, cv=skf, scoring="accuracy")
cv_auc_rf = cross_val_score(rf, X_scaled, y, cv=skf, scoring="roc_auc")
print("\nRandomForest - CV Accuracy:", np.round(cv_acc_rf,3), " mean:", np.round(cv_acc_rf.mean(),3))
print("RandomForest - CV AUC:     ", np.round(cv_auc_rf,3), " mean:", np.round(cv_auc_rf.mean(),3))


# Cellule 11 — Sauvegarde et prédiction sur nouveaux cas
preds = pd.DataFrame({
    "y_true": y_test,
    "y_pred_lr": y_pred_lr,
    "y_pred_dt": y_pred_dt,
    "y_proba_lr": y_proba_lr
})
preds.to_csv("predictions_chicha.csv", index=False)
print("Fichier sauvegardé: predictions_chicha.csv (dans le dossier du notebook)")

# Exemple : nouveaux patients (Age,Sexe,DureeConsommation,FrequenceParSemaine,NbCigarettesParJour,ExpositionPassive,Sport)
nouveaux = pd.DataFrame([
    [30,1,5,4,0,0,1],
    [45,0,10,8,5,1,0],
    [22,1,1,1,0,0,1]
], columns=X.columns)

nouveaux_scaled = scaler.transform(nouveaux)
print("\nPrédictions (Logistic) pour exemples :")
print(lr.predict(nouveaux_scaled))
print("Probabilités (Logistic) :")
print(lr.predict_proba(nouveaux_scaled)[:,1])
