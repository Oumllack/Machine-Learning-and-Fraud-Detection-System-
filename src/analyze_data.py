import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
print("Chargement des données...")
data = pd.read_csv('data/creditcard.csv')

# Afficher les informations de base
print("\nInformations sur les données :")
print(f"Nombre total d'échantillons : {len(data)}")
print(f"Nombre de fraudes : {data['Class'].sum()}")
print(f"Pourcentage de fraudes : {(data['Class'].sum() / len(data) * 100):.2f}%")

# Afficher les noms des colonnes
print("\nNoms des colonnes :")
print(data.columns.tolist())

# Créer un graphique de la distribution des classes
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Class')
plt.title('Distribution des classes (0: Normal, 1: Fraude)')
plt.savefig('results/class_distribution.png')
plt.close()

# Afficher les statistiques descriptives des features
print("\nStatistiques descriptives des features :")
print(data.describe())

# Sauvegarder les statistiques dans un fichier
with open('results/data_statistics.txt', 'w') as f:
    f.write("Statistiques des données :\n")
    f.write(f"Nombre total d'échantillons : {len(data)}\n")
    f.write(f"Nombre de fraudes : {data['Class'].sum()}\n")
    f.write(f"Pourcentage de fraudes : {(data['Class'].sum() / len(data) * 100):.2f}%\n\n")
    f.write("Noms des colonnes :\n")
    f.write(str(data.columns.tolist()) + "\n\n")
    f.write("Statistiques descriptives :\n")
    f.write(str(data.describe())) 