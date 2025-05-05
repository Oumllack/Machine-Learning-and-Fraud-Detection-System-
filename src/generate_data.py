import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def generate_normal_transactions(n_samples):
    """Génère des transactions normales avec des patterns réalistes."""
    n_features = 30
    
    # Génération de base avec des clusters
    n_clusters = 5
    cluster_centers = np.random.normal(0, 0.5, (n_clusters, n_features))
    cluster_sizes = np.random.dirichlet(np.ones(n_clusters) * 2) * n_samples
    
    X = np.zeros((n_samples, n_features))
    start_idx = 0
    for i in range(n_clusters):
        size = int(cluster_sizes[i])
        X[start_idx:start_idx + size] = np.random.normal(
            cluster_centers[i], 
            0.3, 
            (size, n_features)
        )
        start_idx += size
    
    # Ajout de corrélations entre features
    for i in range(0, n_features-1, 2):
        correlation = np.random.uniform(0.4, 0.8)
        X[:, i+1] = correlation * X[:, i] + np.sqrt(1 - correlation**2) * X[:, i+1]
    
    # Ajout de quelques valeurs aberrantes naturelles
    outlier_idx = np.random.choice(n_samples, size=int(0.0005 * n_samples), replace=False)
    X[outlier_idx] = np.random.normal(0, 2, (len(outlier_idx), n_features))
    
    return X

def generate_fraudulent_transactions(n_samples):
    """Génère des transactions frauduleuses avec différents patterns de fraude."""
    n_features = 30
    X = np.zeros((n_samples, n_features))
    
    # Différents types de fraudes avec des patterns plus distincts
    fraud_types = np.random.choice(4, size=n_samples)
    
    # Type 1: Valeurs très anormales sur plusieurs features
    mask_type1 = (fraud_types == 0)
    X[mask_type1] = np.random.normal(2.5, 0.8, (mask_type1.sum(), n_features))
    
    # Type 2: Valeurs extrêmes sur quelques features spécifiques
    mask_type2 = (fraud_types == 1)
    X[mask_type2] = np.random.normal(0, 1, (mask_type2.sum(), n_features))
    random_features = np.random.choice(n_features, size=5, replace=False)
    X[mask_type2][:, random_features] = np.random.normal(4, 0.5, (mask_type2.sum(), 5))
    
    # Type 3: Combinaisons inhabituelles de features
    mask_type3 = (fraud_types == 2)
    X[mask_type3] = np.random.normal(0, 1, (mask_type3.sum(), n_features))
    for i in range(0, n_features-1, 2):
        correlation = np.random.uniform(-0.9, -0.7)  # Corrélations négatives très fortes
        X[mask_type3, i+1] = correlation * X[mask_type3, i] + np.sqrt(1 - correlation**2) * X[mask_type3, i+1]
    
    # Type 4: Patterns cycliques anormaux
    mask_type4 = (fraud_types == 3)
    X[mask_type4] = np.random.normal(0, 1, (mask_type4.sum(), n_features))
    for i in range(0, n_features, 3):
        X[mask_type4, i] = np.sin(np.linspace(0, 4*np.pi, mask_type4.sum())) * 3
    
    return X

def add_time_patterns(df, n_normal, n_fraud):
    """Ajoute des patterns temporels réalistes."""
    # Distribution du temps sur une période de 7 jours (en secondes)
    time_span = 7 * 24 * 60 * 60
    
    # Transactions normales suivent un pattern journalier plus marqué
    hours = np.linspace(0, 24, 24)
    normal_probs = np.exp(-(hours - 14)**2 / 20)  # Pic en milieu de journée
    normal_probs = normal_probs / normal_probs.sum()
    
    normal_time = np.random.choice(
        np.linspace(0, time_span, 24*7),
        size=n_normal,
        p=np.tile(normal_probs, 7) / 7
    )
    
    # Transactions frauduleuses plus concentrées la nuit
    night_hours = np.random.uniform(0, 6 * 3600, n_fraud)  # Entre 0h et 6h
    day_hours = np.random.uniform(0, time_span, n_fraud)   # N'importe quand
    
    # 80% la nuit, 20% n'importe quand
    fraud_time = np.where(
        np.random.random(n_fraud) < 0.8,
        night_hours,
        day_hours
    )
    
    # Combinaison et tri
    df.loc[df['Class'] == 0, 'Time'] = normal_time
    df.loc[df['Class'] == 1, 'Time'] = fraud_time
    return df

def generate_dataset(n_normal=100000, n_fraud=1000):
    """Génère un dataset complet avec transactions normales et frauduleuses."""
    # Génération des features
    X_normal = generate_normal_transactions(n_normal)
    X_fraud = generate_fraudulent_transactions(n_fraud)
    
    # Combinaison des données
    X = np.vstack([X_normal, X_fraud])
    y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Création du DataFrame
    columns = ['V' + str(i) for i in range(1, 31)]
    df = pd.DataFrame(X, columns=columns)
    df['Class'] = y
    
    # Ajout des patterns temporels
    df = add_time_patterns(df, n_normal, n_fraud)
    
    # Mélange final des données
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    print("Génération des données synthétiques...")
    
    # Création du dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    # Génération des données
    df = generate_dataset(n_normal=100000, n_fraud=1000)
    
    # Sauvegarde des données
    output_file = 'data/creditcard.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Données générées et sauvegardées dans {output_file}")
    print(f"Forme du dataset : {df.shape}")
    print(f"Nombre de fraudes : {df['Class'].sum()}")
    print(f"Pourcentage de fraudes : {100 * df['Class'].mean():.2f}%")
    
    # Affichage de quelques statistiques
    print("\nStatistiques des features :")
    print(df.describe().round(2).T)

if __name__ == "__main__":
    main() 