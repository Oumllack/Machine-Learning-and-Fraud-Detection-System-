import pandas as pd
import numpy as np
from models import FraudDetectorXGB
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_model():
    """Charge le modèle entraîné."""
    model = FraudDetectorXGB()
    model.load('models')
    return model

def test_random_transactions(model, data, n_samples=5):
    """Teste le modèle sur des transactions aléatoires."""
    print("\nTest sur transactions aléatoires :")
    print("==================================")
    
    # Sélection aléatoire
    indices = np.random.choice(len(data), n_samples, replace=False)
    samples = data.iloc[indices]
    
    # Prédictions
    X = samples.drop('Class', axis=1)
    scores = model.predict(X)
    
    # Affichage des résultats
    for i, (idx, row) in enumerate(samples.iterrows()):
        print(f"\nTransaction {i+1}:")
        print(f"Score de fraude : {scores[i]:.3f}")
        print(f"Classe réelle : {'Fraude' if row['Class'] == 1 else 'Normal'}")
        print(f"Heure : {row['Time'] // 3600:.0f}h")
        print("Caractéristiques principales :")
        for col in ['V1', 'V2', 'V3', 'V4', 'V5']:
            print(f"- {col}: {row[col]:.3f}")

def plot_time_analysis(model, data):
    """Analyse les prédictions en fonction du temps."""
    print("\nAnalyse temporelle :")
    print("===================")
    
    X = data.drop('Class', axis=1)
    scores = model.predict(X)
    
    # Conversion du temps en heures
    hours = data['Time'] // 3600
    
    # Moyenne des scores par heure
    hourly_scores = pd.DataFrame({
        'Hour': hours,
        'Score': scores,
        'Is_Fraud': data['Class']
    }).groupby('Hour').agg({
        'Score': 'mean',
        'Is_Fraud': 'sum'
    }).reset_index()
    
    # Création du graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Scores moyens par heure
    ax1.plot(hourly_scores['Hour'], hourly_scores['Score'], 'b-')
    ax1.set_title('Score moyen de fraude par heure')
    ax1.set_xlabel('Heure')
    ax1.set_ylabel('Score moyen')
    
    # Nombre de fraudes par heure
    ax2.bar(hourly_scores['Hour'], hourly_scores['Is_Fraud'])
    ax2.set_title('Nombre de fraudes par heure')
    ax2.set_xlabel('Heure')
    ax2.set_ylabel('Nombre de fraudes')
    
    plt.tight_layout()
    plt.savefig('results/time_analysis.png')
    plt.close()

def analyze_feature_importance(model, data):
    """Analyse l'importance des features."""
    print("\nAnalyse des features importantes :")
    print("================================")
    
    # Création des features
    X = data.drop('Class', axis=1)
    X_processed = model._create_features(X)
    
    # Récupération de l'importance des features
    feature_importance = pd.DataFrame({
        'Feature': X_processed.columns,
        'Importance': model.model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Affichage des 10 features les plus importantes
    print("\nTop 10 des features les plus importantes :")
    for _, row in feature_importance.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.3f}")
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))
    plt.xticks(rotation=45)
    plt.title('Top 10 des features les plus importantes')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    plt.close()

def main():
    # Création du dossier results s'il n'existe pas
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print("Chargement des données...")
    data = pd.read_csv('data/creditcard.csv')
    
    print("Chargement du modèle...")
    model = load_model()
    
    # Tests
    test_random_transactions(model, data)
    plot_time_analysis(model, data)
    analyze_feature_importance(model, data)

if __name__ == "__main__":
    main() 