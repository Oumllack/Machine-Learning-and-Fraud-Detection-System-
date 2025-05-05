import os
import json
from data_loader import DataLoader
from models import FraudDetectorXGB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from datetime import datetime

def plot_metrics(metrics, save_path):
    """Visualise les métriques du modèle."""
    metrics_to_plot = {k: v for k, v in metrics.items() if k not in ['threshold', 'best_trial', 'best_value']}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics_to_plot.keys()), y=list(metrics_to_plot.values()))
    plt.title('Métriques du modèle combiné (Isolation Forest + Autoencoder)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics.png'))
    plt.close()

def plot_score_distribution(detector, X_test, y_test, save_path):
    """Visualise la distribution des scores pour les transactions normales et frauduleuses."""
    scores = detector.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({'Score': scores, 'Classe': y_test}), 
                x='Score', hue='Classe', bins=50)
    plt.axvline(x=metrics['threshold'], color='r', linestyle='--', 
                label=f'Seuil optimal: {metrics["threshold"]:.3f}')
    plt.title('Distribution des scores de fraude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'score_distribution.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Visualise la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    """Visualise l'importance des features."""
    importance = model.model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title('Importance des features')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

def main():
    # Création du dossier results s'il n'existe pas
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print("Chargement des données...")
    data = pd.read_csv('data/creditcard.csv')
    
    # Séparation des features et de la target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Nombre d'échantillons d'entraînement : {len(X_train)}")
    print(f"Nombre d'échantillons de test : {len(X_test)}")
    print(f"Nombre de fraudes dans l'ensemble d'entraînement : {y_train.sum()}")
    print(f"Nombre de fraudes dans l'ensemble de test : {y_test.sum()}")
    
    # Initialisation et entraînement du modèle
    print("\nInitialisation du modèle...")
    model = FraudDetectorXGB(contamination=0.001)
    
    print("Entraînement du modèle...")
    model.train(X_train, y_train)
    
    # Évaluation
    print("\nÉvaluation du modèle...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nRésultats :")
    print(f"Précision : {metrics['precision']:.3f}")
    print(f"Rappel : {metrics['recall']:.3f}")
    print(f"F1-score : {metrics['f1']:.3f}")
    print(f"AUC : {metrics['auc']:.3f}")
    print(f"Seuil optimal : {metrics['threshold']:.3f}")
    
    # Sauvegarde des résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/metrics_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Métriques du modèle XGBoost\n")
        f.write("==========================\n\n")
        f.write(f"Précision : {metrics['precision']:.3f}\n")
        f.write(f"Rappel : {metrics['recall']:.3f}\n")
        f.write(f"F1-score : {metrics['f1']:.3f}\n")
        f.write(f"AUC : {metrics['auc']:.3f}\n")
        f.write(f"Seuil optimal : {metrics['threshold']:.3f}\n")
    
    # Sauvegarde du modèle
    model.save('models')
    print(f"\nModèle sauvegardé dans le dossier 'models'")
    print(f"Métriques sauvegardées dans {results_file}")

if __name__ == "__main__":
    main() 