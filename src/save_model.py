import os
import joblib
from models import FraudDetector
from data_loader import DataLoader

def main():
    # Création du dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Chargement des données
    print("Chargement des données...")
    data_loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.get_data()
    
    # Création et entraînement du modèle
    print("Entraînement du modèle...")
    detector = FraudDetector(contamination=0.01)
    detector.train(X_train, y_train)
    
    # Sauvegarde du modèle
    print("Sauvegarde du modèle...")
    detector.save('models')
    
    # Évaluation rapide
    metrics = detector.evaluate(X_test, y_test)
    print("\nRésultats :")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 