import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataLoader:
    def __init__(self, data_path="data/creditcard.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Charge les données depuis le fichier CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas.")
        
        print("Chargement des données...")
        df = pd.read_csv(self.data_path)
        return df
    
    def preprocess_data(self, df):
        """Prétraite les données pour l'entraînement."""
        print("Prétraitement des données...")
        
        # Séparation des features et de la cible
        X = df.drop(['Time', 'Class'], axis=1)
        y = df['Class']
        
        # Standardisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """Divise les données en ensembles d'entraînement, validation et test."""
        # Division train/test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Division train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size/(1-test_size), 
            random_state=42, 
            stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data(self):
        """Charge, prétraite et divise les données."""
        df = self.load_data()
        X, y = self.preprocess_data(df)
        return self.split_data(X, y) 