import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class FraudDetectorXGB:
    def __init__(self, contamination=0.001):
        self.model = None
        self.scaler = StandardScaler()
        self.contamination = contamination
        
    def _create_features(self, X):
        """Crée des features supplémentaires."""
        df = X.copy()
        
        # Features temporelles
        df['hour'] = df['Time'] % 24
        df['day'] = df['Time'] // 24
        
        # Features statistiques sur les composantes principales
        pca_cols = [col for col in df.columns if col.startswith('V')]
        
        # Moyennes et écarts-types par composante
        df['pca_mean'] = df[pca_cols].mean(axis=1)
        df['pca_std'] = df[pca_cols].std(axis=1)
        
        # Somme des valeurs absolues des composantes
        df['pca_abs_sum'] = df[pca_cols].abs().sum(axis=1)
        
        # Nombre de composantes positives/négatives
        df['pca_positive_count'] = (df[pca_cols] > 0).sum(axis=1)
        df['pca_negative_count'] = (df[pca_cols] < 0).sum(axis=1)
        
        # Ratio des composantes positives/négatives
        df['pca_positive_ratio'] = df['pca_positive_count'] / len(pca_cols)
        
        # Features de transaction par heure
        df['transactions_per_hour'] = df.groupby('hour')['Time'].transform('count')
        
        return df
    
    def _create_model(self):
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            scale_pos_weight=100,  # Pour gérer le déséquilibre
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train=None):
        """Entraîne le modèle."""
        print("Prétraitement des données...")
        X_train_processed = self._create_features(X_train)
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        
        print("Entraînement du modèle...")
        self.model = self._create_model()
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_train_scaled, y_train)],
            verbose=True
        )
    
    def predict(self, X):
        """Prédit les anomalies."""
        X_processed = self._create_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """Évalue les performances du modèle."""
        scores = self.predict(X_test)
        
        # Optimisation du seuil
        thresholds = np.linspace(0.1, 0.9, 100)
        best_f1 = 0
        best_threshold = None
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_test, predictions, average='binary'
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Utilisation du meilleur seuil
        predictions = (scores > best_threshold).astype(int)
        
        # Calcul des métriques finales
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary'
        )
        auc = roc_auc_score(y_test, scores)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': best_threshold
        }
    
    def save(self, path):
        """Sauvegarde le modèle."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, f"{path}/model_xgb.joblib")
    
    def load(self, path):
        """Charge le modèle."""
        model_data = joblib.load(f"{path}/model_xgb.joblib")
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 