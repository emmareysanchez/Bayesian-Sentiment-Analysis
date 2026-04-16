"""
Orquestador que conecta el Clustering de Dirichlet (LDA) con la BNN.
"""
import torch
import numpy as np
import logging
from src.models.lda import TopicModeler
from src.models.bnn import BayesianClassifier

log = logging.getLogger(__name__)

class SentimentPipeline:
    def __init__(self, input_dim_bnn, lda_model: TopicModeler, bnn_model: BayesianClassifier):
        """
        Args:
            input_dim_bnn: Suma de dim_bert (768) + n_topics (K)
            lda_model: Instancia entrenada de TopicModeler
            bnn_model: Instancia de BayesianClassifier (PyroModule)
        """
        self.lda = lda_model
        self.bnn = bnn_model
        self.input_dim_bnn = input_dim_bnn

    def get_combined_features(self, bert_embeddings, tfidf_matrix):
        """
        Concatena semántica (BERT) y temática (LDA).
        """
        # 1. Obtener distribución de temas (theta) del LDA
        # theta shape: (N, K)
        theta = self.lda.get_topics(tfidf_matrix)
        
        # 2. Concatenar con BERT embeddings
        # bert_embeddings shape: (N, 768)
        combined = np.hstack((bert_embeddings, theta))
        
        return torch.tensor(combined, dtype=torch.float32)

    def predict_with_uncertainty(self, x_combined, num_samples=100):
        """
        Realiza inferencia estocástica (Monte Carlo) para obtener 
        la media de la predicción y la incertidumbre.
        """
        preds = []
        for _ in range(num_samples):
            # Al ser una BNN, cada 'forward' muestrea pesos distintos de la posterior
            logits = self.bnn(x_combined)
            probs = torch.softmax(logits, dim=-1)
            preds.append(probs)
        
        preds = torch.stack(preds)
        
        # Media de las predicciones (p(y|x))
        mean_probs = preds.mean(dim=0)
        
        # Incertidumbre Epistémica (Varianza de las predicciones)
        # Un valor alto indica que el modelo "duda" porque no conoce el patrón
        epistemic_uncertainty = preds.var(dim=0).mean(dim=-1)
        
        return mean_probs, epistemic_uncertainty