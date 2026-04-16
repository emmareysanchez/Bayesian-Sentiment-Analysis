"""
Implementa Latent Dirichlet Allocation (LDA) para el clustering suave de temas.
"""
import joblib
import logging
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

log = logging.getLogger(__name__)

class TopicModeler:
    """
    Encapsula el modelo LDA para extraer la distribución de temas (theta).
    """
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        # El parámetro 'doc_topic_prior' es el alpha de la distribución de Dirichlet
        self.lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=random_state,
            learning_method='online',
            doc_topic_prior=1/n_components # Prior simétrico
        )

    def fit(self, tfidf_matrix):
        log.info(f"Entrenando LDA con K={self.n_components} temas...")
        self.lda.fit(tfidf_matrix)
        log.info("Entrenamiento de LDA completado.")

    def get_topics(self, tfidf_matrix) -> np.ndarray:
        """
        Retorna la matriz theta (N x K) donde cada fila es una 
        distribución de probabilidad sobre los temas.
        """
        return self.lda.transform(tfidf_matrix)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)