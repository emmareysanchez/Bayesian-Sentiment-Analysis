"""
src/inference/variational.py
Implementación del entrenamiento mediante Inferencia Variacional Estocástica (SVI).
"""
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal
import logging

log = logging.getLogger(__name__)

class VariationalInference:
    def __init__(self, model, lr=0.01):
        """
        Args:
            model: La instancia de BayesianClassifier (BNN)
            lr: Tasa de aprendizaje para el optimizador Adam
        """
        self.model = model
        
        # 1. Definimos la Guía (Posterior Aproximada q(w))
        # AutoDiagonalNormal asume que los pesos siguen distribuciones normales 
        # sin correlación entre ellos (campo medio).
        self.guide = AutoDiagonalNormal(model)
        
        # 2. Optimizador específico de Pyro
        self.optimizer = Adam({"lr": lr})
        
        # 3. Definimos la SVI con la pérdida ELBO
        # La ELBO maximiza la verosimilitud de los datos y minimiza la KL-Divergence
        self.svi = SVI(model, self.guide, self.optimizer, loss=Trace_ELBO())

    def train_step(self, x_batch, y_batch):
        """
        Performs one optimization step and returns the average ELBO loss per sample.
        """
        loss = self.svi.step(x_batch, y_batch.long())
        return loss / x_batch.shape[0]

    def evaluate_loss(self, data_loader):
        """
        Calcula la pérdida ELBO acumulada en un conjunto de datos (val/test).
        """
        total_loss = 0
        for x_batch, y_batch in data_loader:
            # Transformamos y_batch a long para la clasificación
            total_loss += self.svi.evaluate_loss(x_batch, y_batch.long())
        return total_loss / len(data_loader.dataset)

    def get_posterior_predictive(self):
        """
        Retorna la guía entrenada para poder realizar predicciones 
        muestreando de la posterior calculada.
        """
        return self.guide