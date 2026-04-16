"""
Implementación de una Red Neuronal Bayesiana usando Pyro para Inferencia Variacional.
"""
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianClassifier(PyroModule):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
        # Capa oculta 1: Pesos como distribuciones Normales (Prior)
        self.fc1 = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden_dim]).to_event(1))
        
        # Capa de salida: Clasificación binaria (Positivo/Negativo)
        self.out = PyroModule[nn.Linear](hidden_dim, 2)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([2, hidden_dim]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([2]).to_event(1))
        
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        # x es el vector concatenado [BERT, LDA]
        x = self.relu(self.fc1(x))
        logits = self.out(x)
        
        # Verosimilitud (Likelihood): Distribución Categórica para la clasificación
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits