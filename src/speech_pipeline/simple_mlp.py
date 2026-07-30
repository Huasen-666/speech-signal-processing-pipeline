from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values)


def relu_grad(activations: np.ndarray) -> np.ndarray:
    return (activations > 0.0).astype(np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    encoded = np.zeros((len(labels), class_count), dtype=np.float64)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def cross_entropy(probs: np.ndarray, encoded_labels: np.ndarray) -> float:
    return float(-np.mean(np.sum(encoded_labels * np.log(probs + 1e-12), axis=1)))


def standardize(
    values: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0)
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / safe_std, mean, safe_std


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    seed: int = 42


class MLP:
    """Small one-hidden-layer MLP with Adam."""

    def __init__(self, config: MLPConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        hidden_scale = np.sqrt(2.0 / config.input_dim)
        output_scale = np.sqrt(2.0 / config.hidden_dim)
        self.weights = [
            rng.normal(0.0, hidden_scale, size=(config.input_dim, config.hidden_dim)),
            rng.normal(0.0, output_scale, size=(config.hidden_dim, config.output_dim)),
        ]
        self.biases = [
            np.zeros(config.hidden_dim, dtype=np.float64),
            np.zeros(config.output_dim, dtype=np.float64),
        ]
        self.m_weights = [np.zeros_like(weight) for weight in self.weights]
        self.v_weights = [np.zeros_like(weight) for weight in self.weights]
        self.m_biases = [np.zeros_like(bias) for bias in self.biases]
        self.v_biases = [np.zeros_like(bias) for bias in self.biases]
        self.step = 0
        self.cache: list[np.ndarray] = []

    def forward(self, features: np.ndarray) -> np.ndarray:
        hidden_linear = features @ self.weights[0] + self.biases[0]
        hidden = relu(hidden_linear)
        logits = hidden @ self.weights[1] + self.biases[1]
        probs = softmax(logits)
        self.cache = [features, hidden, probs]
        return probs

    def backward(self, encoded_labels: np.ndarray) -> None:
        features, hidden, probs = self.cache
        batch_size = len(features)
        delta_out = (probs - encoded_labels) / max(batch_size, 1)

        grad_w2 = hidden.T @ delta_out
        grad_b2 = delta_out.sum(axis=0)
        delta_hidden = (delta_out @ self.weights[1].T) * relu_grad(hidden)
        grad_w1 = features.T @ delta_hidden
        grad_b1 = delta_hidden.sum(axis=0)

        if self.config.weight_decay > 0.0:
            grad_w1 += self.config.weight_decay * self.weights[0]
            grad_w2 += self.config.weight_decay * self.weights[1]

        self._adam_update([grad_w1, grad_w2], [grad_b1, grad_b2])

    def _adam_update(self, grad_weights: list[np.ndarray], grad_biases: list[np.ndarray]) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.step += 1

        for idx in range(len(self.weights)):
            self.m_weights[idx] = beta1 * self.m_weights[idx] + (1.0 - beta1) * grad_weights[idx]
            self.v_weights[idx] = beta2 * self.v_weights[idx] + (1.0 - beta2) * (grad_weights[idx] ** 2)
            self.m_biases[idx] = beta1 * self.m_biases[idx] + (1.0 - beta1) * grad_biases[idx]
            self.v_biases[idx] = beta2 * self.v_biases[idx] + (1.0 - beta2) * (grad_biases[idx] ** 2)

            m_w_hat = self.m_weights[idx] / (1.0 - beta1**self.step)
            v_w_hat = self.v_weights[idx] / (1.0 - beta2**self.step)
            m_b_hat = self.m_biases[idx] / (1.0 - beta1**self.step)
            v_b_hat = self.v_biases[idx] / (1.0 - beta2**self.step)

            self.weights[idx] -= self.config.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[idx] -= self.config.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def train_epoch(
        self,
        features: np.ndarray,
        encoded_labels: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        losses = []
        indices = rng.permutation(len(features))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            probs = self.forward(features[batch_indices])
            losses.append(cross_entropy(probs, encoded_labels[batch_indices]))
            self.backward(encoded_labels[batch_indices])
        return float(np.mean(losses))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        hidden = relu(features @ self.weights[0] + self.biases[0])
        return softmax(hidden @ self.weights[1] + self.biases[1])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    def snapshot(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [weight.copy() for weight in self.weights], [bias.copy() for bias in self.biases]

    def restore(self, snapshot: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        weights, biases = snapshot
        self.weights = [weight.copy() for weight in weights]
        self.biases = [bias.copy() for bias in biases]

    def to_dict(self) -> dict:
        return {
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "seed": self.config.seed,
            },
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
        }
