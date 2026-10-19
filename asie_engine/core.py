"""Nonlinear lagged predictor with an explicitly predictive weight matrix.

The matrix returned by this module is NOT a causally identified graph. Its
entries summarize trained input-weight magnitudes for predicting target
channels. The matrix convention is C[target, source]: row j is the target
channel and column i is the source channel.
"""

import numpy as np
import scipy.linalg as la


class NonLinearInvariantCausalEngine:
    """Experimental nonlinear lagged predictor (not a causal identifier)."""

    def __init__(
        self,
        num_signals,
        max_lags=3,
        train_window=60,
        latent_factors=1,
        learning_rate=0.01,
        epochs=15,
        random_state=42,
    ):
        for name, value in (("num_signals", num_signals), ("max_lags", max_lags),
                            ("train_window", train_window), ("epochs", epochs)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer.")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if train_window < 2:
            raise ValueError("train_window must be at least 2.")
        if isinstance(latent_factors, (bool, np.bool_)) or not isinstance(latent_factors, (int, np.integer)):
            raise ValueError("latent_factors must be a non-negative integer.")
        if latent_factors < 0:
            raise ValueError("latent_factors must be a non-negative integer.")
        if isinstance(learning_rate, (bool, np.bool_)):
            raise ValueError("learning_rate must be a finite positive number.")
        try:
            learning_rate = float(learning_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("learning_rate must be a finite positive number.") from exc
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number.")
        if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer, type(None))):
            raise ValueError("random_state must be an integer or None.")

        self.num_signals = int(num_signals)
        self.max_lags = int(max_lags)
        self.train_window = int(train_window)
        self.latent_factors = int(latent_factors)
        self.lr = learning_rate
        self.epochs = int(epochs)
        self.rng = np.random.default_rng(random_state)
        self.buffer = []
        self.historical_predictive_weight_matrices = []

    def step(self, current_signals):
        """Ingest one observation; return predictive weights and next prediction.

        Returns None until enough history is available. Matrix entries use
        C[target, source] and are predictive model-weight magnitudes, not
        identified causal effects.
        """
        try:
            current_signals = np.asarray(current_signals, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current_signals must be a finite numeric vector.") from exc
        if current_signals.ndim != 1 or current_signals.shape[0] != self.num_signals:
            raise ValueError(f"current_signals must have shape ({self.num_signals},).")
        if not np.isfinite(current_signals).all():
            raise ValueError("current_signals must contain only finite values.")

        self.buffer.append(current_signals.copy())
        required_history = self.train_window + self.max_lags
        if len(self.buffer) < required_history:
            return None
        if len(self.buffer) > required_history:
            self.buffer.pop(0)

        all_data = np.asarray(self.buffer, dtype=float)
        train_data = all_data[:-1, :]
        # Experimental latent-factor removal is not a causal deconfounding guarantee.
        mean_centered = train_data - np.mean(train_data, axis=0)
        U, S, Vt = la.svd(mean_centered, full_matrices=False)
        if self.latent_factors > 0:
            S_latent = np.zeros_like(S)
            S_latent[:self.latent_factors] = S[:self.latent_factors]
            Z = U @ np.diag(S_latent) @ Vt
            deconfounded_data = mean_centered - Z
        else:
            deconfounded_data = mean_centered

        N = len(deconfounded_data) - self.max_lags
        K = self.num_signals
        X = np.zeros((N, K, self.max_lags))
        Y = np.zeros((N, K))
        for t in range(N):
            for lag in range(1, self.max_lags + 1):
                X[t, :, lag - 1] = deconfounded_data[t + self.max_lags - lag, :]
            Y[t, :] = deconfounded_data[t + self.max_lags, :]

        predictive_weight_matrix = np.zeros((K, K))
        final_predictions = np.zeros(K)
        hidden_dim = 8
        for target in range(K):
            y_target = Y[:, target].reshape(-1, 1)
            W_in = self.rng.standard_normal((K, self.max_lags, hidden_dim)) * 0.1
            W_hidden = self.rng.standard_normal((hidden_dim, 1)) * 0.1
            for _ in range(self.epochs):
                X_flat = np.zeros((N, hidden_dim))
                for source in range(K):
                    X_flat += X[:, source, :] @ W_in[source, :, :]
                hidden_layer = np.tanh(X_flat)
                y_pred = hidden_layer @ W_hidden
                loss_grad = 2 * (y_pred - y_target) / N
                grad_hidden = hidden_layer.T @ loss_grad
                W_hidden -= self.lr * grad_hidden
                d_hidden = (loss_grad @ W_hidden.T) * (1 - hidden_layer ** 2)
                for source in range(K):
                    grad_in = X[:, source, :].T @ d_hidden
                    W_in[source, :, :] -= self.lr * grad_in

            for source in range(K):
                if source != target:
                    magnitude = np.linalg.norm(W_in[source, :, :])
                    predictive_weight_matrix[target, source] = magnitude if magnitude > 0.01 else 0.0

            current_features = X[-1, :, :]
            X_flat_next = np.zeros((1, hidden_dim))
            for source in range(K):
                X_flat_next += current_features[source, :].reshape(1, -1) @ W_in[source, :, :]
            final_predictions[target] = float((np.tanh(X_flat_next) @ W_hidden).item())

        self.historical_predictive_weight_matrices.append(predictive_weight_matrix)
        return predictive_weight_matrix, final_predictions

    def _compute_structural_invariance(self):
        """Cosine similarity between consecutive predictive matrices.

        A zero-norm matrix is inconclusive and returns 0.0 instead of perfect invariance.
        """
        if len(self.historical_predictive_weight_matrices) < 2:
            return 1.0
        previous = self.historical_predictive_weight_matrices[-2].flatten()
        current = self.historical_predictive_weight_matrices[-1].flatten()
        previous_norm = np.linalg.norm(previous)
        current_norm = np.linalg.norm(current)
        if previous_norm == 0 or current_norm == 0:
            return 0.0
        score = float(np.dot(current, previous) / (current_norm * previous_norm))
        return float(np.clip(score, -1.0, 1.0))
