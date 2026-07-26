import numpy as np
import scipy.linalg as la

class NonLinearInvariantCausalEngine:
    def __init__(self, num_signals, max_lags=3, train_window=60, latent_factors=1, learning_rate=0.01, epochs=15):
        self.num_signals = num_signals
        self.max_lags = max_lags
        self.train_window = train_window
        self.latent_factors = latent_factors
        self.lr = learning_rate
        self.epochs = epochs
        
        self.buffer = []
        self.historical_causal_graphs = []

    def step(self, current_signals):
        self.buffer.append(current_signals)
        required_history = self.train_window + self.max_lags
        
        if len(self.buffer) < required_history:
            return None
        if len(self.buffer) > required_history:
            self.buffer.pop(0)
            
        all_data = np.array(self.buffer)
        train_data = all_data[:-1, :]    
        
        # 1. Online Latent Factor De-confounding via SVD
        mean_centered = train_data - np.mean(train_data, axis=0)
        U, S, Vt = la.svd(mean_centered, full_matrices=False)
        
        if self.latent_factors > 0:
            S_latent = np.zeros_like(S)
            S_latent[:self.latent_factors] = S[:self.latent_factors]
            Z = U @ np.diag(S_latent) @ Vt
            deconfounded_data = mean_centered - Z
        else:
            deconfounded_data = mean_centered

        # 2. Extract Lag Matrix
        N = len(deconfounded_data) - self.max_lags
        K = self.num_signals
        X = np.zeros((N, K, self.max_lags))
        Y = np.zeros((N, K))
        
        for t in range(N):
            for l in range(1, self.max_lags + 1):
                X[t, :, l-1] = deconfounded_data[t + self.max_lags - l, :]
            Y[t, :] = deconfounded_data[t + self.max_lags, :]
            
        causal_matrix = np.zeros((K, K))
        final_predictions = np.zeros(K)
        hidden_dim = 8
        
        # 3. Localized MLP Training per Channel
        for j in range(K):
            y_j = Y[:, j].reshape(-1, 1)
            np.random.seed(42 + j)
            W_in_j = np.random.randn(K, self.max_lags, hidden_dim) * 0.1
            W_hid_j = np.random.randn(hidden_dim, 1) * 0.1
            
            for epoch in range(self.epochs):
                X_flat = np.zeros((N, hidden_dim))
                for i in range(K):
                    X_flat += (X[:, i, :] @ W_in_j[i, :, :])
                
                hidden_layer = np.tanh(X_flat)
                y_pred = hidden_layer @ W_hid_j
                
                loss_grad = 2 * (y_pred - y_j) / N
                grad_hid = hidden_layer.T @ loss_grad
                W_hid_j -= self.lr * grad_hid
                
                d_hidden = (loss_grad @ W_hid_j.T) * (1 - hidden_layer**2)
                for i in range(K):
                    grad_in_i = (X[:, i, :].T @ d_hidden)
                    W_in_j[i, :, :] -= self.lr * grad_in_i
            
            for i in range(K):
                if i != j:
                    weight_mag = np.linalg.norm(W_in_j[i, :, :])
                    causal_matrix[i, j] = weight_mag if weight_mag > 0.01 else 0.0
            
            current_feat = X[-1, :, :]
            X_flat_next = np.zeros((1, hidden_dim))
            for i in range(K):
                X_flat_next += (current_feat[i, :].reshape(1, -1) @ W_in_j[i, :, :])
            final_predictions[j] = np.tanh(X_flat_next) @ W_hid_j
            
        self.historical_causal_graphs.append(causal_matrix)
        return causal_matrix, final_predictions

    def _compute_structural_invariance(self):
        if len(self.historical_causal_graphs) < 2:
            return 1.0
        g1 = self.historical_causal_graphs[-1].flatten()
        g2 = self.historical_causal_graphs[-2].flatten()
        norm1, norm2 = np.linalg.norm(g1), np.linalg.norm(g2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(np.dot(g1, g2) / (norm1 * norm2))
