"""Fraud detection models: logistic/boosted baselines and causal temporal GraphSAGE GNN.

Models provided:
1. `LogisticFraudBaseline`: StandardScaler + class-weighted LogisticRegression on graph features
   (and optional tabular-only reference variant for ablation).
2. `BoostedFraudBaseline`: HistGradientBoostingClassifier with imbalance-aware sample weights.
3. `TemporalGraphSAGEModel`: Deterministic 2-layer causal message-passing GNN in NumPy.
   Constructs strictly lower-triangular temporal neighbor adjacency (edge j->i only when t_j < t_i
   within `neighbor_window_days` sharing a device, account, or merchant) so both 1-hop and 2-hop
   message passing are mathematically incapable of leaking future edges.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from graph_fraud_engine.calibration import ProbabilityCalibrator
from graph_fraud_engine.config import BoostedModelConfig, GNNModelConfig, LogisticModelConfig
from graph_fraud_engine.features import FEATURE_NAMES, TABULAR_ONLY_FEATURE_NAMES
from graph_fraud_engine.schemas import TemporalLeakageError, TransactionRecord


def is_pyg_available() -> bool:
    """Return True if optional PyTorch Geometric dependency is importable."""
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


class LogisticFraudBaseline:
    """Standardized logistic regression baseline with linear feature attributions."""

    def __init__(
        self,
        config: Optional[LogisticModelConfig] = None,
        feature_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or LogisticModelConfig()
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.feature_indices = [FEATURE_NAMES.index(f) for f in self.feature_names]
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=seed,
        )
        self.calibrator = ProbabilityCalibrator("none")
        self.is_fitted = False

    def _select(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_indices]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticFraudBaseline":
        Xs = self.scaler.fit_transform(self._select(X_train))
        self.model.fit(Xs, y_train)
        self.is_fitted = True
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(self._select(X))
        return np.asarray(self.model.predict_proba(Xs)[:, 1], dtype=np.float64)

    def fit_calibrator(
        self, X_val: np.ndarray, y_val: np.ndarray, method: str = "platt"
    ) -> "LogisticFraudBaseline":
        self.calibrator = ProbabilityCalibrator(method=method)
        self.calibrator.fit(self.predict_raw(X_val), y_val)
        return self

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X))

    def feature_attributions(self, x_row: np.ndarray) -> Dict[str, float]:
        """Linear log-odds contribution per feature for a single transaction."""
        xs = self.scaler.transform(x_row[self.feature_indices].reshape(1, -1))[0]
        coefs = self.model.coef_[0]
        return {
            name: round(float(coefs[i] * xs[i]), 5)
            for i, name in enumerate(self.feature_names)
        }


class BoostedFraudBaseline:
    """Imbalance-weighted HistGradientBoostingClassifier baseline."""

    def __init__(
        self,
        config: Optional[BoostedModelConfig] = None,
        feature_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or BoostedModelConfig()
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.feature_indices = [FEATURE_NAMES.index(f) for f in self.feature_names]
        self.seed = seed
        self.model = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            l2_regularization=self.config.l2_regularization,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=seed,
        )
        self.calibrator = ProbabilityCalibrator("none")
        self._train_medians: Optional[np.ndarray] = None
        self.is_fitted = False

    def _select(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.feature_indices]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BoostedFraudBaseline":
        X_sel = self._select(X_train)
        self._train_medians = np.median(X_sel, axis=0)
        n_pos = max(1, int(np.sum(y_train == 1)))
        n_neg = max(1, int(np.sum(y_train == 0)))
        pos_w = min(15.0, n_neg / float(n_pos))
        sample_weight = np.where(y_train == 1, pos_w, 1.0)
        self.model.fit(X_sel, y_train, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        X_sel = self._select(X)
        return np.asarray(self.model.predict_proba(X_sel)[:, 1], dtype=np.float64)

    def fit_calibrator(
        self, X_val: np.ndarray, y_val: np.ndarray, method: str = "platt"
    ) -> "BoostedFraudBaseline":
        self.calibrator = ProbabilityCalibrator(method=method)
        self.calibrator.fit(self.predict_raw(X_val), y_val)
        return self

    def predict_calibrated(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X))

    def feature_attributions(self, x_row: np.ndarray) -> Dict[str, float]:
        """Leave-one-feature-to-median occlusion delta in predicted probability."""
        if self._train_medians is None:
            return {}
        x_sel = x_row[self.feature_indices].astype(np.float64)
        base_p = float(self.model.predict_proba(x_sel.reshape(1, -1))[0, 1])
        perturbed = np.tile(x_sel, (len(self.feature_names), 1))
        for i in range(len(self.feature_names)):
            perturbed[i, i] = self._train_medians[i]
        pert_probs = self.model.predict_proba(perturbed)[:, 1]
        return {
            name: round(float(base_p - pert_probs[i]), 5)
            for i, name in enumerate(self.feature_names)
        }


def build_causal_tx_adjacency(
    transactions: List[TransactionRecord],
    neighbor_window_days: int = 30,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Construct a row-normalized strictly causal (lower-triangular in time) transaction adjacency matrix.

    An edge j -> i exists ONLY when:
    - `tx[j].tx_timestamp < tx[i].tx_timestamp` (strictly past)
    - `tx[i].tx_timestamp - tx[j].tx_timestamp <= neighbor_window_days`
    - `tx[i]` and `tx[j]` share a device (weight 3.0), account (weight 1.5), or merchant (weight 0.5).

    Because timestamps are non-decreasing and edges require strict `<`, the matrix is strictly
    causal so 1-hop (`A @ H`) and 2-hop (`A @ (A @ H)`) message passing cannot access future nodes.
    """
    n = len(transactions)
    A = np.zeros((n, n), dtype=np.float64)
    window = timedelta(days=neighbor_window_days)

    by_device: Dict[str, List[int]] = defaultdict(list)
    by_account: Dict[str, List[int]] = defaultdict(list)
    by_merchant: Dict[str, List[int]] = defaultdict(list)

    edge_count = 0
    for i, tx in enumerate(transactions):
        t_i = tx.tx_timestamp
        min_t = t_i - window
        weights: Dict[int, float] = {}

        # Limit merchant fan-in to the most recent 12 transactions for tractability & signal focus
        for j in reversed(by_device[tx.device_id][-24:]):
            t_j = transactions[j].tx_timestamp
            if t_j < min_t:
                break
            if t_j >= t_i:
                raise TemporalLeakageError(
                    f"Causal adjacency violation: {transactions[j].tx_id} ({t_j}) >= {tx.tx_id} ({t_i})"
                )
            weights[j] = weights.get(j, 0.0) + 3.0

        for j in reversed(by_account[tx.account_id][-16:]):
            t_j = transactions[j].tx_timestamp
            if t_j < min_t:
                break
            if t_j < t_i:
                weights[j] = weights.get(j, 0.0) + 1.5

        for j in reversed(by_merchant[tx.merchant_id][-10:]):
            t_j = transactions[j].tx_timestamp
            if t_j < min_t:
                break
            if t_j < t_i:
                weights[j] = weights.get(j, 0.0) + 0.5

        if weights:
            total_w = sum(weights.values())
            for j, w in weights.items():
                A[i, j] = w / total_w
                edge_count += 1

        by_device[tx.device_id].append(i)
        by_account[tx.account_id].append(i)
        by_merchant[tx.merchant_id].append(i)

    stats = {
        "n_nodes": n,
        "n_causal_edges": edge_count,
        "avg_in_degree": round(edge_count / float(max(1, n)), 3),
    }
    return A, stats


class TemporalGraphSAGEModel:
    """Deterministic 2-layer causal GraphSAGE-style message-passing neural network.

    Forward pass (strictly causal adjacency A):
      m1 = A @ X_std
      z1 = [X_std || m1] @ W1 + b1
      h1 = ReLU(z1)
      m2 = A @ h1
      z2 = [h1 || m2] @ W2 + b2
      h2 = ReLU(z2)
      logits = h2 @ w_out + b_out
      probs = sigmoid(logits)
    """

    def __init__(
        self,
        config: Optional[GNNModelConfig] = None,
        feature_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or GNNModelConfig()
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.seed = seed
        self.scaler = StandardScaler()
        self.calibrator = ProbabilityCalibrator("none")
        self.backend = (
            "pyg"
            if (self.config.use_pyg_if_available and is_pyg_available())
            else "numpy_temporal_graphsage"
        )

        # Learned weights
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.w_out: Optional[np.ndarray] = None
        self.b_out: float = 0.0
        self.training_history: List[Dict[str, float]] = []
        self.is_fitted = False

        # Stored historical node states for inductive single-transaction inference
        self._ref_transactions: List[TransactionRecord] = []
        self._ref_X_std: Optional[np.ndarray] = None
        self._ref_H1: Optional[np.ndarray] = None

    def _init_weights(self, in_dim: int, hidden_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / (2 * in_dim)), size=(2 * in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / (2 * hidden_dim)), size=(2 * hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim, dtype=np.float64)
        self.w_out = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim,))
        self.b_out = -2.0  # prior bias aligned with imbalanced fraud rate

    def _forward(
        self, X_std: np.ndarray, A: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        assert self.W1 is not None and self.b1 is not None
        assert self.W2 is not None and self.b2 is not None
        assert self.w_out is not None

        m1 = A @ X_std
        in1 = np.hstack([X_std, m1])
        z1 = in1 @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)

        m2 = A @ h1
        in2 = np.hstack([h1, m2])
        z2 = in2 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)

        logits = h2 @ self.w_out + self.b_out
        probs = _sigmoid(logits)
        cache = {
            "in1": in1,
            "z1": z1,
            "h1": h1,
            "in2": in2,
            "z2": z2,
            "h2": h2,
            "logits": logits,
        }
        return probs, cache

    def fit_temporal(
        self,
        all_transactions: List[TransactionRecord],
        all_X: np.ndarray,
        train_indices: np.ndarray,
        train_y: np.ndarray,
    ) -> "TemporalGraphSAGEModel":
        """Train on `train_indices` using strictly causal message passing over `all_transactions`.

        Note: StandardScaler is fitted ONLY on `all_X[train_indices]` (zero val/test stat leakage).
        Because `A` is strictly lower-triangular in time, nodes in `train_indices` can only
        aggregate from strictly earlier nodes.
        """
        self.scaler.fit(all_X[train_indices])
        X_std = np.clip(self.scaler.transform(all_X), -8.0, 8.0)
        A, _ = build_causal_tx_adjacency(
            all_transactions, neighbor_window_days=self.config.neighbor_window_days
        )

        # Slice strictly to the prefix up to the maximum train index so no future nodes are even in memory during training
        max_train_idx = int(np.max(train_indices)) + 1
        X_tr_all = X_std[:max_train_idx]
        A_tr_all = A[:max_train_idx, :max_train_idx]
        A_tr_T = A_tr_all.T

        in_dim = X_tr_all.shape[1]
        h_dim = self.config.hidden_dim
        self._init_weights(in_dim, h_dim)

        mask = np.zeros(max_train_idx, dtype=np.float64)
        mask[train_indices] = 1.0
        y_full = np.zeros(max_train_idx, dtype=np.float64)
        y_full[train_indices] = train_y.astype(np.float64)

        pos_w = self.config.pos_weight
        sample_w = np.where(y_full == 1.0, pos_w, 1.0) * mask
        denom = float(np.sum(sample_w)) + 1e-8

        # Adam state
        params = [self.W1, self.b1, self.W2, self.b2, self.w_out, np.array([self.b_out])]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        self.training_history = []
        for epoch in range(1, self.config.epochs + 1):
            probs, cache = self._forward(X_tr_all, A_tr_all)

            # Weighted binary cross-entropy loss on matured training nodes only
            p_clip = np.clip(probs, 1e-7, 1.0 - 1e-7)
            loss_vec = -(y_full * np.log(p_clip) + (1.0 - y_full) * np.log(1.0 - p_clip))
            data_loss = float(np.sum(sample_w * loss_vec) / denom)
            reg_loss = 0.5 * wd * float(
                np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.w_out ** 2)
            )

            # Backpropagation through both causal message-passing layers
            d_logits = sample_w * (probs - y_full) / denom  # (N,)
            d_w_out = cache["h2"].T @ d_logits + wd * self.w_out
            d_b_out = np.array([float(np.sum(d_logits))])

            d_h2 = np.outer(d_logits, self.w_out)
            d_z2 = d_h2 * (cache["z2"] > 0.0)
            d_W2 = cache["in2"].T @ d_z2 + wd * self.W2
            d_b2 = np.sum(d_z2, axis=0)

            d_in2 = d_z2 @ self.W2.T
            d_h1_self = d_in2[:, :h_dim]
            d_m2 = d_in2[:, h_dim:]
            d_h1 = d_h1_self + (A_tr_T @ d_m2)

            d_z1 = d_h1 * (cache["z1"] > 0.0)
            d_W1 = cache["in1"].T @ d_z1 + wd * self.W1
            d_b1 = np.sum(d_z1, axis=0)

            grads = [d_W1, d_b1, d_W2, d_b2, d_w_out, d_b_out]
            for idx_p, (p_arr, g_arr) in enumerate(zip(params, grads)):
                m_adam[idx_p] = beta1 * m_adam[idx_p] + (1.0 - beta1) * g_arr
                v_adam[idx_p] = beta2 * v_adam[idx_p] + (1.0 - beta2) * (g_arr ** 2)
                m_hat = m_adam[idx_p] / (1.0 - beta1 ** epoch)
                v_hat = v_adam[idx_p] / (1.0 - beta2 ** epoch)
                p_arr -= lr * m_hat / (np.sqrt(v_hat) + eps)

            self.b_out = float(params[5][0])
            if epoch == 1 or epoch % 15 == 0 or epoch == self.config.epochs:
                self.training_history.append(
                    {"epoch": float(epoch), "loss": round(data_loss + reg_loss, 6)}
                )

        # Store reference inductive cache for full history
        self._update_reference_cache(all_transactions, all_X)
        self.is_fitted = True
        return self

    def _update_reference_cache(
        self, transactions: List[TransactionRecord], X: np.ndarray
    ) -> None:
        X_std = np.clip(self.scaler.transform(X), -8.0, 8.0)
        A, _ = build_causal_tx_adjacency(
            transactions, neighbor_window_days=self.config.neighbor_window_days
        )
        _, cache = self._forward(X_std, A)
        self._ref_transactions = list(transactions)
        self._ref_X_std = X_std
        self._ref_H1 = cache["h1"]

    def predict_raw_temporal(
        self, transactions: List[TransactionRecord], X: np.ndarray
    ) -> np.ndarray:
        """Predict raw fraud probabilities for a chronological list of transactions."""
        X_std = np.clip(self.scaler.transform(X), -8.0, 8.0)
        A, _ = build_causal_tx_adjacency(
            transactions, neighbor_window_days=self.config.neighbor_window_days
        )
        probs, _ = self._forward(X_std, A)
        return probs

    def fit_calibrator(
        self, raw_val_probs: np.ndarray, y_val: np.ndarray, method: str = "platt"
    ) -> "TemporalGraphSAGEModel":
        self.calibrator = ProbabilityCalibrator(method=method)
        self.calibrator.fit(raw_val_probs, y_val)
        return self

    def predict_calibrated_temporal(
        self, transactions: List[TransactionRecord], X: np.ndarray
    ) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw_temporal(transactions, X))

    def predict_single_inductive(
        self, tx: TransactionRecord, x_row: np.ndarray
    ) -> Tuple[float, float, Dict[str, float]]:
        """Inductively score a single (possibly unseen/cold-start) transaction using strictly prior reference states.

        Returns `(raw_prob, calibrated_prob, feature_attributions)`.
        """
        assert self.W1 is not None and self.b1 is not None
        assert self.W2 is not None and self.b2 is not None
        assert self.w_out is not None

        x_std = np.clip(self.scaler.transform(x_row.reshape(1, -1))[0], -8.0, 8.0)
        window = timedelta(days=self.config.neighbor_window_days)
        min_t = tx.tx_timestamp - window

        # Aggregate strictly prior neighbors from reference cache
        neighbor_weights: List[Tuple[int, float]] = []
        if self._ref_X_std is not None and self._ref_H1 is not None:
            for idx, ref_tx in enumerate(self._ref_transactions):
                if ref_tx.tx_timestamp >= tx.tx_timestamp or ref_tx.tx_timestamp < min_t:
                    continue
                if ref_tx.tx_id == tx.tx_id:
                    continue
                w = 0.0
                if ref_tx.device_id == tx.device_id:
                    w += 3.0
                if ref_tx.account_id == tx.account_id:
                    w += 1.5
                if ref_tx.merchant_id == tx.merchant_id:
                    w += 0.5
                if w > 0.0:
                    neighbor_weights.append((idx, w))

        if neighbor_weights and self._ref_X_std is not None and self._ref_H1 is not None:
            idxs = [i for i, _ in neighbor_weights]
            ws = np.array([w for _, w in neighbor_weights], dtype=np.float64)
            ws /= np.sum(ws)
            m1 = ws @ self._ref_X_std[idxs]
            m2 = ws @ self._ref_H1[idxs]
        else:
            m1 = np.zeros_like(x_std)
            m2 = np.zeros(self.config.hidden_dim, dtype=np.float64)

        def _score_from_x_std(x_s: np.ndarray) -> float:
            in1 = np.concatenate([x_s, m1])
            h1 = np.maximum(0.0, in1 @ self.W1 + self.b1)
            in2 = np.concatenate([h1, m2])
            h2 = np.maximum(0.0, in2 @ self.W2 + self.b2)
            logit = float(h2 @ self.w_out + self.b_out)
            return float(_sigmoid(np.array([logit]))[0])

        raw_p = _score_from_x_std(x_std)
        cal_p = float(self.calibrator.transform(np.array([raw_p]))[0])

        # Feature attribution via zero-baseline (standardized mean) occlusion
        attributions: Dict[str, float] = {}
        for i, name in enumerate(self.feature_names):
            x_pert = x_std.copy()
            x_pert[i] = 0.0
            attributions[name] = round(raw_p - _score_from_x_std(x_pert), 5)

        return raw_p, cal_p, attributions


def get_tabular_feature_names() -> List[str]:
    return list(TABULAR_ONLY_FEATURE_NAMES)
