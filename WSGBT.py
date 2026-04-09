# -*- coding: utf-8 -*-
"""
WSGBT: Weakly Supervised Granular-Ball Tree for Anomaly Detection

A clean and minimal reference implementation for public release.

Main components:
1. Granular-ball tree construction
2. Fuzzy leaf membership modeling
3. Path-based structural deviation scoring
4. Leaf sparsity scoring
5. Weakly supervised guidance
6. Uncertainty-aware gating
7. Final anomaly score fusion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans


def minmax01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D array to [0, 1]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    mn = np.min(x)
    mx = np.max(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def robust_sigmoid(x: np.ndarray, clip: float = 12.0) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -clip, clip)
    return 1.0 / (1.0 + np.exp(-x))


def power_sharpen(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Power sharpening after min-max normalization."""
    x = minmax01(x)
    gamma = max(float(gamma), 1e-6)
    return np.power(x, gamma)


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Convert values into normalized rank scores in [0, 1]."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=float)
    order = np.argsort(np.argsort(x))
    return order / (x.size - 1.0)


def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


@dataclass
class BallNode:
    """A node in the granular-ball tree."""
    node_id: int
    sample_idx: np.ndarray
    depth: int
    center: np.ndarray
    radius: float
    parent_id: int = -1
    left: Optional["BallNode"] = None
    right: Optional["BallNode"] = None
    is_leaf: bool = True
    size: int = 0
    sse: float = 0.0


class WSGBT:
    """
    Weakly Supervised Granular-Ball Tree for anomaly detection.

    Parameters
    ----------
    min_samples_split : int
        Minimum number of samples required for node splitting.
    max_depth : int
        Maximum tree depth.
    min_sse_gain : float
        Minimum relative SSE gain required for splitting.
    min_radius : float
        Minimum node radius.
    sparse_alpha : float
        Trade-off in leaf sparsity scoring.
    depth_gamma : float
        Depth weight coefficient in path-based scoring.
    lambda_path : float
        Balance between path score and sparsity score.
    fuzzy_tau : float
        Scale parameter for fuzzy membership.
    max_leaf_for_fuzzy : Optional[int]
        If provided, only top-K leaf memberships are retained.
    eta_ws : float
        Weak supervision strength.
    ws_tau : float
        Kernel scale for weak supervision propagation.
    alpha_ws_anom : float
        Balance between anomaly similarity and normal-distance guidance.
    beta_entropy : float
        Weight for membership entropy in gating.
    beta_pathvar : float
        Weight for path variance in gating.
    tau_gate : float
        Threshold in gate computation.
    eta_mul : float
        Weight of multiplicative interaction term.
    eta_inter : float
        Weight of explicit interaction term.
    eta_tail : float
        Weight of rank-based tail enhancement.
    power_unsup : float
        Exponent for unsupervised score in interaction term.
    power_guide : float
        Exponent for guidance score in interaction term.
    gamma_tail : float
        Sharpening exponent for final score.
    w_base : float
        Weight of base score.
    w_mult : float
        Weight of multiplicative score.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        min_samples_split: int = 20,
        max_depth: int = 6,
        min_sse_gain: float = 1e-4,
        min_radius: float = 1e-8,
        sparse_alpha: float = 0.50,
        depth_gamma: float = 0.30,
        lambda_path: float = 0.68,
        fuzzy_tau: float = 0.95,
        max_leaf_for_fuzzy: Optional[int] = None,
        eta_ws: float = 1.0,
        ws_tau: float = 0.90,
        alpha_ws_anom: float = 0.60,
        beta_entropy: float = 1.0,
        beta_pathvar: float = 1.0,
        tau_gate: float = 0.0,
        eta_mul: float = 1.0,
        eta_inter: float = 1.0,
        eta_tail: float = 1.0,
        power_unsup: float = 1.0,
        power_guide: float = 1.0,
        gamma_tail: float = 1.0,
        w_base: float = 0.30,
        w_mult: float = 0.70,
        random_state: int = 42,
    ) -> None:
        self.min_samples_split = int(min_samples_split)
        self.max_depth = int(max_depth)
        self.min_sse_gain = float(min_sse_gain)
        self.min_radius = float(min_radius)

        self.sparse_alpha = float(sparse_alpha)
        self.depth_gamma = float(depth_gamma)
        self.lambda_path = float(lambda_path)
        self.fuzzy_tau = float(fuzzy_tau)
        self.max_leaf_for_fuzzy = max_leaf_for_fuzzy

        self.eta_ws = float(eta_ws)
        self.ws_tau = float(ws_tau)
        self.alpha_ws_anom = float(alpha_ws_anom)

        self.beta_entropy = float(beta_entropy)
        self.beta_pathvar = float(beta_pathvar)
        self.tau_gate = float(tau_gate)

        self.eta_mul = float(eta_mul)
        self.eta_inter = float(eta_inter)
        self.eta_tail = float(eta_tail)
        self.power_unsup = float(power_unsup)
        self.power_guide = float(power_guide)
        self.gamma_tail = float(gamma_tail)
        self.w_base = float(w_base)
        self.w_mult = float(w_mult)

        self.random_state = int(random_state)

        self.X: Optional[np.ndarray] = None
        self.root: Optional[BallNode] = None
        self.nodes: List[BallNode] = []
        self.leaf_nodes: List[BallNode] = []

        self.parent_map_: Dict[int, int] = {}
        self.node_by_id_: Dict[int, BallNode] = {}
        self.leaf_sparse_map_: Dict[int, float] = {}

        self.labeled_anom_idx_ = np.array([], dtype=int)
        self.labeled_norm_idx_ = np.array([], dtype=int)

        self.path_score_: Optional[np.ndarray] = None
        self.sparse_score_: Optional[np.ndarray] = None
        self.unsup_score_: Optional[np.ndarray] = None
        self.guide_score_: Optional[np.ndarray] = None
        self.uncertainty_score_: Optional[np.ndarray] = None
        self.path_var_score_: Optional[np.ndarray] = None
        self.gate_score_: Optional[np.ndarray] = None
        self.mult_score_: Optional[np.ndarray] = None
        self.inter_score_: Optional[np.ndarray] = None
        self.tail_rank_score_: Optional[np.ndarray] = None
        self.decision_scores_: Optional[np.ndarray] = None

    # =====================================================
    # Tree construction
    # =====================================================
    def _calc_center_radius_sse(self, X_sub: np.ndarray) -> Tuple[np.ndarray, float, float]:
        center = np.mean(X_sub, axis=0)
        d = np.linalg.norm(X_sub - center, axis=1)
        radius = np.max(d) if d.size > 0 else 0.0
        sse = np.sum(d ** 2)
        return center, float(radius), float(sse)

    def _create_node(self, sample_idx: np.ndarray, depth: int, parent_id: int = -1) -> BallNode:
        X_sub = self.X[sample_idx]
        center, radius, sse = self._calc_center_radius_sse(X_sub)
        node = BallNode(
            node_id=len(self.nodes),
            sample_idx=np.asarray(sample_idx, dtype=int),
            depth=int(depth),
            center=center,
            radius=max(radius, self.min_radius),
            parent_id=parent_id,
            size=len(sample_idx),
            sse=sse,
        )
        self.nodes.append(node)
        self.parent_map_[node.node_id] = parent_id
        self.node_by_id_[node.node_id] = node
        return node

    def _try_split(self, node: BallNode) -> bool:
        if node.depth >= self.max_depth:
            return False
        if node.size < self.min_samples_split:
            return False
        if node.radius <= self.min_radius:
            return False

        X_sub = self.X[node.sample_idx]
        if X_sub.shape[0] < 4:
            return False
        if np.allclose(np.std(X_sub, axis=0), 0):
            return False

        try:
            km = KMeans(n_clusters=2, n_init=5, random_state=self.random_state)
            labels = km.fit_predict(X_sub)
        except Exception:
            return False

        left_local = np.where(labels == 0)[0]
        right_local = np.where(labels == 1)[0]
        if len(left_local) == 0 or len(right_local) == 0:
            return False
        if min(len(left_local), len(right_local)) < max(2, int(0.05 * node.size)):
            return False

        left_idx = node.sample_idx[left_local]
        right_idx = node.sample_idx[right_local]

        _, _, sse_left = self._calc_center_radius_sse(self.X[left_idx])
        _, _, sse_right = self._calc_center_radius_sse(self.X[right_idx])

        gain = node.sse - (sse_left + sse_right)
        gain_ratio = gain / (node.sse + 1e-12)

        if gain_ratio < self.min_sse_gain:
            return False

        node.left = self._create_node(left_idx, depth=node.depth + 1, parent_id=node.node_id)
        node.right = self._create_node(right_idx, depth=node.depth + 1, parent_id=node.node_id)
        node.is_leaf = False
        return True

    def _build_recursive(self, node: BallNode) -> None:
        if not self._try_split(node):
            node.is_leaf = True
            return
        self._build_recursive(node.left)
        self._build_recursive(node.right)

    def _get_path_node_ids_from_leaf(self, leaf_id: int) -> List[int]:
        path = []
        cur = leaf_id
        while cur != -1:
            path.append(cur)
            cur = self.parent_map_[cur]
        path.reverse()
        return path

    # =====================================================
    # Fuzzy membership
    # =====================================================
    def _compute_leaf_membership_matrix(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        m = len(self.leaf_nodes)
        if m == 0:
            raise RuntimeError("No leaf nodes found. Please fit the model first.")

        centers = np.vstack([leaf.center for leaf in self.leaf_nodes])
        radii = np.array([max(leaf.radius, self.min_radius) for leaf in self.leaf_nodes], dtype=float)

        diff = X[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff ** 2, axis=2)

        sigma2 = (self.fuzzy_tau * radii) ** 2
        sigma2 = np.maximum(sigma2, 1e-12)
        membership = np.exp(-dist2 / (2.0 * sigma2[None, :]))

        if self.max_leaf_for_fuzzy is not None and m > self.max_leaf_for_fuzzy:
            k = max(1, min(int(self.max_leaf_for_fuzzy), m))
            idx_topk = np.argpartition(-membership, k - 1, axis=1)[:, :k]
            sparse_membership = np.zeros_like(membership)
            row_idx = np.arange(n)[:, None]
            sparse_membership[row_idx, idx_topk] = membership[row_idx, idx_topk]
            membership = sparse_membership

        row_sum = np.maximum(np.sum(membership, axis=1, keepdims=True), 1e-12)
        membership = membership / row_sum
        return membership

    def _build_leaf_sparse_map(self) -> None:
        leaf_sizes = np.array([leaf.size for leaf in self.leaf_nodes], dtype=float)
        leaf_radii = np.array([leaf.radius for leaf in self.leaf_nodes], dtype=float)

        min_size, max_size = np.min(leaf_sizes), np.max(leaf_sizes)
        min_rad, max_rad = np.min(leaf_radii), np.max(leaf_radii)

        sparse_map = {}
        for leaf in self.leaf_nodes:
            size_ratio = 0.0 if max_size - min_size < 1e-12 else (leaf.size - min_size) / (max_size - min_size)
            rad_ratio = 0.0 if max_rad - min_rad < 1e-12 else (leaf.radius - min_rad) / (max_rad - min_rad)
            sparse = self.sparse_alpha * (1.0 - size_ratio) + (1.0 - self.sparse_alpha) * rad_ratio
            sparse_map[leaf.node_id] = float(sparse)

        self.leaf_sparse_map_ = sparse_map

    # =====================================================
    # Score computation
    # =====================================================
    def _compute_path_level_deviation_matrix(self, X: np.ndarray, membership: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        m = len(self.leaf_nodes)

        leaf_paths = [self._get_path_node_ids_from_leaf(leaf.node_id) for leaf in self.leaf_nodes]
        max_len = max(len(p) for p in leaf_paths)

        level_dev_expected = np.zeros((n, max_len), dtype=float)
        level_weight_expected = np.zeros((n, max_len), dtype=float)

        for j in range(m):
            mu_j = membership[:, j]
            if np.all(mu_j < 1e-15):
                continue

            path_ids = leaf_paths[j]
            for level_idx in range(max_len):
                nid = path_ids[level_idx] if level_idx < len(path_ids) else path_ids[-1]
                node = self.node_by_id_[nid]

                d = np.linalg.norm(X - node.center.reshape(1, -1), axis=1)
                local_dev = d / (node.radius + 1e-12)
                depth_weight = 1.0 + self.depth_gamma * node.depth

                level_dev_expected[:, level_idx] += mu_j * local_dev
                level_weight_expected[:, level_idx] += mu_j * depth_weight

        level_weight_expected = np.maximum(level_weight_expected, 1e-12)
        return level_dev_expected / level_weight_expected

    def _compute_fuzzy_path_score(self, level_dev_expected: np.ndarray) -> np.ndarray:
        mean_part = np.mean(level_dev_expected, axis=1)
        max_part = np.max(level_dev_expected, axis=1)
        path_score = 0.65 * mean_part + 0.35 * max_part
        return minmax01(path_score)

    def _compute_path_variance_score(self, level_dev_expected: np.ndarray) -> np.ndarray:
        return minmax01(np.var(level_dev_expected, axis=1))

    def _compute_fuzzy_sparse_score(self, membership: np.ndarray) -> np.ndarray:
        leaf_sparse = np.array([self.leaf_sparse_map_[leaf.node_id] for leaf in self.leaf_nodes], dtype=float)
        sparse_score = membership @ leaf_sparse
        top_leaf = np.max(membership * leaf_sparse.reshape(1, -1), axis=1)
        return minmax01(sparse_score + top_leaf)

    def _compute_membership_entropy_score(self, membership: np.ndarray) -> np.ndarray:
        m = membership.shape[1]
        entropy = -np.sum(membership * np.log(membership + 1e-12), axis=1)
        if m > 1:
            entropy = entropy / np.log(m + 1e-12)
        else:
            entropy = np.zeros_like(entropy)
        return minmax01(entropy)

    def _compute_weak_supervised_guidance_score(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        has_anom = len(self.labeled_anom_idx_) > 0
        has_norm = len(self.labeled_norm_idx_) > 0

        if (not has_anom) and (not has_norm):
            return np.zeros(n, dtype=float)

        all_radii = np.array([leaf.radius for leaf in self.leaf_nodes], dtype=float)
        scale = float(np.median(all_radii)) if len(all_radii) > 0 else 1.0
        scale = max(scale, 1e-6)
        sigma2 = (self.ws_tau * scale) ** 2

        sim_to_anom = np.zeros(n, dtype=float)
        far_from_norm = np.zeros(n, dtype=float)

        if has_anom:
            Xa = self.X[self.labeled_anom_idx_]
            d2a = pairwise_sq_dists(X, Xa)
            sim_mat = np.exp(-d2a / (2.0 * sigma2))
            k = min(3, sim_mat.shape[1])
            topk = np.partition(sim_mat, -k, axis=1)[:, -k:]
            sim_to_anom = minmax01(np.mean(topk, axis=1))

        if has_norm:
            Xn = self.X[self.labeled_norm_idx_]
            d2n = pairwise_sq_dists(X, Xn)
            near_norm = np.exp(-d2n / (2.0 * sigma2))
            k = min(3, near_norm.shape[1])
            topk = np.partition(near_norm, -k, axis=1)[:, -k:]
            near_norm = np.mean(topk, axis=1)
            far_from_norm = 1.0 - minmax01(near_norm)

        if has_anom and has_norm:
            guide = self.alpha_ws_anom * sim_to_anom + (1.0 - self.alpha_ws_anom) * far_from_norm
        elif has_anom:
            guide = sim_to_anom
        else:
            guide = far_from_norm

        guide = self.eta_ws * power_sharpen(guide, gamma=1.25)
        return minmax01(guide)

    def _compute_gate_score(self, uncertainty_score: np.ndarray, path_var_score: np.ndarray) -> np.ndarray:
        gate_input = self.beta_entropy * uncertainty_score + self.beta_pathvar * path_var_score - self.tau_gate
        gate = robust_sigmoid(2.2 * gate_input)
        return minmax01(gate)

    def _compose_final_score(
        self,
        unsup_score: np.ndarray,
        guide_score: np.ndarray,
        gate_score: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u = minmax01(unsup_score)
        g = minmax01(guide_score)
        t = minmax01(gate_score)

        base_term = u
        mult_term = minmax01(u * (1.0 + self.eta_mul * t * g))
        inter_term = minmax01(
            self.eta_inter * (np.power(u, self.power_unsup)) * (np.power(g, self.power_guide))
        )
        tail_rank = minmax01(rank_normalize(mult_term + inter_term))

        final_raw = self.w_base * base_term + self.w_mult * mult_term + inter_term + self.eta_tail * tail_rank
        final_raw = minmax01(final_raw)
        final_score = minmax01(power_sharpen(final_raw, gamma=self.gamma_tail))
        return final_score, mult_term, inter_term, tail_rank

    # =====================================================
    # Public API
    # =====================================================
    def fit(
        self,
        X: np.ndarray,
        labeled_anom_idx: Optional[Sequence[int]] = None,
        labeled_norm_idx: Optional[Sequence[int]] = None,
    ) -> "WSGBT":
        """
        Fit the WSGBT model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.
        labeled_anom_idx : Optional[Sequence[int]]
            Indices of labeled anomalies.
        labeled_norm_idx : Optional[Sequence[int]]
            Indices of labeled normal samples.

        Returns
        -------
        self : WSGBT
        """
        self.X = np.asarray(X, dtype=float)
        n = self.X.shape[0]

        self.nodes = []
        self.leaf_nodes = []
        self.parent_map_ = {}
        self.node_by_id_ = {}

        self.labeled_anom_idx_ = np.asarray(
            labeled_anom_idx if labeled_anom_idx is not None else [],
            dtype=int
        ).reshape(-1)

        self.labeled_norm_idx_ = np.asarray(
            labeled_norm_idx if labeled_norm_idx is not None else [],
            dtype=int
        ).reshape(-1)

        self.root = self._create_node(np.arange(n), depth=0, parent_id=-1)
        self._build_recursive(self.root)
        self.leaf_nodes = [node for node in self.nodes if node.is_leaf]

        self._build_leaf_sparse_map()

        membership = self._compute_leaf_membership_matrix(self.X)
        level_dev_expected = self._compute_path_level_deviation_matrix(self.X, membership)

        path_score = self._compute_fuzzy_path_score(level_dev_expected)
        sparse_score = self._compute_fuzzy_sparse_score(membership)
        unsup_score = minmax01(self.lambda_path * path_score + (1.0 - self.lambda_path) * sparse_score)

        uncertainty_score = self._compute_membership_entropy_score(membership)
        path_var_score = self._compute_path_variance_score(level_dev_expected)
        guide_score = self._compute_weak_supervised_guidance_score(self.X)
        gate_score = self._compute_gate_score(uncertainty_score, path_var_score)

        final_score, mult_term, inter_term, tail_rank = self._compose_final_score(
            unsup_score=unsup_score,
            guide_score=guide_score,
            gate_score=gate_score,
        )

        self.path_score_ = path_score
        self.sparse_score_ = sparse_score
        self.unsup_score_ = unsup_score
        self.guide_score_ = guide_score
        self.uncertainty_score_ = uncertainty_score
        self.path_var_score_ = path_var_score
        self.gate_score_ = gate_score
        self.mult_score_ = mult_term
        self.inter_score_ = inter_term
        self.tail_rank_score_ = tail_rank
        self.decision_scores_ = final_score

        return self

    def decision_function(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters
        ----------
        X : Optional[np.ndarray]
            If None, return training scores.
            Otherwise, compute scores for new samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Higher values indicate stronger anomaly tendency.
        """
        if X is None:
            if self.decision_scores_ is None:
                raise RuntimeError("Model has not been fitted yet.")
            return self.decision_scores_

        X = np.asarray(X, dtype=float)

        membership = self._compute_leaf_membership_matrix(X)
        level_dev_expected = self._compute_path_level_deviation_matrix(X, membership)

        path_score = self._compute_fuzzy_path_score(level_dev_expected)
        sparse_score = self._compute_fuzzy_sparse_score(membership)
        unsup_score = minmax01(self.lambda_path * path_score + (1.0 - self.lambda_path) * sparse_score)

        uncertainty_score = self._compute_membership_entropy_score(membership)
        path_var_score = self._compute_path_variance_score(level_dev_expected)
        guide_score = self._compute_weak_supervised_guidance_score(X)
        gate_score = self._compute_gate_score(uncertainty_score, path_var_score)

        final_score, _, _, _ = self._compose_final_score(
            unsup_score=unsup_score,
            guide_score=guide_score,
            gate_score=gate_score,
        )
        return minmax01(final_score)

    def fit_predict_score(
        self,
        X: np.ndarray,
        labeled_anom_idx: Optional[Sequence[int]] = None,
        labeled_norm_idx: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Fit the model and return anomaly scores on X.
        """
        self.fit(
            X=X,
            labeled_anom_idx=labeled_anom_idx,
            labeled_norm_idx=labeled_norm_idx,
        )
        return self.decision_function()

    def get_tree_stats(self) -> Dict[str, int]:
        """Return simple tree statistics."""
        return {
            "num_nodes": len(self.nodes),
            "num_leaf_nodes": len(self.leaf_nodes),
            "max_depth_observed": max((node.depth for node in self.nodes), default=0),
        }


if __name__ == "__main__":
    # Minimal example
    rng = np.random.RandomState(42)

    n_normal = 180
    n_anom = 20
    n_features = 6

    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    X_anom = rng.normal(loc=4.0, scale=1.2, size=(n_anom, n_features))
    X = np.vstack([X_normal, X_anom])

    labeled_anom_idx = np.array([n_normal, n_normal + 1, n_normal + 2], dtype=int)
    labeled_norm_idx = np.array([0, 1, 2, 3, 4], dtype=int)

    model = WSGBT(
        max_depth=6,
        min_samples_split=20,
        min_sse_gain=1e-4,
        sparse_alpha=0.50,
        depth_gamma=0.30,
        lambda_path=0.68,
        fuzzy_tau=0.95,
        ws_tau=0.90,
        alpha_ws_anom=0.60,
        w_base=0.30,
        w_mult=0.70,
        random_state=42,
    )

    scores = model.fit_predict_score(
        X,
        labeled_anom_idx=labeled_anom_idx,
        labeled_norm_idx=labeled_norm_idx,
    )

    print("WSGBT finished.")
    print("Score shape:", scores.shape)
    print("Score range: [{:.6f}, {:.6f}]".format(scores.min(), scores.max()))
    print("Tree stats:", model.get_tree_stats())
