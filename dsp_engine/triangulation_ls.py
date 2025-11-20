# triangulation_ls.py

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from .bearing_collect import BearingSet
from .node_output import TriangulationResult


def _bearing_deg_to_direction(bearing_deg: float) -> np.ndarray:
    """
    Convert a global bearing angle (deg) to a 2D direction vector.

    Convention:
        - bearing_deg = 0°   → pointing to +Y (North)
        - bearing_deg = 90°  → pointing to +X (East)
        - bearing_deg = 180° → pointing to -Y (South)
        - bearing_deg = 270° → pointing to -X (West)

    This matches typical "map" / compass convention.
    """
    theta_rad = np.deg2rad(bearing_deg)
    dx = np.sin(theta_rad)   # East component
    dy = np.cos(theta_rad)   # North component
    return np.array([dx, dy], dtype=float)


def _line_normal_from_bearing(bearing_deg: float) -> np.ndarray:
    """
    Given a bearing direction, construct a unit normal vector n such that:

        n · (target_pos - node_pos) = 0

    for a target lying exactly on the bearing ray.

    If d = [dx, dy] is the direction, one valid normal is:
        n = [-dy, dx]   (90° rotation)
    """
    d = _bearing_deg_to_direction(bearing_deg)
    # Rotate by +90° to get a normal
    n = np.array([-d[1], d[0]], dtype=float)
    # Normalize for numerical stability
    norm = np.linalg.norm(n)
    if norm == 0.0:
        return n
    return n / norm


def triangulate_bearing_set(
    bearing_set: BearingSet,
    min_nodes: int = 2,
    min_confidence: float = 0.0
) -> Optional[TriangulationResult]:
    """
    Least-squares 2D triangulation from a set of global bearings.

    Parameters
    ----------
    bearing_set : BearingSet
        Time-aligned bearings from multiple nodes.
        Uses:
          - bearing_set.x, bearing_set.y          : node positions [m]
          - bearing_set.bearings_deg              : global bearings [deg]
          - bearing_set.confidences               : detection confidences (0..1)
          - bearing_set.t_center                  : fusion time [s]
          - bearing_set.node_ids                  : node identifiers
    min_nodes : int
        Minimum number of nodes required to attempt a solution.
    min_confidence : float
        Optionally drop nodes with confidence below this threshold.

    Returns
    -------
    TriangulationResult or None
        None if there are not enough valid bearings or if the system is ill-conditioned.
    """
    x_nodes = bearing_set.x
    y_nodes = bearing_set.y
    bearings_deg = bearing_set.bearings_deg
    confidences = bearing_set.confidences
    node_ids = bearing_set.node_ids

    n = len(node_ids)
    if n < min_nodes:
        return None

    # Filter by confidence if requested
    mask = np.ones(n, dtype=bool)
    if min_confidence > 0.0:
        mask &= (confidences >= min_confidence)

    if mask.sum() < min_nodes:
        return None

    x_nodes = x_nodes[mask]
    y_nodes = y_nodes[mask]
    bearings_deg = bearings_deg[mask]
    confidences = confidences[mask]
    node_ids_filtered: List[str] = [nid for nid, m in zip(node_ids, mask) if m]

    m = len(node_ids_filtered)
    if m < min_nodes:
        return None

    # Build normal-equation system A * p ≈ b, where p = [x, y]^T
    # For each node i:
    #     n_i · (p - p_i) = 0
    # →   n_i · p = n_i · p_i
    A = np.zeros((m, 2), dtype=float)
    b = np.zeros(m, dtype=float)

    for i in range(m):
        n_i = _line_normal_from_bearing(float(bearings_deg[i]))
        p_i = np.array([x_nodes[i], y_nodes[i]], dtype=float)

        A[i, :] = n_i
        b[i] = np.dot(n_i, p_i)

    # Confidence-based weights (optional but recommended)
    # Higher confidence → larger weight.
    # Add small epsilon to avoid all-zero W if confidences are zero.
    w = np.clip(confidences.astype(float), 0.0, 1.0)
    if np.all(w == 0.0):
        w[:] = 1.0

    # Normalize weights so max = 1.0 (for numerical stability)
    w /= (np.max(w) + 1e-9)

    # Build weighted least squares: (A^T W A) p = A^T W b
    W = np.diag(w)
    At = A.T

    try:
        AtWA = At @ W @ A
        AtWb = At @ W @ b

        # Solve for p_hat = [x, y]
        p_hat = np.linalg.solve(AtWA, AtWb)
    except np.linalg.LinAlgError:
        # Ill-conditioned system (e.g., nearly parallel bearings)
        return None

    x_hat, y_hat = float(p_hat[0]), float(p_hat[1])

    # Compute residuals: signed distance from each line
    residuals = A @ p_hat - b  # shape (m,)

    # Approximate covariance matrix from residuals and normal equations
    # sigma^2 ≈ (r^T r) / (m - 2)   (2 parameters: x, y)
    cov = None
    dof = m - 2
    if dof > 0:
        rss = float(residuals.T @ residuals)
        sigma2 = rss / dof if dof > 0 else 0.0
        try:
            AtWA_inv = np.linalg.inv(AtWA)
            cov = sigma2 * AtWA_inv
        except np.linalg.LinAlgError:
            cov = None

    return TriangulationResult(
        t_center=float(bearing_set.t_center),
        x=x_hat,
        y=y_hat,
        node_ids=node_ids_filtered,
        residuals=residuals,
        cov=cov
    )


def triangulate_all_sets(
    bearing_sets: List[BearingSet],
    min_nodes: int = 2,
    min_confidence: float = 0.0
) -> List[TriangulationResult]:
    """
    Convenience helper: run triangulation over a list of BearingSet objects.

    Parameters
    ----------
    bearing_sets : list of BearingSet
    min_nodes : int
        Minimum number of nodes required for each set.
    min_confidence : float
        Minimum confidence per node (optional).

    Returns
    -------
    List[TriangulationResult]
        Only successful triangulations (None filtered out).
    """
    results: List[TriangulationResult] = []

    for bset in bearing_sets:
        res = triangulate_bearing_set(
            bset,
            min_nodes=min_nodes,
            min_confidence=min_confidence
        )
        if res is not None:
            results.append(res)

    return results
