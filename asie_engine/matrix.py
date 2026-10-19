"""Canonical matrix/edge conversion helpers for ASIE.

The contract is C[target, source]: an edge source -> target is stored at
matrix[target, source]. Edge tuples are (source_index, target_index) or
(source_index, target_index, weight). These helpers encode representation only;
they do not infer or validate causal relationships.
"""

import math
from numbers import Integral, Real

import numpy as np


def edge_to_matrix(edges, num_signals):
    """Convert indexed edges into a square C[target, source] matrix."""
    if isinstance(num_signals, bool) or not isinstance(num_signals, Integral) or num_signals <= 0:
        raise ValueError("num_signals must be a positive integer.")

    matrix = np.zeros((int(num_signals), int(num_signals)), dtype=float)
    seen = set()
    for edge in edges:
        try:
            parts = tuple(edge)
        except TypeError as exc:
            raise ValueError("Each edge must be a 2- or 3-item tuple.") from exc
        if len(parts) == 2:
            source, target = parts
            weight = 1.0
        elif len(parts) == 3:
            source, target, weight = parts
        else:
            raise ValueError("Each edge must be a 2- or 3-item tuple.")

        for name, index in (("source", source), ("target", target)):
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ValueError(f"{name} index must be an integer.")
            if index < 0 or index >= num_signals:
                raise ValueError(f"{name} index is outside the matrix bounds.")
        if source == target:
            raise ValueError("Self-edges are not supported by this DAG adapter.")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(float(weight)):
            raise ValueError("Edge weight must be a finite real number.")

        key = (int(source), int(target))
        if key in seen:
            raise ValueError(f"Duplicate edge {source}->{target}.")
        seen.add(key)
        matrix[int(target), int(source)] = float(weight)

    return matrix


def matrix_to_edges(matrix, threshold=0.0, include_self=False):
    """Convert a square C[target, source] matrix into (source, target, weight) tuples."""
    try:
        values = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix must be a finite numeric square array.") from exc
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be a square 2D array.")
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values.")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative real number.")
    if not isinstance(include_self, bool):
        raise ValueError("include_self must be a boolean.")

    edges = []
    for target in range(values.shape[0]):
        for source in range(values.shape[1]):
            if source == target and not include_self:
                continue
            weight = float(values[target, source])
            if abs(weight) > threshold:
                edges.append((source, target, weight))
    return edges
