# node_output.py

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class NodePose:
    """
    Static pose of a sensor node in a local 2D coordinate system.

    x, y are typically in meters (e.g., ENU frame after lat/lon → local transform).
    heading_deg is the node's local "0°" reference (e.g., facing North).
    """
    node_id: str
    x: float
    y: float
    heading_deg: float = 0.0


@dataclass
class NodeDetection:
    """
    Single detection from one node at one global timestamp.

    Fields are intentionally redundant (x, y are node position)
    so downstream code does not need to look up NodePose again.
    """
    node_id: str
    t_global: float        # global time in seconds (monotonic / synced)
    x: float               # node position x [m]
    y: float               # node position y [m]

    # Bearings in degrees:
    #  - bearing_local_deg: relative to node's local heading (0° = node's forward)
    #  - bearing_global_deg: absolute bearing in global frame (0° = North, CW)
    bearing_local_deg: float
    bearing_global_deg: float

    confidence: float      # detection confidence (0..1)
    detected: bool         # True if "drone present" decision is positive
    snr_db: float = 0.0    # optional SNR estimate in dB


@dataclass
class TriangulationResult:
    """
    Result of triangulating one BearingSet.

    x, y          : Estimated target position [m] in global 2D frame.
    t_center      : Fusion time (same as BearingSet.t_center).
    node_ids      : Nodes that contributed to this solution.
    residuals     : Geometric residuals per node (distance to each bearing line) [m].
    cov           : 2x2 covariance matrix of the position estimate (optional).
    """
    t_center: float
    x: float
    y: float
    node_ids: List[str]
    residuals: np.ndarray
    cov: Optional[np.ndarray] = None
