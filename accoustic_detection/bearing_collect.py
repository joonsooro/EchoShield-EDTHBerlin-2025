# bearing_collect.py
"""
Collect bearings from multiple nodes and group them into
time-aligned sets for triangulation.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from node_output import NodeDetection


@dataclass
class BearingSet:
    """
    All bearings available in a small time window.

    t_center        : center time of this fusion window (global seconds)
    node_ids        : list of node ids contributing
    x, y            : arrays of node positions (same length as node_ids)
    bearings_deg    : global bearings theta_i [deg] for each node
    confidences     : confidence c_i for each node
    """
    t_center: float
    node_ids: List[str]
    x: np.ndarray
    y: np.ndarray
    bearings_deg: np.ndarray
    confidences: np.ndarray


def collect_bearings_from_all_nodes(
    all_node_detections: List[List[NodeDetection]],
    fusion_dt: float = 0.1,
    min_nodes: int = 2,
) -> List[BearingSet]:
    """
    Group node detections into time bins of width fusion_dt.

    Parameters
    ----------
    all_node_detections : list of lists
        Outer list over nodes, inner list = detections from that node.
    fusion_dt : float
        Fusion time step [s]. Detections whose t_global fall into the
        same bin are fused together.
    min_nodes : int
        Require at least this many nodes in a bin to keep it.

    Returns
    -------
    List[BearingSet]
    """
    # Flatten all detections
    flat: List[NodeDetection] = [
        d for det_list in all_node_detections for d in det_list
    ]

    if len(flat) == 0:
        return []

    # Sort by global time
    flat.sort(key=lambda d: d.t_global)

    # Use first detection as time origin for binning
    t0 = flat[0].t_global

    # Bin by integer index k = round((t - t0) / fusion_dt)
    bins: Dict[int, List[NodeDetection]] = {}
    for d in flat:
        k = int(round((d.t_global - t0) / fusion_dt))
        bins.setdefault(k, []).append(d)

    bearing_sets: List[BearingSet] = []

    for k, dets in bins.items():
        if len(dets) < min_nodes:
            continue

        t_center = t0 + k * fusion_dt

        node_ids = [d.node_id for d in dets]
        x = np.array([d.x for d in dets], dtype=float)
        y = np.array([d.y for d in dets], dtype=float)
        bearings = np.array([d.bearing_global_deg for d in dets], dtype=float)
        confidences = np.array([d.confidence for d in dets], dtype=float)

        bearing_sets.append(
            BearingSet(
                t_center=t_center,
                node_ids=node_ids,
                x=x,
                y=y,
                bearings_deg=bearings,
                confidences=confidences,
            )
        )

    # Optional: sort sets in time
    bearing_sets.sort(key=lambda s: s.t_center)
    return bearing_sets
