# bearing_transform.py
"""
Convert local node bearings (theta_i) to global bearings.

Global bearing = psi_i + theta_i  (wrapped to [0, 360) deg)
"""

from .node_output import NodeDetection, NodePose

from typing import Iterable, List
import numpy as np



def local_to_global_bearing(theta_local_deg: float, heading_deg: float) -> float:
    """
    Simple scalar helper.

    theta_local_deg : bearing in node's local frame
    heading_deg     : node heading in global frame

    return: bearing in global frame [deg]
    """
    return (theta_local_deg + heading_deg) % 360.0


def attach_global_bearings(
    detections: Iterable[NodeDetection],
    node_pose: NodePose,
) -> List[NodeDetection]:
    """
    Take NodeDetections that only have local bearings and
    fill in the global bearing field.

    If your NodeDetection already stores bearing_global_deg,
    you don't *have* to use this – it's just a clean helper.
    """

    out: List[NodeDetection] = []

    for d in detections:
        theta_global = local_to_global_bearing(d.bearing_local_deg,
                                               node_pose.heading_deg)
        d.bearing_global_deg = float(theta_global)
        out.append(d)

    return out
