#!/usr/bin/env python3
# node_output.py
"""
Data structures for node detection outputs.
Used by bearing collection and triangulation modules.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NodePose:
    """
    Position and orientation of a sensor node.

    Attributes
    ----------
    node_id : str
        Unique identifier for the node
    x : float
        X position in global frame [meters]
    y : float
        Y position in global frame [meters]
    heading_deg : float
        Node heading in global frame [degrees]
        0° = North, 90° = East, 180° = South, 270° = West
    gps_lat : float, optional
        GPS latitude [degrees]
    gps_lon : float, optional
        GPS longitude [degrees]
    gps_alt : float, optional
        GPS altitude [meters above sea level]
    """
    node_id: str
    x: float
    y: float
    heading_deg: float = 0.0
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None


@dataclass
class NodeDetection:
    """
    Single detection event from one node at a specific time.

    Attributes
    ----------
    node_id : str
        Identifier of the node that made the detection
    t_global : float
        Global timestamp [seconds] - must be synchronized across nodes
    x : float
        Node X position in global frame [meters]
    y : float
        Node Y position in global frame [meters]
    bearing_local_deg : float
        Bearing in node's local frame [degrees]
        (angle relative to node's heading)
    bearing_global_deg : float
        Bearing in global frame [degrees]
        (angle in world coordinates)
    confidence : float
        Detection confidence [0.0 - 1.0]
    detected : bool
        Whether drone was detected (True) or not (False)
    snr_db : float, optional
        Signal-to-noise ratio [dB]
    gps_lat : float, optional
        Node GPS latitude [degrees]
    gps_lon : float, optional
        Node GPS longitude [degrees]
    gps_alt : float, optional
        Node GPS altitude [meters]
    """
    node_id: str
    t_global: float
    x: float
    y: float
    bearing_local_deg: float
    bearing_global_deg: float
    confidence: float
    detected: bool
    snr_db: Optional[float] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
