#!/usr/bin/env python3
"""
Multi-Node Acoustic Drone Detection System
==========================================

System for distributed drone detection using multiple nodes with stereo microphones.
Each node independently detects and estimates the Direction of Arrival (DOA).

Architecture:
- Multiple sensor nodes, each with stereo microphones
- Each node runs detection independently
- Outputs: Node ID, Detection status, Angle, Confidence
- Ready for triangulation/multi-lateration for 3D positioning

Usage:
    from multi_node_detector import MultiNodeDetector, SensorNode

    # Define nodes
    nodes = [
        SensorNode(id=1, position=(0, 0), mic_spacing=0.14),
        SensorNode(id=2, position=(10, 0), mic_spacing=0.14),
        SensorNode(id=3, position=(5, 8.66), mic_spacing=0.14)
    ]

    # Create detector
    detector = MultiNodeDetector(nodes)

    # Process audio files from each node
    results = detector.process_all_nodes(audio_files)
"""

import os
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

from main_drone_detector import DetectorConfig, process_audio_file


# ============================================================================
# Sensor Node Definition
# ============================================================================

@dataclass
class SensorNode:
    """
    Represents a single sensor node with stereo microphones.

    Attributes
    ----------
    id : int
        Unique node identifier
    position : tuple of float
        (x, y) position in meters (for local triangulation)
        OR None if using GPS coordinates
    mic_spacing : float
        Microphone spacing in meters
    is_active : bool
        Whether this node is currently active
    audio_file : str, optional
        Path to audio file from this node
    gps_lat : float, optional
        GPS latitude (degrees)
    gps_lon : float, optional
        GPS longitude (degrees)
    gps_alt : float, optional
        GPS altitude (meters above sea level)
    gps_timestamp : float, optional
        GPS timestamp (Unix time or as recorded)
    """
    id: int
    position: Optional[Tuple[float, float]] = None  # (x, y) in meters, or None if using GPS
    mic_spacing: float = 0.14  # meters
    is_active: bool = True
    audio_file: Optional[str] = None
    gps_lat: Optional[float] = None  # GPS latitude
    gps_lon: Optional[float] = None  # GPS longitude
    gps_alt: Optional[float] = None  # GPS altitude (m)
    gps_timestamp: Optional[float] = None  # GPS time

    def __repr__(self):
        if self.gps_lat is not None and self.gps_lon is not None:
            return f"Node{self.id}@GPS({self.gps_lat:.6f},{self.gps_lon:.6f})"
        elif self.position is not None:
            return f"Node{self.id}@({self.position[0]:.1f},{self.position[1]:.1f}m)"
        else:
            return f"Node{self.id}"


@dataclass
class NodeDetectionResult:
    """
    Detection result from a single node.

    Attributes
    ----------
    node_id : int
        Node identifier
    is_active : bool
        Whether node was active
    detected : bool
        Whether drone was detected
    confidence : float
        Detection confidence [0-1]
    angle_deg : float or None
        Direction of Arrival in degrees (None if not stereo or failed)
    angle_std_deg : float or None
        Standard deviation of angle estimate
    snr_db : float
        Average signal-to-noise ratio
    timestamp : float
        Unix timestamp of detection (processing time if no GPS)
    position : tuple or None
        Node position (x, y) in local coordinates
    mic_spacing : float
        Microphone spacing used
    gps_lat : float or None
        GPS latitude
    gps_lon : float or None
        GPS longitude
    gps_alt : float or None
        GPS altitude
    gps_timestamp : float or None
        GPS timestamp from recording
    """
    node_id: int
    is_active: bool
    detected: bool
    confidence: float
    angle_deg: Optional[float]
    angle_std_deg: Optional[float]
    snr_db: float
    timestamp: float
    position: Optional[Tuple[float, float]]
    mic_spacing: float
    processing_time: float
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    gps_timestamp: Optional[float] = None

    def to_dict(self):
        """Convert to dictionary for JSON export."""
        return asdict(self)


# ============================================================================
# Multi-Node Detector
# ============================================================================

class MultiNodeDetector:
    """
    Multi-node acoustic drone detection system.

    Manages multiple sensor nodes and coordinates detection across the network.
    """

    def __init__(self, nodes: List[SensorNode], config: Optional[DetectorConfig] = None):
        """
        Initialize multi-node detector.

        Parameters
        ----------
        nodes : list of SensorNode
            List of sensor nodes in the network
        config : DetectorConfig, optional
            Detection configuration (uses tuned defaults if None)
        """
        self.nodes = nodes
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> DetectorConfig:
        """Get default tuned configuration for low-SNR drone detection."""
        return DetectorConfig(
            # Audio preprocessing
            target_fs=16000,
            max_duration=10.0,

            # Framing
            frame_length_ms=64.0,
            hop_length_ms=32.0,
            window_type='hann',

            # FFT
            nfft=1024,

            # Filtering - disabled for low SNR
            use_harmonic_filter=False,

            # Detection - tuned for low-SNR conditions
            detector_f0=100,
            detector_n_harmonics=5,
            detector_band_hz=(50, 2000),
            detector_harmonic_bw=60,

            # Evidence weights
            weight_snr=0.3,
            weight_harmonic=0.2,
            weight_temporal=0.5,  # Emphasize temporal consistency

            # Thresholds - relaxed for low SNR
            snr_range_db=(-5.0, 20.0),
            harmonic_min_snr_db=0.5,
            temporal_window=7,
            confidence_threshold=0.20,

            # Stereo/DOA - will be overridden per node
            mic_spacing_m=0.14,

            # Output
            verbose=False  # Disable for batch processing
        )

    def check_active_nodes(self) -> Tuple[int, List[SensorNode]]:
        """
        Check how many nodes are active.

        Returns
        -------
        count : int
            Number of active nodes
        active_nodes : list of SensorNode
            List of active nodes
        """
        active = [node for node in self.nodes if node.is_active]
        return len(active), active

    def process_node(self, node: SensorNode) -> NodeDetectionResult:
        """
        Process audio from a single node.

        Parameters
        ----------
        node : SensorNode
            Node to process

        Returns
        -------
        result : NodeDetectionResult
            Detection result from this node
        """
        if not node.is_active:
            return NodeDetectionResult(
                node_id=node.id,
                is_active=False,
                detected=False,
                confidence=0.0,
                angle_deg=None,
                angle_std_deg=None,
                snr_db=0.0,
                timestamp=node.gps_timestamp if node.gps_timestamp else time.time(),
                position=node.position,
                mic_spacing=node.mic_spacing,
                processing_time=0.0,
                gps_lat=node.gps_lat,
                gps_lon=node.gps_lon,
                gps_alt=node.gps_alt,
                gps_timestamp=node.gps_timestamp
            )

        if not node.audio_file or not os.path.exists(node.audio_file):
            print(f"Warning: Node {node.id} - Audio file not found: {node.audio_file}")
            return NodeDetectionResult(
                node_id=node.id,
                is_active=True,
                detected=False,
                confidence=0.0,
                angle_deg=None,
                angle_std_deg=None,
                snr_db=0.0,
                timestamp=node.gps_timestamp if node.gps_timestamp else time.time(),
                position=node.position,
                mic_spacing=node.mic_spacing,
                processing_time=0.0,
                gps_lat=node.gps_lat,
                gps_lon=node.gps_lon,
                gps_alt=node.gps_alt,
                gps_timestamp=node.gps_timestamp
            )

        # Update config with node-specific mic spacing
        self.config.mic_spacing_m = node.mic_spacing

        # Process audio
        start_time = time.time()
        results = process_audio_file(node.audio_file, self.config)
        processing_time = time.time() - start_time

        # Extract results
        # Use GPS timestamp if available, otherwise current time
        result_timestamp = node.gps_timestamp if node.gps_timestamp else time.time()

        return NodeDetectionResult(
            node_id=node.id,
            is_active=True,
            detected=results['overall_detected'],
            confidence=results['mean_confidence'],
            angle_deg=results['mean_doa_deg'],
            angle_std_deg=results['std_doa_deg'],
            snr_db=results['mean_snr_db'],
            timestamp=result_timestamp,
            position=node.position,
            mic_spacing=node.mic_spacing,
            processing_time=processing_time,
            gps_lat=node.gps_lat,
            gps_lon=node.gps_lon,
            gps_alt=node.gps_alt,
            gps_timestamp=node.gps_timestamp
        )

    def process_all_nodes(self, audio_files: Optional[Dict[int, str]] = None) -> List[NodeDetectionResult]:
        """
        Process audio from all nodes.

        Parameters
        ----------
        audio_files : dict, optional
            Mapping of node_id -> audio_file_path
            If None, uses audio_file attribute from each node

        Returns
        -------
        results : list of NodeDetectionResult
            Detection results from all nodes
        """
        # Update audio files if provided
        if audio_files:
            for node in self.nodes:
                if node.id in audio_files:
                    node.audio_file = audio_files[node.id]

        # Check active nodes
        n_active, active_nodes = self.check_active_nodes()

        print(f"="*70)
        print(f"MULTI-NODE DETECTION SYSTEM")
        print(f"="*70)
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Active nodes: {n_active}")
        print(f"Inactive nodes: {len(self.nodes) - n_active}")
        print()

        # Process each node
        results = []
        for i, node in enumerate(self.nodes):
            print(f"[{i+1}/{len(self.nodes)}] Processing {node}...")
            result = self.process_node(node)
            results.append(result)

            if result.is_active:
                status = "DETECTED" if result.detected else "NO DETECTION"
                angle_str = f"{result.angle_deg:.3f}°" if result.angle_deg is not None else "N/A (mono)"
                print(f"  Status: {status}")
                print(f"  --- Required Outputs for Node {result.node_id} ---")

                # Show GPS timestamp if available, otherwise processing timestamp
                if result.gps_timestamp:
                    print(f"  Time (t):           {result.gps_timestamp:.3f} s (GPS)")
                else:
                    print(f"  Time (t):           {result.timestamp:.3f} s")

                print(f"  Bearing (θ{result.node_id}):       {angle_str}")
                print(f"  Confidence (c{result.node_id}):    {result.confidence:.3f}")

                # Show GPS coordinates if available, otherwise local position
                if result.gps_lat is not None and result.gps_lon is not None:
                    print(f"  GPS Location:       ({result.gps_lat:.6f}°, {result.gps_lon:.6f}°)")
                    if result.gps_alt is not None:
                        print(f"  GPS Altitude:       {result.gps_alt:.2f} m")
                elif result.position is not None:
                    print(f"  Node Pose (x{result.node_id}, y{result.node_id}): ({result.position[0]:.2f}, {result.position[1]:.2f}) m")

                print(f"  --- Additional Info ---")
                print(f"  SNR: {result.snr_db:.1f} dB")
                print(f"  Angle precision: ±{result.angle_std_deg:.3f}°" if result.angle_std_deg else "")
                print(f"  Processing time: {result.processing_time:.2f}s")
            else:
                print(f"  Status: INACTIVE")
            print()

        return results

    def get_detection_summary(self, results: List[NodeDetectionResult]) -> Dict:
        """
        Generate summary statistics from multi-node results.

        Parameters
        ----------
        results : list of NodeDetectionResult
            Results from all nodes

        Returns
        -------
        summary : dict
            Summary statistics
        """
        active_results = [r for r in results if r.is_active]
        detected_results = [r for r in active_results if r.detected]

        # Angles from detecting nodes (stereo only)
        angles = [r.angle_deg for r in detected_results if r.angle_deg is not None]

        summary = {
            'total_nodes': len(self.nodes),
            'active_nodes': len(active_results),
            'detecting_nodes': len(detected_results),
            'detection_rate': len(detected_results) / len(active_results) if active_results else 0.0,
            'mean_confidence': np.mean([r.confidence for r in detected_results]) if detected_results else 0.0,
            'mean_snr_db': np.mean([r.snr_db for r in detected_results]) if detected_results else 0.0,
            'angles_available': len(angles) > 0,
            'mean_angle_deg': np.mean(angles) if angles else None,
            'angles_deg': angles if angles else None,
            'node_positions': [r.position for r in detected_results],
            'timestamp': time.time()
        }

        return summary

    def get_required_outputs(self, results: List[NodeDetectionResult]) -> List[Dict]:
        """
        Extract required outputs for each node: time, bearing, confidence, pose.

        Parameters
        ----------
        results : list of NodeDetectionResult
            Results from all nodes

        Returns
        -------
        outputs : list of dict
            List of dictionaries with required fields for each node:
            - node_id (i)
            - time (t) - GPS timestamp if available
            - bearing (θi) in degrees
            - confidence (ci) in [0-1]
            - GPS location (lat, lon, alt) OR local pose (xi, yi)
        """
        outputs = []
        for result in results:
            if result.is_active:
                output = {
                    'node_id': result.node_id,
                    't': result.gps_timestamp if result.gps_timestamp else result.timestamp,  # GPS time preferred
                    'theta_i': result.angle_deg,  # Bearing θi (degrees)
                    'c_i': result.confidence,  # Confidence ci
                    'detected': result.detected  # Boolean detection status
                }

                # Add GPS coordinates if available
                if result.gps_lat is not None and result.gps_lon is not None:
                    output['gps_lat'] = result.gps_lat
                    output['gps_lon'] = result.gps_lon
                    if result.gps_alt is not None:
                        output['gps_alt'] = result.gps_alt
                # Otherwise add local position
                elif result.position is not None:
                    output['x_i'] = result.position[0]
                    output['y_i'] = result.position[1]

                outputs.append(output)
        return outputs

    def export_results(self, results: List[NodeDetectionResult], output_file: str):
        """
        Export results to JSON file.

        Parameters
        ----------
        results : list of NodeDetectionResult
            Detection results
        output_file : str
            Output JSON file path
        """
        summary = self.get_detection_summary(results)

        # Convert numpy types to Python types for JSON serialization
        def convert_to_python_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            return obj

        export_data = {
            'required_outputs': convert_to_python_types(self.get_required_outputs(results)),
            'summary': convert_to_python_types(summary),
            'node_results': [convert_to_python_types(r.to_dict()) for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Results exported to: {output_file}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example: 3-node triangular array for drone localization
    """

    # Define 3 nodes in triangular configuration
    # Node positions in meters (x, y)
    nodes = [
        SensorNode(id=1, position=(0.0, 0.0), mic_spacing=0.14, is_active=True),
        SensorNode(id=2, position=(10.0, 0.0), mic_spacing=0.14, is_active=True),
        SensorNode(id=3, position=(5.0, 8.66), mic_spacing=0.14, is_active=True)
    ]

    print("Multi-Node Drone Detection System - Example")
    print("=" * 70)
    print("\nNode Configuration:")
    for node in nodes:
        print(f"  {node}")
    print()

    # Create detector
    detector = MultiNodeDetector(nodes)

    # For this example, use the same test file for all nodes
    # In real deployment, each node would have its own audio file
    test_file = "data/newdata/sensorlog_audio_20251114_201859.wav"

    if os.path.exists(test_file):
        audio_files = {
            1: test_file,
            2: test_file,
            3: test_file
        }

        # Process all nodes
        results = detector.process_all_nodes(audio_files)

        # Print summary
        print("="*70)
        print("DETECTION SUMMARY")
        print("="*70)
        summary = detector.get_detection_summary(results)
        print(f"Active nodes: {summary['active_nodes']}/{summary['total_nodes']}")
        print(f"Detecting nodes: {summary['detecting_nodes']}")
        print(f"Detection rate: {summary['detection_rate']*100:.1f}%")
        if summary['detecting_nodes'] > 0:
            print(f"Mean confidence: {summary['mean_confidence']:.3f}")
            print(f"Mean SNR: {summary['mean_snr_db']:.1f} dB")
        if summary['angles_available']:
            print(f"Angles from nodes: {summary['angles_deg']}")
            print(f"Mean angle: {summary['mean_angle_deg']:.1f}°")
        print()

        # Print required outputs for integration
        print("="*70)
        print("REQUIRED OUTPUTS FOR TRIANGULATION/POSITIONING")
        print("="*70)
        required = detector.get_required_outputs(results)
        for node_out in required:
            print(f"Node {node_out['node_id']}:")
            print(f"  t        = {node_out['t']:.3f} s")
            print(f"  θ{node_out['node_id']}       = {node_out['theta_i']:.3f}°")
            print(f"  c{node_out['node_id']}       = {node_out['c_i']:.3f}")
            print(f"  (x{node_out['node_id']}, y{node_out['node_id']}) = ({node_out['x_i']:.2f}, {node_out['y_i']:.2f}) m")
            print(f"  Detected = {node_out['detected']}")
            print()

        # Export results
        detector.export_results(results, "results/multi_node_results.json")

    else:
        print(f"Test file not found: {test_file}")
        print("Please provide audio files for each node.")
