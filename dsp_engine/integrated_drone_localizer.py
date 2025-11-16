#!/usr/bin/env python3
"""
Integrated Drone Localization System
=====================================

Complete pipeline from multi-node audio detection to drone position tracking:
1. Multi-node detection (per-frame)
2. Bearing collection (time synchronization)
3. Triangulation (position calculation)
4. Output: Drone trajectory over time

Usage:
    from integrated_drone_localizer import IntegratedDroneLocalizer, SensorNode

    nodes = [
        SensorNode(id=1, position=(0, 0), audio_file="node1.wav"),
        SensorNode(id=2, position=(10, 0), audio_file="node2.wav"),
        SensorNode(id=3, position=(5, 8.66), audio_file="node3.wav"),
    ]

    localizer = IntegratedDroneLocalizer(nodes)
    results = localizer.process_and_localize()
    localizer.plot_trajectory(results)
"""

import os
import sys
import json
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import multi-node detector
from multi_node_detector import MultiNodeDetector, SensorNode, DetectorConfig

# Import localization modules
from node_output import NodeDetection, NodePose
from bearing_collect import collect_bearings_from_all_nodes, BearingSet
from bearing_transform import attach_global_bearings
from triangulation_ls import triangulate_bearing_set, triangulate_all_sets, TriangulationResult

# Import tracking
from alpha_beta_tracker import AlphaBetaTracker2D, AlphaBetaConfig
try:
    from kalman_tracker_2d import KalmanTracker2D, KalmanConfig2D
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False


class IntegratedDroneLocalizer:
    """
    Complete drone localization system integrating:
    - Multi-node detection
    - Bearing collection
    - Triangulation
    - Tracking (optional)
    """

    def __init__(
        self,
        nodes: List[SensorNode],
        config: Optional[DetectorConfig] = None,
        fusion_dt: float = 0.1,
        min_nodes_for_triangulation: int = 2,
        use_tracking: bool = True,
        use_kalman: bool = False
    ):
        """
        Initialize integrated localizer.

        Parameters
        ----------
        nodes : list of SensorNode
            Sensor nodes in the network
        config : DetectorConfig, optional
            Detection configuration
        fusion_dt : float
            Time window for fusing detections from multiple nodes [seconds]
        min_nodes_for_triangulation : int
            Minimum number of nodes required for triangulation
        use_tracking : bool
            Enable position tracking/smoothing (default: True)
        use_kalman : bool
            If True: use Kalman filter, If False: use Alpha-Beta tracker (default: False)
        """
        self.nodes = nodes
        self.fusion_dt = fusion_dt
        self.min_nodes = min_nodes_for_triangulation
        self.use_tracking = use_tracking
        self.use_kalman = use_kalman

        # Create multi-node detector
        self.detector = MultiNodeDetector(nodes, config)

        # Initialize tracker
        self.tracker = None
        if self.use_tracking:
            if self.use_kalman:
                if not KALMAN_AVAILABLE:
                    print("WARNING: Kalman filter not available. Falling back to Alpha-Beta tracker.")
                    self.use_kalman = False
                else:
                    print("Using Kalman Filter for tracking")
                    self.tracker = KalmanTracker2D(KalmanConfig2D())

            if not self.use_kalman:
                print("Using Alpha-Beta Tracker for tracking")
                self.tracker = AlphaBetaTracker2D(AlphaBetaConfig())

    def process_and_localize(self) -> Dict:
        """
        Complete pipeline: detection → bearing collection → triangulation.

        Returns
        -------
        results : dict
            {
                'node_detections': List[List[NodeDetection]],  # Per node, per frame
                'bearing_sets': List[BearingSet],  # Time-synchronized groups
                'triangulation_results': List[TriangulationResult],  # Positions over time
                'summary': dict
            }
        """
        print("="*70)
        print("INTEGRATED DRONE LOCALIZATION PIPELINE")
        print("="*70)
        print()

        # ====================================================================
        # STEP 1: Multi-node detection (get per-frame detections)
        # ====================================================================
        print("[STEP 1/3] Multi-Node Detection (Frame-by-Frame)")
        print("-"*70)

        node_detections_all = self._extract_frame_detections()

        # Count total detections
        total_detections = sum(len(dets) for dets in node_detections_all)
        print(f"\nTotal detections across all nodes: {total_detections}")
        for i, dets in enumerate(node_detections_all):
            print(f"  Node {self.nodes[i].id}: {len(dets)} detections")
        print()

        # ====================================================================
        # STEP 2: Bearing collection (time synchronization)
        # ====================================================================
        print("[STEP 2/3] Bearing Collection (Time Synchronization)")
        print("-"*70)

        bearing_sets = collect_bearings_from_all_nodes(
            node_detections_all,
            fusion_dt=self.fusion_dt,
            min_nodes=self.min_nodes
        )

        print(f"Created {len(bearing_sets)} time-synchronized bearing sets")
        print(f"Fusion time window: {self.fusion_dt*1000:.0f} ms")
        if len(bearing_sets) > 0:
            nodes_per_set = [len(bset.node_ids) for bset in bearing_sets]
            print(f"Nodes per set: min={min(nodes_per_set)}, max={max(nodes_per_set)}, mean={np.mean(nodes_per_set):.1f}")
        print()

        # ====================================================================
        # STEP 3: Triangulation (position calculation)
        # ====================================================================
        print("[STEP 3/3] Triangulation (Position Calculation)")
        print("-"*70)

        triangulation_results = triangulate_all_sets(
            bearing_sets,
            min_nodes=self.min_nodes
        )

        print(f"Successful triangulations: {len(triangulation_results)}")
        if len(triangulation_results) > 0:
            residuals = [np.mean(np.abs(res.residuals)) for res in triangulation_results]
            print(f"Mean residual error: {np.mean(residuals):.3f} m")
            print(f"Max residual error: {np.max(residuals):.3f} m")
        print()

        # ====================================================================
        # STEP 4: Tracking (optional - smooth positions)
        # ====================================================================
        tracked_positions = None
        tracked_velocities = None

        if self.use_tracking and len(triangulation_results) > 0:
            print("[STEP 4/4] Tracking (Position Smoothing)")
            print("-"*70)

            tracker_name = "Kalman Filter" if self.use_kalman else "Alpha-Beta Tracker"
            print(f"Using {tracker_name} to smooth trajectory...")

            tracked_positions = []
            tracked_velocities = []

            for res in triangulation_results:
                pos, vel = self.tracker.update(res.x, res.y, res.t_center)
                tracked_positions.append(pos)
                tracked_velocities.append(vel)

            tracked_positions = np.array(tracked_positions)
            tracked_velocities = np.array(tracked_velocities)

            # Calculate tracking improvement
            raw_positions = np.array([[res.x, res.y] for res in triangulation_results])
            position_diff = np.linalg.norm(tracked_positions - raw_positions, axis=1)
            mean_diff = np.mean(position_diff)

            print(f"Tracked {len(tracked_positions)} positions")
            print(f"Mean position adjustment: {mean_diff:.3f} m")
            print(f"Mean velocity: ({np.mean(tracked_velocities[:, 0]):.2f}, {np.mean(tracked_velocities[:, 1]):.2f}) m/s")
            print()

        # ====================================================================
        # Summary
        # ====================================================================
        summary = {
            'total_nodes': len(self.nodes),
            'total_detections': total_detections,
            'bearing_sets': len(bearing_sets),
            'successful_triangulations': len(triangulation_results),
            'fusion_dt': self.fusion_dt,
            'min_nodes': self.min_nodes,
            'tracking_enabled': self.use_tracking,
            'tracker_type': 'Kalman' if self.use_kalman else 'Alpha-Beta' if self.use_tracking else None
        }

        if len(triangulation_results) > 0:
            positions = np.array([[res.x, res.y] for res in triangulation_results])
            summary['mean_position_raw'] = positions.mean(axis=0).tolist()
            summary['std_position_raw'] = positions.std(axis=0).tolist()
            summary['position_range_x'] = [positions[:, 0].min(), positions[:, 0].max()]
            summary['position_range_y'] = [positions[:, 1].min(), positions[:, 1].max()]

            if tracked_positions is not None:
                summary['mean_position_tracked'] = tracked_positions.mean(axis=0).tolist()
                summary['std_position_tracked'] = tracked_positions.std(axis=0).tolist()
                summary['mean_velocity'] = tracked_velocities.mean(axis=0).tolist()
                summary['std_velocity'] = tracked_velocities.std(axis=0).tolist()

        print("="*70)
        print("LOCALIZATION SUMMARY")
        print("="*70)
        print(f"Drone positions calculated: {len(triangulation_results)}")
        if len(triangulation_results) > 0:
            print(f"Mean position (raw): ({summary['mean_position_raw'][0]:.2f}, {summary['mean_position_raw'][1]:.2f}) m")
            print(f"Position std (raw): (±{summary['std_position_raw'][0]:.2f}, ±{summary['std_position_raw'][1]:.2f}) m")
            if tracked_positions is not None:
                print(f"Mean position (tracked): ({summary['mean_position_tracked'][0]:.2f}, {summary['mean_position_tracked'][1]:.2f}) m")
                print(f"Position std (tracked): (±{summary['std_position_tracked'][0]:.2f}, ±{summary['std_position_tracked'][1]:.2f}) m")
                print(f"Mean velocity: ({summary['mean_velocity'][0]:.2f}, {summary['mean_velocity'][1]:.2f}) m/s")
        print()

        return {
            'node_detections': node_detections_all,
            'bearing_sets': bearing_sets,
            'triangulation_results': triangulation_results,
            'tracked_positions': tracked_positions,
            'tracked_velocities': tracked_velocities,
            'summary': summary
        }

    def _extract_frame_detections(self) -> List[List[NodeDetection]]:
        """
        Process all nodes and extract frame-by-frame detections.

        Returns
        -------
        all_detections : list of lists
            Outer list: one per node
            Inner list: NodeDetection objects for each frame
        """
        from main_drone_detector import process_audio_file

        all_node_detections = []

        for node in self.nodes:
            if not node.is_active or not node.audio_file:
                all_node_detections.append([])
                continue

            print(f"Processing Node {node.id}...")

            # Update config with node-specific mic spacing
            self.detector.config.mic_spacing_m = node.mic_spacing

            # Process audio file
            results = process_audio_file(node.audio_file, self.detector.config)

            # Extract frame-by-frame detections
            frame_times = results['frame_times']
            confidences = results['confidences']
            detections = results['detections']
            snr_values = results['snr_values']
            doa_angles_rad = results['doa_angles']  # Can be None for mono

            node_detections = []

            for i in range(len(frame_times)):
                # Only include frames where detection occurred
                if detections[i]:
                    # Get bearing angle (if stereo)
                    if doa_angles_rad is not None:
                        bearing_deg = float(np.degrees(doa_angles_rad[i]))
                    else:
                        bearing_deg = 0.0  # Default if mono

                    # Create NodeDetection
                    det = NodeDetection(
                        node_id=str(node.id),
                        t_global=float(frame_times[i]),
                        x=float(node.position[0] if node.position else 0.0),
                        y=float(node.position[1] if node.position else 0.0),
                        bearing_local_deg=bearing_deg,  # Assuming node heading = 0°
                        bearing_global_deg=bearing_deg,  # Same if heading = 0°
                        confidence=float(confidences[i]),
                        detected=True,
                        snr_db=float(snr_values[i]),
                        gps_lat=node.gps_lat,
                        gps_lon=node.gps_lon,
                        gps_alt=node.gps_alt
                    )
                    node_detections.append(det)

            all_node_detections.append(node_detections)
            print(f"  → {len(node_detections)} detections")

        return all_node_detections

    def plot_trajectory(self, results: Dict, save_path: Optional[str] = None):
        """
        Visualize drone trajectory and node positions.

        Parameters
        ----------
        results : dict
            Results from process_and_localize()
        save_path : str, optional
            Path to save plot
        """
        triangulation_results = results['triangulation_results']

        if len(triangulation_results) == 0:
            print("No triangulation results to plot!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ================================================================
        # Plot 1: Trajectory + Nodes
        # ================================================================
        ax = axes[0]

        # Extract positions
        times = np.array([res.t_center for res in triangulation_results])
        positions_raw = np.array([[res.x, res.y] for res in triangulation_results])
        tracked_positions = results.get('tracked_positions')

        # Plot raw drone trajectory
        ax.scatter(positions_raw[:, 0], positions_raw[:, 1],
                  c='lightblue', s=80, alpha=0.6,
                  label='Raw Positions', zorder=2, edgecolors='blue')
        ax.plot(positions_raw[:, 0], positions_raw[:, 1],
               'b--', alpha=0.3, linewidth=1, label='Raw Trajectory')

        # Plot tracked trajectory (if available)
        if tracked_positions is not None:
            scatter = ax.scatter(tracked_positions[:, 0], tracked_positions[:, 1],
                                c=times, cmap='viridis', s=100,
                                label='Tracked Positions', zorder=3, edgecolors='black')
            ax.plot(tracked_positions[:, 0], tracked_positions[:, 1],
                   'g-', alpha=0.8, linewidth=2, label='Tracked Trajectory')
        else:
            scatter = ax.scatter(positions_raw[:, 0], positions_raw[:, 1],
                                c=times, cmap='viridis', s=100,
                                label='Drone Positions', zorder=3)

        # Plot nodes
        for node in self.nodes:
            if node.position:
                ax.plot(node.position[0], node.position[1],
                       'rs', markersize=15, markeredgecolor='black',
                       markeredgewidth=2, label=f'Node {node.id}' if node == self.nodes[0] else '')
                ax.text(node.position[0], node.position[1] + 0.5,
                       f'N{node.id}', fontsize=12, fontweight='bold',
                       ha='center')

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Drone Trajectory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='upper right')

        # Colorbar for time
        cbar = plt.colorbar(scatter, ax=ax, label='Time (s)')

        # ================================================================
        # Plot 2: Position vs Time
        # ================================================================
        ax = axes[1]

        # Plot raw positions
        ax.plot(times, positions_raw[:, 0], 'b--o', label='X (raw)', linewidth=1.5, alpha=0.6, markersize=4)
        ax.plot(times, positions_raw[:, 1], 'r--o', label='Y (raw)', linewidth=1.5, alpha=0.6, markersize=4)

        # Plot tracked positions if available
        if tracked_positions is not None:
            ax.plot(times, tracked_positions[:, 0], 'b-', label='X (tracked)', linewidth=2.5)
            ax.plot(times, tracked_positions[:, 1], 'r-', label='Y (tracked)', linewidth=2.5)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Position (m)', fontsize=12)
        title = 'Position vs Time'
        if tracked_positions is not None:
            tracker_name = 'Kalman' if self.use_kalman else 'Alpha-Beta'
            title += f' ({tracker_name} Tracking)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def export_results(self, results: Dict, output_file: str):
        """
        Export localization results to JSON.

        Parameters
        ----------
        results : dict
            Results from process_and_localize()
        output_file : str
            Output JSON file path
        """
        triangulation_results = results['triangulation_results']
        tracked_positions = results.get('tracked_positions')
        tracked_velocities = results.get('tracked_velocities')

        # Convert to serializable format
        export_data = {
            'summary': results['summary'],
            'positions_raw': [
                {
                    't': res.t_center,
                    'x': res.x,
                    'y': res.y,
                    'node_ids': res.node_ids,
                    'ranges': res.ranges.tolist(),
                    'residuals': res.residuals.tolist()
                }
                for res in triangulation_results
            ]
        }

        # Add tracked positions if available
        if tracked_positions is not None and tracked_velocities is not None:
            times = [res.t_center for res in triangulation_results]
            export_data['positions_tracked'] = [
                {
                    't': t,
                    'x': float(tracked_positions[i, 0]),
                    'y': float(tracked_positions[i, 1]),
                    'vx': float(tracked_velocities[i, 0]),
                    'vy': float(tracked_velocities[i, 1])
                }
                for i, t in enumerate(times)
            ]

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
    nodes = [
        SensorNode(id=1, position=(0.0, 0.0), mic_spacing=0.14, is_active=True),
        SensorNode(id=2, position=(10.0, 0.0), mic_spacing=0.14, is_active=True),
        SensorNode(id=3, position=(5.0, 8.66), mic_spacing=0.14, is_active=True)
    ]

    # Test file (same for all nodes in this example)
    test_file = "data/newdata/sensorlog_audio_20251114_201859.wav"

    if os.path.exists(test_file):
        # Assign audio files to nodes
        for node in nodes:
            node.audio_file = test_file

        # Create localizer with Alpha-Beta tracker (default)
        print("\n" + "="*70)
        print("TEST 1: Alpha-Beta Tracker (use_kalman=False)")
        print("="*70)
        localizer = IntegratedDroneLocalizer(
            nodes,
            fusion_dt=0.1,  # 100ms time windows
            min_nodes_for_triangulation=2,
            use_tracking=True,      # Enable tracking
            use_kalman=False        # Use Alpha-Beta (default)
        )

        # Process and localize
        results = localizer.process_and_localize()

        # Plot trajectory
        os.makedirs("results", exist_ok=True)
        localizer.plot_trajectory(results, save_path="results/drone_trajectory_alphabeta.png")

        # Export results
        localizer.export_results(results, "results/localization_results_alphabeta.json")

        # -----------------------------------------------------------------
        # Test with Kalman Filter
        # -----------------------------------------------------------------
        print("\n" + "="*70)
        print("TEST 2: Kalman Filter (use_kalman=True)")
        print("="*70)
        localizer_kalman = IntegratedDroneLocalizer(
            nodes,
            fusion_dt=0.1,
            min_nodes_for_triangulation=2,
            use_tracking=True,      # Enable tracking
            use_kalman=True         # Use Kalman filter
        )

        results_kalman = localizer_kalman.process_and_localize()
        localizer_kalman.plot_trajectory(results_kalman, save_path="results/drone_trajectory_kalman.png")
        localizer_kalman.export_results(results_kalman, "results/localization_results_kalman.json")

    else:
        print(f"Test file not found: {test_file}")
        print("Please provide audio files for each node.")
