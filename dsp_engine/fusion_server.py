#!/usr/bin/env python3
"""
Real-Time Fusion and Tracking Server
=====================================

Central application that continuously ingests NodeDetection events from
multiple sensor nodes, performs time-synchronized triangulation, and
maintains a stable, real-time track of the target's position and velocity.

Architecture:

    [Node 1] ──┐
    [Node 2] ──┼─→ [Ingestion Queue] → [Fusion Tick Loop] → [Tracker] → [Output]
    [Node 3] ──┘        ↓                      ↓                 ↓
                   Buffer by Node      Bearing Collection   Position &
                                      + Triangulation        Velocity

Key Features:
- **Real-time Processing**: Fixed-rate fusion tick (e.g., 10 Hz)
- **Time Synchronization**: Groups detections into time bins for triangulation
- **Stateful Tracking**: Kalman or Alpha-Beta filter for smooth trajectories
- **Bounded Memory**: Sliding window removes old detections
- **Efficient**: Only processes new data since last tick

Usage:
    # Initialize server
    server = FusionServer(
        node_ids=['1', '2', '3'],
        fusion_tick_rate=0.1,  # 10 Hz
        use_kalman=True
    )

    # Start server (blocking)
    server.run()

    # In separate threads/processes, nodes send detections:
    server.add_detection(detection_event)
"""

import json
import queue
import socket
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import detection and localization modules
from node_output import NodeDetection, NodePose
from bearing_collect import collect_bearings_from_all_nodes, BearingSet
from triangulation_ls import triangulate_bearing_set, TriangulationResult

# Import tracking modules
try:
    from kalman_tracker_2d import KalmanTracker2D
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    warnings.warn("KalmanTracker2D not available")

try:
    from alpha_beta_tracker import AlphaBetaTracker, AlphaBetaConfig
    ALPHA_BETA_AVAILABLE = True
except ImportError:
    ALPHA_BETA_AVAILABLE = False
    warnings.warn("AlphaBetaTracker not available")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FusionConfig:
    """
    Configuration for the fusion server.

    Attributes
    ----------
    fusion_tick_rate : float
        How often the fusion loop runs [seconds]
        Example: 0.1 = 10 Hz, 0.05 = 20 Hz
    detection_window_seconds : float
        How long to keep old detections in buffer [seconds]
        Older detections are pruned to prevent memory growth
    bearing_fusion_dt : float
        Time window for grouping detections into bearing sets [seconds]
        Smaller = finer time resolution, but requires more nodes
    min_nodes_for_triangulation : int
        Minimum nodes required to perform triangulation
        2 = minimum for 2D triangulation, 3+ = better accuracy
    use_kalman : bool
        If True, use Kalman filter. If False, use Alpha-Beta tracker
    verbose : bool
        If True, print detailed status information
    """
    fusion_tick_rate: float = 0.1  # 10 Hz
    detection_window_seconds: float = 5.0  # Keep 5 seconds of history
    bearing_fusion_dt: float = 0.1  # 100ms time bins
    min_nodes_for_triangulation: int = 2
    use_kalman: bool = True
    verbose: bool = True
    udp_host: str = "0.0.0.0"
    udp_port: int = 5005


# ============================================================================
# Fusion Server
# ============================================================================

class FusionServer:
    """
    Real-time fusion and tracking server for multi-node drone detection.

    This server continuously:
    1. Ingests NodeDetection events from multiple nodes
    2. Groups detections by time (bearing sets)
    3. Triangulates drone position from bearing angles
    4. Updates tracker for smooth position/velocity estimates
    5. Outputs real-time track to console/logging

    The server uses a fixed-rate tick loop for predictable, real-time behavior.
    """

    def __init__(
        self,
        node_ids: List[str],
        node_poses: Optional[Dict[str, NodePose]] = None,
        config: Optional[FusionConfig] = None
    ):
        """
        Initialize the fusion server.

        Parameters
        ----------
        node_ids : List[str]
            List of node IDs that will send detections
            Example: ['1', '2', '3']
        node_poses : Dict[str, NodePose], optional
            Dictionary mapping node_id to NodePose objects
            If provided, used for validation and display
        config : FusionConfig, optional
            Server configuration. If None, uses defaults.
        """
        # ====================================================================
        # Store configuration
        # ====================================================================
        self.config = config if config is not None else FusionConfig()
        self.node_ids = node_ids
        self.node_poses = node_poses if node_poses is not None else {}

        # ====================================================================
        # Step 1A: Initialize persistent state
        # ====================================================================

        # Detection buffer: stores NodeDetection objects by node
        # Structure: {'node_1': [det1, det2, ...], 'node_2': [...], ...}
        self.detections_by_node: Dict[str, List[NodeDetection]] = {
            node_id: [] for node_id in node_ids
        }

        # Ingestion queue: thread-safe queue for receiving detections
        # Nodes send detections here, ingestion thread pulls from here
        self.detection_queue = queue.Queue()

        # Network socket for node telemetry
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind((self.config.udp_host, self.config.udp_port))
        self.udp_socket.settimeout(0.5)

        # Time-keeping: tracks last processed timestamp to avoid reprocessing
        self.last_fused_time = 0.0

        # Tracker: initialize once and reuse (maintains state)
        if self.config.use_kalman:
            if not KALMAN_AVAILABLE:
                raise RuntimeError("Kalman tracker requested but not available")
            self.tracker = KalmanTracker2D()
            self.tracker_name = "Kalman"
        else:
            if not ALPHA_BETA_AVAILABLE:
                raise RuntimeError("Alpha-Beta tracker requested but not available")
            # Use default alpha-beta config (can be customized)
            alpha_beta_config = AlphaBetaConfig(alpha=0.85, beta=0.005)
            self.tracker = AlphaBetaTracker(alpha_beta_config)
            self.tracker_name = "Alpha-Beta"

        # Control flags
        self.running = False
        self.ingestion_thread = None
        self.fusion_thread = None
        self.network_thread = None

        # Statistics
        self.stats = {
            'total_detections_received': 0,
            'total_bearing_sets_created': 0,
            'total_triangulations': 0,
            'total_tracker_updates': 0,
            'fusion_ticks': 0
        }

        # ====================================================================
        # Print initialization info
        # ====================================================================
        if self.config.verbose:
            print("="*70)
            print("Real-Time Fusion and Tracking Server")
            print("="*70)
            print(f"\nConfiguration:")
            print(f"  Nodes: {len(self.node_ids)} ({', '.join(self.node_ids)})")
            print(f"  Fusion tick rate: {self.config.fusion_tick_rate}s ({1/self.config.fusion_tick_rate:.1f} Hz)")
            print(f"  Bearing fusion window: {self.config.bearing_fusion_dt}s")
            print(f"  Detection buffer window: {self.config.detection_window_seconds}s")
            print(f"  Min nodes for triangulation: {self.config.min_nodes_for_triangulation}")
            print(f"  Tracker: {self.tracker_name}")
            print(f"  UDP listener: {self.config.udp_host}:{self.config.udp_port}")
            print(f"\nServer initialized. Ready to receive detections.")
            print("="*70)

    # ========================================================================
    # Step 2: Ingestion Logic
    # ========================================================================

    def add_detection(self, detection: NodeDetection):
        """
        Add a detection to the ingestion queue.

        This is the main entry point for nodes to send detections to the server.
        Thread-safe - can be called from multiple threads/processes.

        Parameters
        ----------
        detection : NodeDetection
            Detection event from a node. Must have:
            - node_id: Which node sent this
            - t_global: Global synchronized timestamp
            - bearing_global_deg: Bearing angle in global frame
            - x, y: Node position
            - confidence: Detection confidence
            - detected: True if drone detected

        Example
        -------
        # Node sends detection
        detection = NodeDetection(
            node_id='1',
            t_global=1.234,
            x=0.0, y=0.0,
            bearing_local_deg=45.0,
            bearing_global_deg=45.0,
            confidence=0.85,
            detected=True,
            snr_db=15.0
        )
        server.add_detection(detection)
        """
        self.detection_queue.put(detection)

    def _ingestion_worker(self):
        """
        Ingestion thread worker.

        Continuously pulls detections from the queue and adds them to the
        appropriate node's buffer. Runs in a separate thread.

        This is Step 2A: The "Fake Network" Queue ingestion mechanism.
        """
        if self.config.verbose:
            print("\n[Ingestion Thread] Started")

        while self.running:
            try:
                # Block for up to 0.1 seconds waiting for detection
                # This allows thread to exit gracefully when running=False
                detection = self.detection_queue.get(timeout=0.1)

                # Extract node ID
                node_id = detection.node_id

                # Validate node ID
                if node_id not in self.detections_by_node:
                    warnings.warn(f"Received detection from unknown node: {node_id}")
                    continue

                # Add to buffer
                self.detections_by_node[node_id].append(detection)

                # Update statistics
                self.stats['total_detections_received'] += 1

                if self.config.verbose:
                    print(f"[Ingestion] Received detection from Node {node_id} "
                          f"at t={detection.t_global:.3f}s "
                          f"(confidence={detection.confidence:.2f})")

            except queue.Empty:
                # No detection available, continue waiting
                continue
            except Exception as e:
                print(f"[Ingestion] Error: {e}")
                continue

        if self.config.verbose:
            print("[Ingestion Thread] Stopped")

    # ========================================================================
    # Step 2B: Network Listener
    # ========================================================================

    def _network_listener(self):
        """
        Receive UDP packets from remote nodes, parse JSON, and enqueue detections.
        """
        if self.config.verbose:
            print(f"\n[Network Thread] Listening on "
                  f"{self.config.udp_host}:{self.config.udp_port}")

        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                continue

            try:
                message = json.loads(data.decode('utf-8'))
            except json.JSONDecodeError as e:
                warnings.warn(f"Malformed JSON from {addr}: {e}")
                continue

            detection = self._dict_to_detection(message)
            if detection is None:
                continue

            self.add_detection(detection)

        if self.config.verbose:
            print("[Network Thread] Stopped")

    def _dict_to_detection(self, payload: Dict) -> Optional[NodeDetection]:
        """
        Validate incoming payload and convert to NodeDetection.
        """
        required = ['node_id', 't_global', 'x_node', 'y_node',
                    'bearing_deg', 'confidence']
        missing = [key for key in required if key not in payload]
        if missing:
            warnings.warn(f"Detection missing fields: {missing}")
            return None

        try:
            node_id = str(payload['node_id'])
            t_global = float(payload['t_global'])
            x_node = float(payload['x_node'])
            y_node = float(payload['y_node'])
            bearing_deg = float(payload['bearing_deg'])
            confidence = float(payload['confidence'])
            snr_db = float(payload.get('snr_db', 0.0))
        except (TypeError, ValueError) as exc:
            warnings.warn(f"Invalid detection payload: {exc}")
            return None

        detection = NodeDetection(
            node_id=node_id,
            t_global=t_global,
            x=x_node,
            y=y_node,
            bearing_local_deg=bearing_deg,
            bearing_global_deg=bearing_deg,
            confidence=confidence,
            detected=True,
            snr_db=snr_db
        )

        return detection

    # ========================================================================
    # Step 3: Core Fusion Tick Loop
    # ========================================================================

    def _fusion_worker(self):
        """
        Fusion thread worker - the main processing loop.

        This implements Step 3: The core "Fusion Tick" Loop.
        Runs at a fixed rate (config.fusion_tick_rate) and performs:
        1. Data preparation (flatten detection buffer)
        2. Bearing collection (time synchronization)
        3. Filter new work (only process new bearing sets)
        4. Triangulation (calculate drone position)
        5. Tracker update (smooth position/velocity)
        6. State management (prune old detections)
        """
        if self.config.verbose:
            print("\n[Fusion Thread] Started")

        while self.running:
            # ================================================================
            # Step 3A: Timer
            # ================================================================
            time.sleep(self.config.fusion_tick_rate)

            self.stats['fusion_ticks'] += 1
            tick_start_time = time.time()

            # ================================================================
            # Step 3B: Data Preparation
            # ================================================================
            # Create flat list of all node detections for bearing collection
            all_node_detections = list(self.detections_by_node.values())

            # Check if we have any detections at all
            total_detections = sum(len(dets) for dets in all_node_detections)
            if total_detections == 0:
                # No detections yet, skip this tick
                continue

            # ================================================================
            # Step 3C: Call Bearing Collection
            # ================================================================
            # Group detections into time-synchronized bearing sets
            try:
                bearing_sets = collect_bearings_from_all_nodes(
                    all_node_detections,
                    fusion_dt=self.config.bearing_fusion_dt,
                    min_nodes=self.config.min_nodes_for_triangulation
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"[Fusion] Bearing collection failed: {e}")
                continue

            if len(bearing_sets) == 0:
                # No valid bearing sets formed yet
                continue

            self.stats['total_bearing_sets_created'] += len(bearing_sets)

            # ================================================================
            # Step 3D: Filter for New Work (Critical Efficiency Step)
            # ================================================================
            # Only process bearing sets that are newer than last_fused_time
            new_sets = [
                bset for bset in bearing_sets
                if bset.t_center > self.last_fused_time
            ]

            if len(new_sets) == 0:
                # No new data to process
                continue

            if self.config.verbose:
                print(f"\n[Fusion Tick {self.stats['fusion_ticks']}] "
                      f"Processing {len(new_sets)} new bearing set(s)")

            # ================================================================
            # Step 3E: Triangulation
            # ================================================================
            # Calculate drone position for each new bearing set
            triangulation_results = []

            for bset in new_sets:
                try:
                    result = triangulate_bearing_set(
                        bset,
                        min_nodes=self.config.min_nodes_for_triangulation
                    )
                except Exception as e:
                    if self.config.verbose:
                        print(f"[Fusion] Triangulation failed for t={bset.t_center:.3f}s: {e}")
                    continue

                if result is None:
                    continue

                triangulation_results.append(result)
                self.stats['total_triangulations'] += 1

            # ================================================================
            # Step 3F: Update the Tracker
            # ================================================================
            # For each triangulation result, update tracker and print output
            for result in triangulation_results:
                try:
                    # Update tracker with new measurement
                    tracked_position, tracked_velocity = self.tracker.update(
                        result.x,
                        result.y,
                        result.t_center
                    )

                    self.stats['total_tracker_updates'] += 1

                    # Print real-time track information
                    self._print_track_update(result, tracked_position, tracked_velocity)

                except Exception as e:
                    if self.config.verbose:
                        print(f"[Fusion] Tracker update failed: {e}")
                    continue

            # ================================================================
            # Step 3G: Update the System Time
            # ================================================================
            # Update last_fused_time to prevent reprocessing
            if new_sets:
                self.last_fused_time = new_sets[-1].t_center

            # ================================================================
            # Step 4: State Management - Sliding Window
            # ================================================================
            self._prune_old_detections()

            # ================================================================
            # Tick complete
            # ================================================================
            tick_duration = time.time() - tick_start_time
            if self.config.verbose and tick_duration > self.config.fusion_tick_rate:
                print(f"[Fusion] WARNING: Tick took {tick_duration:.3f}s "
                      f"(longer than tick rate {self.config.fusion_tick_rate}s)")

        if self.config.verbose:
            print("[Fusion Thread] Stopped")

    # ========================================================================
    # Step 4A: Sliding Window for State Management
    # ========================================================================

    def _prune_old_detections(self):
        """
        Remove old detections from buffer to prevent unbounded memory growth.

        Implements Step 4A: The Sliding Window.

        Keeps only detections within the configured time window
        (config.detection_window_seconds). Older detections are discarded.
        """
        current_time = time.time()

        for node_id in self.detections_by_node:
            old_list = self.detections_by_node[node_id]

            # Keep only recent detections
            new_list = [
                d for d in old_list
                if d.t_global > current_time - self.config.detection_window_seconds
            ]

            # Update buffer
            self.detections_by_node[node_id] = new_list

            # Log if we pruned any
            pruned_count = len(old_list) - len(new_list)
            if pruned_count > 0 and self.config.verbose:
                print(f"[State Management] Pruned {pruned_count} old detection(s) from Node {node_id}")

    # ========================================================================
    # Output and Display
    # ========================================================================

    def _print_track_update(
        self,
        triangulation_result: TriangulationResult,
        tracked_position: Tuple[float, float],
        tracked_velocity: Tuple[float, float]
    ):
        """
        Print formatted track update to console.

        Parameters
        ----------
        triangulation_result : TriangulationResult
            Raw triangulation output (time, position, residuals)
        tracked_position : Tuple[float, float]
            Filtered (x, y) from tracker
        tracked_velocity : Tuple[float, float]
            Filtered (vx, vy) from tracker
        """
        timestamp = triangulation_result.t_center
        raw_x, raw_y = triangulation_result.x, triangulation_result.y
        pos_x, pos_y = tracked_position
        vel_x, vel_y = tracked_velocity

        n_nodes = len(triangulation_result.node_ids)
        mean_residual = float(np.mean(np.abs(triangulation_result.residuals)))
        max_residual = float(np.max(np.abs(triangulation_result.residuals)))
        cov = triangulation_result.cov
        cov_summary = None
        if cov is not None:
            cov_summary = (
                float(np.sqrt(max(cov[0, 0], 0.0))),
                float(np.sqrt(max(cov[1, 1], 0.0)))
            )

        print("\n" + "="*70)
        print(f"TRACK UPDATE @ t={timestamp:.3f}s")
        print("="*70)

        # Raw triangulation
        print(f"\nRaw Triangulation:")
        print(f"  Position: ({raw_x:.2f}, {raw_y:.2f}) m")
        print(f"  Nodes contributing: {n_nodes}")
        print(f"  Residuals (mean / max): {mean_residual:.3f} / {max_residual:.3f} m")
        if cov_summary is not None:
            print(f"  Std dev (x / y): {cov_summary[0]:.3f} / {cov_summary[1]:.3f} m")

        # Tracked estimate
        print(f"\nTracked Estimate ({self.tracker_name}):")
        print(f"  Position: ({pos_x:.2f}, {pos_y:.2f}) m")
        print(f"  Velocity: ({vel_x:.2f}, {vel_y:.2f}) m/s")

        # Compute speed and heading
        speed = float(np.hypot(vel_x, vel_y))
        heading = float((np.degrees(np.arctan2(vel_y, vel_x)) + 360.0) % 360.0)
        print(f"  Speed: {speed:.2f} m/s")
        print(f"  Heading: {heading:.1f}°")

        print("="*70)

    # ========================================================================
    # Server Control
    # ========================================================================

    def start(self):
        """
        Start the fusion server (non-blocking).

        Starts two threads:
        1. Ingestion thread: Pulls detections from queue
        2. Fusion thread: Performs fusion and tracking

        Returns immediately. Call stop() to shutdown, or run() for blocking.
        """
        if self.running:
            print("[Server] Already running")
            return

        self.running = True

        # Start ingestion thread
        self.ingestion_thread = threading.Thread(
            target=self._ingestion_worker,
            name="IngestionThread",
            daemon=True
        )
        self.ingestion_thread.start()

        # Start UDP listener thread
        self.network_thread = threading.Thread(
            target=self._network_listener,
            name="NetworkThread",
            daemon=True
        )
        self.network_thread.start()

        # Start fusion thread
        self.fusion_thread = threading.Thread(
            target=self._fusion_worker,
            name="FusionThread",
            daemon=True
        )
        self.fusion_thread.start()

        if self.config.verbose:
            print("\n[Server] Started successfully")
            print("[Server] Press Ctrl+C to stop")

    def stop(self):
        """
        Stop the fusion server.

        Gracefully shuts down ingestion and fusion threads.
        """
        if not self.running:
            print("[Server] Not running")
            return

        if self.config.verbose:
            print("\n[Server] Stopping...")

        self.running = False

        # Wait for threads to finish
        if self.ingestion_thread:
            self.ingestion_thread.join(timeout=1.0)
        if self.network_thread:
            self.network_thread.join(timeout=1.0)
        if self.fusion_thread:
            self.fusion_thread.join(timeout=1.0)

        try:
            self.udp_socket.close()
        except OSError:
            pass

        if self.config.verbose:
            print("[Server] Stopped")
            self._print_statistics()

    def run(self):
        """
        Start the server and run until interrupted (blocking).

        This is a convenience method that starts the server and waits
        for Ctrl+C to stop it.
        """
        self.start()

        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[Server] Keyboard interrupt received")
        finally:
            self.stop()

    def _print_statistics(self):
        """Print server statistics."""
        print("\n" + "="*70)
        print("Server Statistics")
        print("="*70)
        print(f"  Total detections received: {self.stats['total_detections_received']}")
        print(f"  Total bearing sets created: {self.stats['total_bearing_sets_created']}")
        print(f"  Total triangulations: {self.stats['total_triangulations']}")
        print(f"  Total tracker updates: {self.stats['total_tracker_updates']}")
        print(f"  Total fusion ticks: {self.stats['fusion_ticks']}")
        print("="*70)


# ============================================================================
# Example Usage / Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example: Create a fusion server and test with simulated detections
    """
    import random

    print("="*70)
    print("Fusion Server Test")
    print("="*70)

    # Define node configuration
    node_ids = ['1', '2', '3']
    node_poses = {
        '1': NodePose(node_id='1', x=0.0, y=0.0, heading_deg=0.0),
        '2': NodePose(node_id='2', x=100.0, y=0.0, heading_deg=0.0),
        '3': NodePose(node_id='3', x=50.0, y=86.6, heading_deg=0.0)
    }

    # Create fusion config
    config = FusionConfig(
        fusion_tick_rate=0.5,  # 2 Hz (slow for testing)
        detection_window_seconds=10.0,
        bearing_fusion_dt=0.2,  # 200ms bins
        min_nodes_for_triangulation=2,
        use_kalman=False,  # Use Alpha-Beta for simpler output
        verbose=True
    )

    # Create server
    server = FusionServer(node_ids, node_poses, config)

    # Start server in separate thread
    server.start()

    print("\n[Test] Simulating detections from 3 nodes...")
    print("[Test] Drone flying in a circle at (50, 50) with radius 20m")

    # Simulate drone flying in a circle
    try:
        t = 0.0
        dt = 0.1  # Send detections every 100ms

        for i in range(50):  # Run for 5 seconds
            # Drone position (circular path)
            drone_x = 50.0 + 20.0 * np.cos(0.5 * t)
            drone_y = 50.0 + 20.0 * np.sin(0.5 * t)

            # Create detections from each node
            for node_id, pose in node_poses.items():
                # Calculate bearing from node to drone
                dx = drone_x - pose.x
                dy = drone_y - pose.y
                bearing_global_deg = np.degrees(np.arctan2(dx, dy)) % 360

                # Create detection (with some noise and occasional misses)
                if random.random() > 0.2:  # 80% detection rate
                    detection = NodeDetection(
                        node_id=node_id,
                        t_global=t,
                        x=pose.x,
                        y=pose.y,
                        bearing_local_deg=bearing_global_deg,
                        bearing_global_deg=bearing_global_deg + random.gauss(0, 2.0),  # Add noise
                        confidence=random.uniform(0.7, 0.95),
                        detected=True,
                        snr_db=random.uniform(10.0, 20.0)
                    )
                    server.add_detection(detection)

            time.sleep(dt)
            t += dt

        # Let server process remaining detections
        time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[Test] Interrupted")
    finally:
        server.stop()

    print("\n[Test] Complete")
