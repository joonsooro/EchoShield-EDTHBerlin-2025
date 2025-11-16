# alpha_beta_tracker.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AlphaBetaConfig:
    alpha: float = 0.85     # position correction weight (0–1)
    beta: float = 0.005     # velocity correction weight (0–1, but small)
    min_dt: float = 1e-3    # to avoid division by zero
    max_dt: float = 1.0     # clamp huge time gaps

class AlphaBetaTracker2D:
    """
    Simple 2D α-β tracker for (x, y) position and (vx, vy) velocity.

    State vector:
        x = [x, y, vx, vy]^T
    """

    def __init__(self, config: AlphaBetaConfig = None):
        self.cfg = config or AlphaBetaConfig()
        self.initialized = False
        self.x = np.zeros(4, dtype=float)   # [x, y, vx, vy]
        self.t_last = None

    def reset(self):
        """Reset the tracker (e.g., when track is lost)."""
        self.initialized = False
        self.t_last = None
        self.x[:] = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Current filtered position."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current filtered velocity."""
        return float(self.x[2]), float(self.x[3])

    def _compute_dt(self, t: float) -> float:
        if self.t_last is None:
            return 0.0
        dt = float(t - self.t_last)
        # clamp dt to reasonable bounds
        dt = max(self.cfg.min_dt, min(self.cfg.max_dt, dt))
        return dt

    def update(self, z_x: float, z_y: float, t: float):
        """
        Process a new measurement at time t.

        Parameters
        ----------
        z_x, z_y : float
            Measured position (global frame).
        t : float
            Timestamp in seconds (monotonic, same clock as other nodes).

        Returns
        -------
        pos : (float, float)
            Filtered position (x, y)
        vel : (float, float)
            Filtered velocity (vx, vy)
        """
        z = np.array([z_x, z_y], dtype=float)

        # First measurement: just initialize
        if not self.initialized:
            self.x[0:2] = z          # position
            self.x[2:4] = 0.0        # velocity
            self.t_last = float(t)
            self.initialized = True
            return self.position, self.velocity

        dt = self._compute_dt(t)

        # 1) Predict (constant velocity)
        x_pred = self.x.copy()
        x_pred[0] += x_pred[2] * dt   # x = x + vx*dt
        x_pred[1] += x_pred[3] * dt   # y = y + vy*dt
        # vx, vy unchanged

        # 2) Innovation (measurement residual)
        residual = z - x_pred[0:2]    # r = z - x_pred_pos

        # 3) Correct with α–β
        alpha = self.cfg.alpha
        beta = self.cfg.beta

        x_corr = x_pred.copy()
        x_corr[0:2] = x_pred[0:2] + alpha * residual
        x_corr[2:4] = x_pred[2:4] + (beta / dt) * residual

        # Update internal state
        self.x = x_corr
        self.t_last = float(t)

        return self.position, self.velocity
