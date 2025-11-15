# edge_bearing/estimator.py

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from . import framing_windowing
from . import fft
from . import gcc_phat_doa
from . import bearing_transform


@dataclass
class BearingState:
    """
    Optional internal state for smoothing across multiple chunks.
    You can extend this later (e.g., running mean, variance, last N bearings).
    """
    last_bearings: List[float] = field(default_factory=list)
    last_confidences: List[float] = field(default_factory=list)
    max_history: int = 10


def _frame_multi_channel(
    x: np.ndarray,
    fs: int,
    frame_length_ms: float = 32.0,
    hop_length_ms: float = 16.0,
    window: str = "hann",
) -> np.ndarray:
    """Frame a multi-channel signal using framing_windowing.frame_and_window.

    Args:
        x: Array of shape (n_samples, n_mics).
        fs: Sampling rate.
        frame_length_ms: Frame length in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        window: Window type string passed to frame_and_window.

    Returns:
        frames: Array of shape (n_frames, frame_length, n_mics).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (n_samples, n_mics), got shape={x.shape}")

    n_samples, n_mics = x.shape
    if n_mics < 2:
        raise ValueError("Need at least 2 mics for bearing estimation")

    frames_per_ch = []
    min_frames = None

    # Frame each channel independently using the real framing_windowing API
    for ch in range(n_mics):
        frames_ch, _times = framing_windowing.frame_and_window(
            x[:, ch],
            fs,
            frame_length_ms=frame_length_ms,
            hop_length_ms=hop_length_ms,
            window_type=window,
            zero_pad=False,
            normalize_energy=False,
        )
        if min_frames is None:
            min_frames = frames_ch.shape[0]
        else:
            min_frames = min(min_frames, frames_ch.shape[0])
        frames_per_ch.append(frames_ch)

    # Trim all channels to the same number of frames and stack into 3D tensor
    assert min_frames is not None
    trimmed = [fr[:min_frames] for fr in frames_per_ch]
    frames = np.stack(trimmed, axis=-1)  # (n_frames, frame_length, n_mics)
    return frames


def _normalize_audio_shape(audio_chunk: np.ndarray) -> np.ndarray:
    """
    Ensure shape (n_samples, n_mics). Many pipelines use (n_mics, n_samples),
    so we transpose if needed.
    """
    if audio_chunk.ndim != 2:
        raise ValueError(f"audio_chunk must be 2D, got shape={audio_chunk.shape}")

    n0, n1 = audio_chunk.shape
    # heuristic: if first dim is very small (<=4) treat as (n_mics, n_samples)
    if n0 <= 4 and n1 > n0:
        return audio_chunk.T.astype(np.float32, copy=False)
    return audio_chunk.astype(np.float32, copy=False)


def _select_active_frames(frames: np.ndarray, top_k: int = 8) -> np.ndarray:
    """
    Very simple energy-based frame selection.
    frames: shape (n_frames, frame_length, n_mics)

    Returns:
        indices of frames to run DOA on.
    """
    n_frames = frames.shape[0]
    if n_frames == 0:
        return np.array([], dtype=int)

    # total energy across mics
    energy = np.mean(np.sum(frames ** 2, axis=-1), axis=1)  # (n_frames,)
    # threshold = median energy
    thr = np.median(energy)
    active = np.where(energy >= thr)[0]
    if active.size == 0:
        return np.array([], dtype=int)

    # keep at most top_k by energy
    active_sorted = active[np.argsort(energy[active])[::-1]]
    return active_sorted[: min(top_k, active_sorted.size)]


def _select_active_frames_from_spectra(spectra: np.ndarray, top_k: int = 8) -> np.ndarray:
    """
    Energy-based frame selection using frequency-domain magnitude.

    spectra: ndarray with first dimension = n_frames, remaining dims = freq / channels.
    Returns indices of frames to run DOA on.
    """
    if spectra.ndim < 2:
        return np.array([], dtype=int)

    n_frames = spectra.shape[0]
    if n_frames == 0:
        return np.array([], dtype=int)

    # Use magnitude of spectra and collapse all non-frame axes into one for energy computation.
    mag = np.abs(spectra)
    flat = mag.reshape(n_frames, -1)
    energy = np.mean(flat, axis=1)  # (n_frames,)

    thr = np.median(energy)
    active = np.where(energy >= thr)[0]
    if active.size == 0:
        return np.array([], dtype=int)

    active_sorted = active[np.argsort(energy[active])[::-1]]
    return active_sorted[: min(top_k, active_sorted.size)]


def estimate_bearing(
    audio_chunk: np.ndarray,
    fs: int,
    mic_geometry: dict,
    state: Optional[BearingState] = None,
) -> Tuple[Optional[float], float, BearingState]:
    """
    Single-node bearing estimation using:
      - framing_windowing
      - fft
      - gcc_phat_doa
      - bearing_transform

    Args:
        audio_chunk: np.ndarray of shape (n_samples, n_mics) or (n_mics, n_samples).
        fs: sampling rate (e.g., 16000 or 48000).
        mic_geometry: { "positions": ..., "orientation_deg": float, ... }
        state: previous BearingState for smoothing (can be None for stateless).

    Returns:
        bearing_deg: float in [0, 360), or None if no reliable detection.
        confidence: float in [0.0, 1.0] (rough, can be refined later).
        state_out: updated BearingState.
    """
    # 0) init state
    if state is None:
        state = BearingState()

    # 1) shape normalization: (n_samples, n_mics)
    x = _normalize_audio_shape(audio_chunk)
    n_samples, n_mics = x.shape
    if n_mics < 2:
        # if monomic, DOA is not possible (Will return None here)
        return None, 0.0, state

    # 2) framing & windowing (multi-channel)
    # Use framing_windowing.frame_and_window per channel and stack into
    # a 3D tensor of shape (n_frames, frame_length, n_mics).
    frames = _frame_multi_channel(
        x,
        fs=fs,
        frame_length_ms=32.0,
        hop_length_ms=16.0,
        window="hann",
    )

    if frames.ndim != 3:
        raise ValueError(f"Expected frames to be 3D, got shape={frames.shape}")

    n_frames, frame_length, n_mics = frames.shape

    # 3) FFT on one reference channel for simple energy-based gating.
    # fft.compute_fft_per_frame has signature
    #   compute_fft_per_frame(frames: np.ndarray, fs, nfft=None, ...)
    # and expects frames shaped (n_frames, frame_length).
    freqs, spectrum, magnitude, magnitude_db = fft.compute_fft_per_frame(
        frames[:, :, 0],  # use first mic as reference
        fs=fs,
        nfft=None,
        remove_dc=True,
        window_compensation=False,
    )

    # 4) Energy-based frame selection from frequency domain. We use
    # the magnitude spectrum as input to the selector.
    active_idx = _select_active_frames_from_spectra(magnitude, top_k=8)
    if active_idx.size == 0:
        return None, 0.0, state

    doa_angles: List[float] = []

    # Get microphone spacing and heading from geometry. These keys must
    # be provided by the caller; sensible defaults are used as fallback.
    mic_spacing_m = float(
        mic_geometry.get("mic_spacing_m",
                         mic_geometry.get("d", 0.15))  # default 15 cm
    )
    heading_deg = float(
        mic_geometry.get("heading_deg",
                         mic_geometry.get("orientation_deg", 0.0))
    )

    # Maximum TDOA based on geometry
    max_tau = mic_spacing_m / gcc_phat_doa.SOUND_SPEED

    for idx in active_idx:
        frame = frames[idx]  # (frame_length, n_mics)

        # Use first two mics for a simple linear 2-mic array model.
        sig0 = frame[:, 0]
        sig1 = frame[:, 1]

        try:
            tau = gcc_phat_doa.gcc_phat_pair(
                sig0,
                sig1,
                fs=float(fs),
                max_tau=max_tau,
                interp=16,
            )
        except Exception:
            # If GCC-PHAT fails on this frame, skip it.
            continue

        # Convert TDOA to local DoA (radians → degrees)
        theta_rad = gcc_phat_doa.tdoa_to_doa_linear(
            tau,
            d=mic_spacing_m,
            c=gcc_phat_doa.SOUND_SPEED,
        )
        theta_deg_local = float(np.degrees(theta_rad))

        # Convert local DoA to global bearing using bearing_transform API
        bearing_global = bearing_transform.local_to_global_bearing(
            theta_deg_local,
            heading_deg,
        )
        if not np.isnan(bearing_global):
            doa_angles.append(float(bearing_global))

    if not doa_angles:
        # If no DOA could be extracted, return no bearing.
        return None, 0.0, state

    # 5) Average global bearings from valid frames.
    bearing_deg_raw = float(np.mean(doa_angles))

    # 6) Simple smoothing using recent history.
    state.last_bearings.append(float(bearing_deg_raw))
    state.last_confidences.append(1.0)  # placeholder, can be refined later

    # limit history length
    if len(state.last_bearings) > state.max_history:
        state.last_bearings = state.last_bearings[-state.max_history :]
        state.last_confidences = state.last_confidences[-state.max_history :]

    # Simple moving average over recent bearings
    bearing_deg_smooth = float(np.mean(state.last_bearings))

    # confidence is the ratio of successful DOA frames to active frames (rough placeholder).
    frac = len(doa_angles) / max(1, active_idx.size)
    confidence = float(np.clip(frac, 0.0, 1.0))

    return bearing_deg_smooth, confidence, state
