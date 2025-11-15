"""
gcc_phat_doa.py

Direction of Arrival (DoA) Estimation for Acoustic Drone Detection
===================================================================

This module implements Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
for estimating the direction of arrival of acoustic signals using microphone arrays.
This is critical for drone detection and localization:

1. **Spatial Localization**: Determines the angular position of the drone relative
   to the microphone array, enabling tracking and threat assessment

2. **Multi-Target Discrimination**: Helps distinguish between multiple drones or
   reject non-drone sources (birds, aircraft, vehicles)

3. **Enhanced Detection**: Spatial filtering improves signal-to-noise ratio by
   focusing on specific directions

Key Concepts:
- **TDOA (Time Difference of Arrival)**: Time delay between signals arriving at
  different microphones. For a source at angle θ, TDOA = (d/c) * sin(θ)

- **GCC-PHAT**: Cross-correlation with phase transform weighting. More robust to
  reverberation and noise than basic cross-correlation

- **Linear Array Geometry**: Microphones arranged in a line. Simple but effective
  for azimuth estimation in drone detection

- **Sound Speed**: ~343 m/s at 20°C. Varies with temperature and humidity

References:
- GCC-PHAT is standard for TDOA estimation in reverberant environments
- Typical drone detection uses 2-8 microphone arrays with 5-30 cm spacing
- Accuracy: ~5-10° angular resolution with good SNR
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


# Physical constants
SOUND_SPEED = 343.0  # m/s at 20°C, sea level
SPEED_OF_LIGHT = 343000.0  # mm/s (for mm distance units if needed)


# ============================================================================
# Input Validation Functions
# ============================================================================

def _validate_signals(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate input signals for GCC-PHAT.

    Parameters
    ----------
    x, y : np.ndarray
        Input signals from two microphones

    Returns
    -------
    x, y : np.ndarray
        Validated signals

    Raises
    ------
    ValueError
        If signals are invalid or incompatible
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()

    # Check for empty signals
    if x.size == 0 or y.size == 0:
        raise ValueError("Input signals cannot be empty")

    # Check for same length
    if len(x) != len(y):
        raise ValueError(
            f"Signals must have same length. Got x: {len(x)}, y: {len(y)}"
        )

    # Check minimum length
    if len(x) < 2:
        raise ValueError(
            f"Signals too short for correlation. Got {len(x)} samples, need at least 2."
        )

    # Check for NaN/Inf
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Signals contain NaN values")

    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("Signals contain Inf values")

    # Warn if signals are all zeros (silent)
    if np.all(x == 0) or np.all(y == 0):
        warnings.warn(
            "One or both signals are all zeros. TDOA estimation will be unreliable.",
            UserWarning
        )

    # Warn if signals have very low energy
    energy_x = np.sum(x ** 2)
    energy_y = np.sum(y ** 2)
    if energy_x < 1e-10 or energy_y < 1e-10:
        warnings.warn(
            f"Very low signal energy (x: {energy_x:.2e}, y: {energy_y:.2e}). "
            f"TDOA estimation may be unreliable.",
            UserWarning
        )

    return x, y


def _validate_multi_channel_frames(frames: np.ndarray) -> np.ndarray:
    """
    Validate multi-channel frames array.

    Parameters
    ----------
    frames : np.ndarray
        Multi-channel frames array

    Returns
    -------
    frames : np.ndarray
        Validated frames array

    Raises
    ------
    ValueError
        If frames are invalid
    """
    frames = np.asarray(frames, dtype=np.float64)

    # Check dimensionality
    if frames.ndim != 3:
        raise ValueError(
            f"Multi-channel frames must be 3D array [n_frames, frame_len, n_channels]. "
            f"Got shape {frames.shape} with {frames.ndim} dimensions."
        )

    n_frames, frame_len, n_channels = frames.shape

    # Check for reasonable dimensions
    if n_frames == 0:
        raise ValueError("Number of frames cannot be zero")

    if frame_len < 2:
        raise ValueError(
            f"Frame length too short for correlation. Got {frame_len}, need at least 2."
        )

    if n_channels < 2:
        raise ValueError(
            f"Need at least 2 channels for DoA estimation. Got {n_channels}."
        )

    # Check for NaN/Inf
    if np.any(np.isnan(frames)):
        raise ValueError("Frames contain NaN values")

    if np.any(np.isinf(frames)):
        raise ValueError("Frames contain Inf values")

    return frames


def _validate_array_geometry(
    mic_spacing: float,
    n_channels: int,
    c: float = SOUND_SPEED
) -> None:
    """
    Validate microphone array geometry parameters.

    Parameters
    ----------
    mic_spacing : float
        Distance between adjacent microphones [meters]
    n_channels : int
        Number of microphones
    c : float
        Sound speed [m/s]

    Raises
    ------
    ValueError
        If geometry parameters are invalid or lead to spatial aliasing
    """
    if mic_spacing <= 0:
        raise ValueError(f"Microphone spacing must be positive. Got {mic_spacing} m.")

    if mic_spacing > 1.0:
        warnings.warn(
            f"Very large microphone spacing ({mic_spacing} m). "
            f"This may cause spatial aliasing for high frequencies.",
            UserWarning
        )

    if mic_spacing < 0.01:
        warnings.warn(
            f"Very small microphone spacing ({mic_spacing*1000:.1f} mm). "
            f"TDOA resolution may be limited by sampling rate.",
            UserWarning
        )

    if c <= 0:
        raise ValueError(f"Sound speed must be positive. Got {c} m/s.")

    if c < 300 or c > 400:
        warnings.warn(
            f"Unusual sound speed ({c} m/s). Typical range is 331-343 m/s. "
            f"Check if temperature compensation is needed.",
            UserWarning
        )

    # Check for spatial aliasing
    # For unambiguous DoA, need d < λ/2, where λ = c/f
    # At typical drone frequency (1 kHz), λ = 0.343 m
    # So for 1 kHz, need d < 0.172 m (17.2 cm)
    max_unambiguous_freq = c / (2 * mic_spacing)
    if max_unambiguous_freq < 1000:
        warnings.warn(
            f"Microphone spacing ({mic_spacing*100:.1f} cm) may cause spatial aliasing "
            f"above {max_unambiguous_freq:.0f} Hz. For unambiguous DoA up to 5 kHz, "
            f"consider spacing < {c/(2*5000)*100:.1f} cm.",
            UserWarning
        )


def _validate_sampling_rate(fs: Union[int, float], mic_spacing: float) -> float:
    """
    Validate sampling rate for TDOA estimation.

    Parameters
    ----------
    fs : int or float
        Sampling rate [Hz]
    mic_spacing : float
        Microphone spacing [m]

    Returns
    -------
    fs : float
        Validated sampling rate

    Raises
    ------
    ValueError
        If sampling rate is invalid
    """
    fs = float(fs)

    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive. Got {fs} Hz.")

    if fs < 8000:
        warnings.warn(
            f"Low sampling rate ({fs} Hz). TDOA resolution will be limited. "
            f"Consider fs ≥ 16 kHz for better angular accuracy.",
            UserWarning
        )

    # Check TDOA resolution
    # Maximum TDOA = mic_spacing / c
    # TDOA resolution = 1/fs
    # Need several samples per maximum TDOA for good resolution
    max_tdoa = mic_spacing / SOUND_SPEED
    tdoa_samples = max_tdoa * fs
    if tdoa_samples < 2:
        warnings.warn(
            f"TDOA resolution limited: maximum TDOA is {tdoa_samples:.1f} samples. "
            f"For spacing {mic_spacing*100:.1f} cm, consider fs ≥ {SOUND_SPEED/mic_spacing*4:.0f} Hz "
            f"for better resolution.",
            UserWarning
        )

    return fs


# ============================================================================
# Core GCC-PHAT Algorithm
# ============================================================================

def gcc_phat_pair(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    max_tau: float,
    interp: int = 16,
    epsilon: float = 1e-15,
) -> float:
    """
    Compute Time Difference of Arrival (TDOA) between two signals using GCC-PHAT.

    GCC-PHAT (Generalized Cross-Correlation with Phase Transform) is robust to
    reverberation and noise. It emphasizes phase information while suppressing
    magnitude variations, making it ideal for drone detection in outdoor environments.

    Algorithm:
    1. Compute FFT of both signals
    2. Calculate cross-power spectrum: R(ω) = X(ω) * conj(Y(ω))
    3. Apply PHAT weighting: R_phat(ω) = R(ω) / |R(ω)| (phase-only)
    4. Inverse FFT to get cross-correlation
    5. Find peak within valid TDOA range
    6. Convert lag to time delay

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays (same length) containing signals from two microphones.
        Should be from the same time frame after framing/windowing.
    fs : float
        Sampling rate [Hz]. Must match the rate used during acquisition.
    max_tau : float
        Maximum allowed absolute time delay [seconds].
        Based on array geometry: max_tau = d_max / c, where d_max is the
        maximum distance between microphones and c is sound speed.
        This limits the search range and prevents spurious peaks.
    interp : int, default=16
        Interpolation factor for sub-sample TDOA estimation.
        Higher values give finer delay resolution but increase computation.
        Typical: 8-32. 16 provides ~0.004 ms resolution at 16 kHz.
    epsilon : float, default=1e-15
        Small constant to avoid division by zero in PHAT normalization.
        Should be much smaller than typical signal magnitudes.

    Returns
    -------
    tau_hat : float
        Estimated time delay [seconds] of y relative to x.
        - tau_hat > 0: signal arrives at y later (source closer to x)
        - tau_hat < 0: signal arrives at y earlier (source closer to y)
        - tau_hat = 0: source is equidistant (broadside arrival)

    Examples
    --------
    >>> # Simple example: delayed sine wave
    >>> fs = 16000  # 16 kHz
    >>> t = np.arange(0, 0.1, 1/fs)
    >>> true_delay = 0.002  # 2 ms delay
    >>> x = np.sin(2 * np.pi * 200 * t)
    >>> y = np.sin(2 * np.pi * 200 * (t - true_delay))
    >>>
    >>> # Estimate TDOA
    >>> mic_spacing = 0.1  # 10 cm
    >>> max_tau = mic_spacing / 343.0  # ~0.29 ms
    >>> tau_est = gcc_phat_pair(x, y, fs, max_tau)
    >>> print(f"True delay: {true_delay*1000:.2f} ms")
    >>> print(f"Estimated: {tau_est*1000:.2f} ms")

    Notes
    -----
    - Works best with broadband signals (drones produce broadband harmonics)
    - Robust to reverberation due to phase-only weighting
    - May fail for very low SNR (<0 dB) or highly correlated noise
    - For drones, typical TDOAs are 0.1-1.0 ms for 5-30 cm array spacing
    - Consider using multiple frame averaging for more stable estimates

    References
    ----------
    Knapp, C. and Carter, G. (1976). "The generalized correlation method for
    estimation of time delay." IEEE Transactions on Acoustics, Speech, and
    Signal Processing.
    """

    # ========================================================================
    # Step 1: Validate inputs
    # ========================================================================

    x, y = _validate_signals(x, y)
    fs = _validate_sampling_rate(fs, max_tau * SOUND_SPEED)

    if max_tau <= 0:
        raise ValueError(f"max_tau must be positive. Got {max_tau} seconds.")

    if interp < 1:
        raise ValueError(f"Interpolation factor must be >= 1. Got {interp}.")

    if interp > 64:
        warnings.warn(
            f"Very high interpolation factor ({interp}). "
            f"This increases computation without much benefit.",
            UserWarning
        )

    # ========================================================================
    # Step 2: Compute FFT size (with padding for better resolution)
    # ========================================================================

    # Use zero-padding to avoid circular correlation artifacts
    # FFT size = sum of lengths gives linear cross-correlation
    n = x.shape[0] + y.shape[0]

    # ========================================================================
    # Step 3: Compute FFTs
    # ========================================================================

    try:
        X = np.fft.rfft(x, n=n)
        Y = np.fft.rfft(y, n=n)
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {e}") from e

    # ========================================================================
    # Step 4: Compute cross-power spectrum
    # ========================================================================

    # Cross-power spectrum: R(ω) = X(ω) * conj(Y(ω))
    R = X * np.conj(Y)

    # Check for zero cross-power (uncorrelated signals)
    if np.all(np.abs(R) < epsilon):
        warnings.warn(
            "Cross-power spectrum is nearly zero. Signals may be uncorrelated. "
            "Returning zero TDOA.",
            UserWarning
        )
        return 0.0

    # ========================================================================
    # Step 5: Apply PHAT weighting (phase transform)
    # ========================================================================

    # PHAT: normalize by magnitude to get phase-only correlation
    # R_phat(ω) = R(ω) / |R(ω)|
    # This makes the method robust to reverberation and magnitude variations
    R_norm = R / (np.abs(R) + epsilon)

    # ========================================================================
    # Step 6: Compute cross-correlation via inverse FFT
    # ========================================================================

    # Apply interpolation for sub-sample accuracy
    # Interpolation increases the number of points in the correlation
    try:
        cc = np.fft.irfft(R_norm, n=n * interp)
    except Exception as e:
        raise RuntimeError(f"Inverse FFT failed: {e}") from e

    # ========================================================================
    # Step 7: Limit search to valid TDOA range
    # ========================================================================

    # Convert max_tau to samples (in the interpolated domain)
    max_shift = int(interp * fs * max_tau)

    # Ensure max_shift doesn't exceed correlation length
    if max_shift >= cc.size // 2:
        max_shift = cc.size // 2 - 1
        warnings.warn(
            f"max_tau ({max_tau:.4f} s) is too large for signal length. "
            f"Limiting search range to {max_shift/(interp*fs):.4f} s.",
            UserWarning
        )

    if max_shift < 1:
        warnings.warn(
            f"max_tau ({max_tau:.4f} s) is very small. "
            f"Only checking {max_shift} lag(s). Consider increasing max_tau.",
            UserWarning
        )

    # ========================================================================
    # Step 8: Find peak in cross-correlation
    # ========================================================================

    # Build lag array: [-max_shift, ..., -1, 0, 1, ..., max_shift]
    lags = np.arange(-max_shift, max_shift + 1)

    # Extract relevant portion of cross-correlation
    # CC is computed circularly, so we need to shift:
    # Negative lags are at the end of cc, positive lags at the beginning
    cc_shifted = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # Find lag with maximum correlation magnitude
    best_idx = np.argmax(np.abs(cc_shifted))
    best_lag = lags[best_idx]

    # Optional: could use parabolic interpolation for even finer resolution
    # For now, use simple peak detection

    # ========================================================================
    # Step 9: Convert lag to time delay
    # ========================================================================

    # tau = lag / (interp * fs)
    # Divide by interp because lags are in interpolated sample space
    tau_hat = best_lag / float(interp * fs)

    return tau_hat


# ============================================================================
# TDOA to DoA Conversion
# ============================================================================

def tdoa_to_doa_linear(
    tau: float,
    d: float,
    c: float = SOUND_SPEED,
) -> float:
    """
    Convert Time Difference of Arrival (TDOA) to Direction of Arrival (DoA)
    for a 2-microphone linear array.

    For a linear array with two microphones separated by distance d, if a
    plane wave arrives at angle θ relative to broadside:
        τ = (d/c) * sin(θ)

    Solving for θ:
        θ = arcsin(c * τ / d)

    Coordinate System:
    - θ = 0°: Broadside (perpendicular to array axis)
    - θ > 0: Source on positive side (towards mic 2)
    - θ < 0: Source on negative side (towards mic 1)
    - θ = ±90°: Endfire (along array axis)

    Parameters
    ----------
    tau : float
        Time difference of arrival [seconds].
        Positive: signal arrives at mic 2 first
        Negative: signal arrives at mic 1 first
    d : float
        Distance between the two microphones [meters].
        Must be positive.
    c : float, default=343.0
        Sound speed [m/s]. Depends on temperature:
        c ≈ 331.3 + 0.606 * T_celsius

    Returns
    -------
    theta : float
        Angle of arrival [radians], measured from broadside.
        Range: [-π/2, π/2] (±90°)
        - 0: Broadside (perpendicular arrival)
        - Positive: Source in positive half-plane
        - Negative: Source in negative half-plane
        - ±π/2: Endfire (parallel arrival)

    Examples
    --------
    >>> # Broadside arrival (90° incidence)
    >>> tau = 0.0  # No time delay
    >>> d = 0.1  # 10 cm spacing
    >>> theta = tdoa_to_doa_linear(tau, d)
    >>> print(f"Angle: {np.degrees(theta):.1f}°")  # 0.0°

    >>> # 45° arrival
    >>> tau = (d / 343.0) * np.sin(np.radians(45))
    >>> theta = tdoa_to_doa_linear(tau, d)
    >>> print(f"Angle: {np.degrees(theta):.1f}°")  # ~45.0°

    >>> # Endfire arrival (0° incidence)
    >>> tau = d / 343.0  # Maximum delay
    >>> theta = tdoa_to_doa_linear(tau, d)
    >>> print(f"Angle: {np.degrees(theta):.1f}°")  # ~90.0°

    Notes
    -----
    - Assumes far-field plane wave (source >> array size)
    - Only valid for |c*τ/d| ≤ 1 (clipped to avoid arcsin domain error)
    - For near-field sources, need more complex spherical wave model
    - Angle is ambiguous: θ and -θ give same correlation pattern
    - Multi-microphone arrays (>2) help resolve ambiguities
    """

    if d <= 0:
        raise ValueError(f"Microphone distance must be positive. Got {d} m.")

    if c <= 0:
        raise ValueError(f"Sound speed must be positive. Got {c} m/s.")

    # Calculate arcsin argument
    arg = c * tau / d

    # Check if TDOA is physically possible for this geometry
    if abs(arg) > 1.1:  # Allow 10% margin for numerical errors
        warnings.warn(
            f"TDOA ({tau*1000:.3f} ms) is too large for microphone spacing ({d*100:.1f} cm). "
            f"Maximum theoretical TDOA is {d/c*1000:.3f} ms. "
            f"This may indicate an error in TDOA estimation or array geometry.",
            UserWarning
        )

    # Clip to [-1, 1] to avoid domain error in arcsin
    # This is necessary due to numerical errors in GCC-PHAT
    arg = np.clip(arg, -1.0, 1.0)

    # Compute angle
    theta = np.arcsin(arg)

    return theta


def temperature_compensated_sound_speed(temperature_celsius: float) -> float:
    """
    Calculate sound speed as a function of air temperature.

    Sound speed varies significantly with temperature, affecting DoA accuracy.
    This function provides temperature compensation for better localization.

    Formula (empirical):
        c ≈ 331.3 + 0.606 * T [m/s]

    Valid for temperatures -20°C to +40°C at sea level.

    Parameters
    ----------
    temperature_celsius : float
        Air temperature in degrees Celsius

    Returns
    -------
    c : float
        Sound speed in m/s

    Examples
    --------
    >>> c_0 = temperature_compensated_sound_speed(0)   # 0°C
    >>> c_20 = temperature_compensated_sound_speed(20)  # 20°C
    >>> c_40 = temperature_compensated_sound_speed(40)  # 40°C
    >>> print(f"Sound speed at 0°C: {c_0:.1f} m/s")   # 331.3
    >>> print(f"Sound speed at 20°C: {c_20:.1f} m/s")  # 343.4
    >>> print(f"Sound speed at 40°C: {c_40:.1f} m/s")  # 355.5

    Notes
    -----
    - Assumes dry air at sea level
    - Humidity has minor effect (~0.1-0.6% variation)
    - For precise applications, consider full atmospheric model
    """
    if temperature_celsius < -50 or temperature_celsius > 50:
        warnings.warn(
            f"Temperature ({temperature_celsius}°C) is outside typical range (-20 to 40°C). "
            f"Sound speed calculation may be inaccurate.",
            UserWarning
        )

    c = 331.3 + 0.606 * temperature_celsius
    return c


# ============================================================================
# Multi-Channel DoA Estimation
# ============================================================================

def estimate_doa_gcc_phat(
    frames: np.ndarray,
    fs: float,
    mic_spacing: float,
    ref_channel: int = 0,
    c: float = SOUND_SPEED,
    interp: int = 16,
    robust_averaging: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Direction of Arrival (DoA) for multi-channel acoustic frames using GCC-PHAT.

    This function processes multi-channel audio frames to estimate the angular
    position of an acoustic source (drone) over time. It uses a linear microphone
    array and computes TDOA between microphone pairs, then converts to DoA angles.

    Process Flow (per frame):
    1. Select reference microphone
    2. For each other microphone, compute TDOA using GCC-PHAT
    3. Convert each TDOA to a DoA angle
    4. Combine multiple angle estimates (averaging or median)
    5. Return time series of DoA estimates

    Parameters
    ----------
    frames : np.ndarray
        Multi-channel frames array of shape [n_frames, frame_len, n_channels].
        - n_frames: Number of time frames
        - frame_len: Samples per frame (after windowing)
        - n_channels: Number of microphones
        Frames should be windowed (Hann/Hamming) for best results.
    fs : float
        Sampling rate [Hz]. Must match acquisition rate.
    mic_spacing : float
        Distance between adjacent microphones [meters] in the linear array.
        For a uniform linear array (ULA) along x-axis with spacing d:
        - Mics at positions: 0, d, 2d, 3d, ... (n_channels-1)*d
        Example: 0.1 m = 10 cm spacing (typical for drone detection)
    ref_channel : int, default=0
        Index of reference microphone (0-indexed).
        All TDOAs are computed relative to this microphone.
        Typically use 0 (first mic) or center mic for symmetric arrays.
    c : float, default=343.0
        Sound speed [m/s]. Default is for 20°C air.
        Use temperature_compensated_sound_speed() for better accuracy.
    interp : int, default=16
        Interpolation factor for GCC-PHAT (sub-sample resolution).
        Higher values give finer TDOA estimates. Typical: 8-32.
    robust_averaging : bool, default=True
        If True, use median to combine multiple angle estimates (robust to outliers).
        If False, use mean (faster but sensitive to outliers).

    Returns
    -------
    doa_series : np.ndarray
        Estimated DoA angles [radians] for each frame, shape [n_frames].
        - Range: [-π/2, π/2] (±90°)
        - 0 rad = broadside (perpendicular to array)
        - Positive = source on positive side
        - Negative = source on negative side
    tdoa_series : np.ndarray
        Raw TDOA estimates [seconds] for each microphone pair and frame.
        Shape: [n_frames, n_pairs], where n_pairs = n_channels - 1.
        Useful for debugging and advanced processing.

    Raises
    ------
    ValueError
        If input dimensions are invalid or n_channels < 2

    Examples
    --------
    >>> # Setup: 4-microphone array, 10 cm spacing
    >>> n_mics = 4
    >>> mic_spacing = 0.1  # 10 cm
    >>> fs = 16000  # 16 kHz
    >>>
    >>> # Generate some frames (e.g., from framing_windowing.py)
    >>> # For this example, use random data
    >>> n_frames = 100
    >>> frame_len = 512
    >>> frames = np.random.randn(n_frames, frame_len, n_mics)
    >>>
    >>> # Estimate DoA
    >>> doa_angles, tdoas = estimate_doa_gcc_phat(
    ...     frames, fs, mic_spacing,
    ...     ref_channel=0,
    ...     c=343.0,
    ...     interp=16
    ... )
    >>>
    >>> # Convert to degrees for interpretation
    >>> doa_degrees = np.degrees(doa_angles)
    >>> print(f"Mean DoA: {doa_degrees.mean():.1f}°")
    >>> print(f"DoA std: {doa_degrees.std():.1f}°")

    >>> # Track drone movement over time
    >>> import matplotlib.pyplot as plt
    >>> frame_times = np.arange(n_frames) * (frame_len / fs)
    >>> plt.plot(frame_times, doa_degrees)
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Angle of Arrival (degrees)')
    >>> plt.title('Drone Angular Position Over Time')
    >>> plt.grid(True)
    >>> plt.show()

    Notes
    -----
    - Assumes far-field plane wave (drone distance >> array size)
    - Requires at least 2 microphones (more is better for robustness)
    - Array geometry: linear (ULA) along x-axis
    - For drones, expect angles to change slowly (tracking smoothness)
    - Consider temporal smoothing (moving average) for stable tracking
    - Front-back ambiguity: angle θ and -θ are indistinguishable with linear array
    - Use circular or 2D array to resolve 360° azimuth

    Performance Tips:
    - Use windowed frames (Hann/Hamming) for better cross-correlation
    - Typical frame length: 32-64 ms for good SNR vs temporal resolution
    - For real-time, process frames as they arrive
    - Consider multi-frame averaging for more stable estimates

    Limitations:
    - Fails in highly reverberant environments (indoors)
    - Requires broadband source (drones are good: harmonics provide broadband energy)
    - Low SNR (<0 dB) degrades accuracy significantly
    - Multiple simultaneous drones cause interference
    """

    # ========================================================================
    # Step 1: Validate inputs
    # ========================================================================

    frames = _validate_multi_channel_frames(frames)
    n_frames, frame_len, n_channels = frames.shape

    if ref_channel < 0 or ref_channel >= n_channels:
        raise ValueError(
            f"ref_channel must be in range [0, {n_channels-1}]. Got {ref_channel}."
        )

    _validate_array_geometry(mic_spacing, n_channels, c)
    fs = _validate_sampling_rate(fs, mic_spacing)

    # ========================================================================
    # Step 2: Calculate maximum physical TDOA
    # ========================================================================

    # Maximum distance from reference mic to furthest mic
    max_idx_offset = max(
        abs(ch - ref_channel) for ch in range(n_channels)
    )
    d_max = mic_spacing * max_idx_offset
    max_tau = d_max / c  # Maximum TDOA in seconds

    # ========================================================================
    # Step 3: Initialize output arrays
    # ========================================================================

    # Number of microphone pairs (all relative to reference)
    n_pairs = n_channels - 1

    tdoa_series = np.zeros((n_frames, n_pairs), dtype=np.float32)
    doa_series = np.zeros(n_frames, dtype=np.float32)

    # ========================================================================
    # Step 4: Process each frame
    # ========================================================================

    for i in range(n_frames):
        frame = frames[i, :, :]  # Shape: [frame_len, n_channels]
        x_ref = frame[:, ref_channel]  # Reference microphone signal

        pair_idx = 0
        theta_list = []

        # 4.1: Loop over all other microphones
        for ch in range(n_channels):
            if ch == ref_channel:
                continue  # Skip reference mic itself

            x_other = frame[:, ch]

            # 4.2: Estimate TDOA using GCC-PHAT
            try:
                tau_hat = gcc_phat_pair(
                    x_ref,
                    x_other,
                    fs=fs,
                    max_tau=max_tau,
                    interp=interp,
                )
            except Exception as e:
                warnings.warn(
                    f"GCC-PHAT failed for frame {i}, channel {ch}: {e}. Using tau=0.",
                    UserWarning
                )
                tau_hat = 0.0

            tdoa_series[i, pair_idx] = tau_hat

            # 4.3: Convert TDOA to DoA angle
            # Distance between these two specific microphones
            d_pair = mic_spacing * abs(ch - ref_channel)

            try:
                theta = tdoa_to_doa_linear(tau_hat, d_pair, c=c)
            except Exception as e:
                warnings.warn(
                    f"TDOA to DoA conversion failed for frame {i}, channel {ch}: {e}. Using theta=0.",
                    UserWarning
                )
                theta = 0.0

            theta_list.append(theta)

            pair_idx += 1

        # 4.4: Combine multiple angle estimates for this frame
        if len(theta_list) > 0:
            if robust_averaging:
                # Use median (robust to outliers)
                doa_series[i] = float(np.median(theta_list))
            else:
                # Use mean (simpler, but sensitive to outliers)
                doa_series[i] = float(np.mean(theta_list))
        else:
            # Fallback if no valid estimates
            doa_series[i] = 0.0
            warnings.warn(
                f"No valid DoA estimates for frame {i}. Setting to 0.",
                UserWarning
            )

    return doa_series, tdoa_series


# ============================================================================
# Utility Functions
# ============================================================================

def smooth_doa_series(
    doa_series: np.ndarray,
    window_size: int = 5,
    method: str = 'median'
) -> np.ndarray:
    """
    Smooth DoA time series using moving average or median filter.

    Useful for reducing jitter in drone tracking. Drones move relatively slowly
    (angular velocity ~1-10°/s), so rapid angle changes are likely noise.

    Parameters
    ----------
    doa_series : np.ndarray
        DoA angles over time [radians], shape [n_frames]
    window_size : int, default=5
        Size of smoothing window (odd number recommended)
    method : str, default='median'
        Smoothing method: 'median' (robust) or 'mean' (simple)

    Returns
    -------
    smoothed : np.ndarray
        Smoothed DoA series [radians]

    Examples
    --------
    >>> doa_noisy = np.radians([10, 12, 8, 45, 11, 13, 10, 9])  # 45° is outlier
    >>> doa_smooth = smooth_doa_series(doa_noisy, window_size=3, method='median')
    >>> print(np.degrees(doa_smooth))  # Outlier suppressed
    """
    if window_size < 1:
        raise ValueError(f"Window size must be >= 1. Got {window_size}.")

    if window_size % 2 == 0:
        warnings.warn(
            f"Even window size ({window_size}). Odd sizes are recommended for symmetry.",
            UserWarning
        )

    if window_size >= len(doa_series):
        warnings.warn(
            f"Window size ({window_size}) >= signal length ({len(doa_series)}). "
            f"Returning original signal.",
            UserWarning
        )
        return doa_series.copy()

    # Pad signal at boundaries (reflect mode)
    half_window = window_size // 2
    padded = np.pad(doa_series, half_window, mode='edge')

    smoothed = np.zeros_like(doa_series)

    for i in range(len(doa_series)):
        window = padded[i:i + window_size]

        if method == 'median':
            smoothed[i] = np.median(window)
        elif method == 'mean':
            smoothed[i] = np.mean(window)
        else:
            raise ValueError(f"Unknown smoothing method: {method}. Use 'median' or 'mean'.")

    return smoothed


# ============================================================================
# Demo / Test Code
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of GCC-PHAT DoA estimation for acoustic drone detection.

    This demo:
    1. Simulates a drone signal at a known angle
    2. Generates multi-channel microphone array signals with time delays
    3. Estimates DoA using GCC-PHAT
    4. Compares estimated vs true angle
    """
    print("=" * 70)
    print("Acoustic Drone Detection: GCC-PHAT DoA Estimation Demo")
    print("=" * 70)

    # ========================================================================
    # Configuration
    # ========================================================================

    fs = 16000  # 16 kHz
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Microphone array setup (linear array)
    n_mics = 4
    mic_spacing = 0.1  # 10 cm between adjacent mics
    c = 343.0  # Sound speed at 20°C

    # True drone position
    true_angle_deg = 30.0  # 30° from broadside
    true_angle_rad = np.radians(true_angle_deg)

    print(f"\nArray Configuration:")
    print(f"  Number of microphones: {n_mics}")
    print(f"  Microphone spacing: {mic_spacing * 100:.1f} cm")
    print(f"  Total array length: {(n_mics-1) * mic_spacing * 100:.1f} cm")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Sound speed: {c} m/s")

    print(f"\nTrue Drone Position:")
    print(f"  Angle: {true_angle_deg:.1f}° from broadside")

    # ========================================================================
    # Generate synthetic drone signal
    # ========================================================================

    # Drone acoustic signature (multi-harmonic)
    fundamental = 150  # Hz
    signal = (
        0.5 * np.sin(2 * np.pi * fundamental * t) +
        0.3 * np.sin(2 * np.pi * 2 * fundamental * t) +
        0.2 * np.sin(2 * np.pi * 3 * fundamental * t) +
        0.03 * np.random.randn(len(t))  # Noise
    )

    # ========================================================================
    # Simulate multi-channel array reception
    # ========================================================================

    # Calculate time delays for each microphone
    # For mic at position x_i = i * d, delay = (x_i / c) * sin(θ)
    mic_positions = np.arange(n_mics) * mic_spacing
    time_delays = (mic_positions / c) * np.sin(true_angle_rad)

    print(f"\nMicrophone time delays (relative to mic 0):")
    for i, delay in enumerate(time_delays):
        print(f"  Mic {i}: {delay * 1000:.3f} ms")

    # Generate delayed signals for each microphone
    multi_channel = np.zeros((len(t), n_mics))
    for i in range(n_mics):
        # Delay signal by shifting (simple interpolation)
        delay_samples = int(time_delays[i] * fs)
        if delay_samples > 0:
            multi_channel[delay_samples:, i] = signal[:-delay_samples]
        elif delay_samples < 0:
            multi_channel[:delay_samples, i] = signal[-delay_samples:]
        else:
            multi_channel[:, i] = signal

    # ========================================================================
    # Frame the multi-channel signal
    # ========================================================================

    frame_length_ms = 64.0  # 64 ms frames
    hop_length_ms = 32.0    # 50% overlap
    frame_length_samples = int(fs * frame_length_ms / 1000)
    hop_length_samples = int(fs * hop_length_ms / 1000)

    n_frames = 1 + (len(signal) - frame_length_samples) // hop_length_samples

    frames = np.zeros((n_frames, frame_length_samples, n_mics))

    for i in range(n_frames):
        start = i * hop_length_samples
        end = start + frame_length_samples
        frames[i, :, :] = multi_channel[start:end, :]

    print(f"\nFraming:")
    print(f"  Frame length: {frame_length_ms} ms ({frame_length_samples} samples)")
    print(f"  Hop length: {hop_length_ms} ms ({hop_length_samples} samples)")
    print(f"  Number of frames: {n_frames}")
    print(f"  Frames shape: {frames.shape}")

    # ========================================================================
    # Estimate DoA using GCC-PHAT
    # ========================================================================

    print(f"\nEstimating DoA using GCC-PHAT...")

    doa_series, tdoa_series = estimate_doa_gcc_phat(
        frames,
        fs,
        mic_spacing,
        ref_channel=0,
        c=c,
        interp=16,
        robust_averaging=True
    )

    # Convert to degrees
    doa_degrees = np.degrees(doa_series)

    print(f"\nDoA Estimation Results:")
    print(f"  Mean estimated angle: {doa_degrees.mean():.1f}°")
    print(f"  Std deviation: {doa_degrees.std():.1f}°")
    print(f"  True angle: {true_angle_deg:.1f}°")
    print(f"  Error: {abs(doa_degrees.mean() - true_angle_deg):.1f}°")

    # ========================================================================
    # Visualization
    # ========================================================================

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: DoA over time
        frame_times = np.arange(n_frames) * (hop_length_samples / fs)
        axes[0].plot(frame_times, doa_degrees, marker='o', linestyle='-', label='Estimated DoA')
        axes[0].axhline(true_angle_deg, color='r', linestyle='--', label=f'True DoA ({true_angle_deg}°)')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Angle of Arrival (degrees)')
        axes[0].set_title('DoA Estimation Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: TDOA for each microphone pair
        for pair_idx in range(tdoa_series.shape[1]):
            axes[1].plot(frame_times, tdoa_series[:, pair_idx] * 1000,
                        marker='o', linestyle='-', label=f'Pair {pair_idx+1}')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('TDOA (milliseconds)')
        axes[1].set_title('Time Difference of Arrival for Each Microphone Pair')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = '/mnt/d/edth/gcc_phat_doa_demo.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")

    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
