"""
harmonic_filter.py

Frequency and Harmonic Filtering for Acoustic Drone Detection
==============================================================

This module provides frequency-domain filtering techniques specifically designed
for enhancing drone acoustic signatures and suppressing noise. Filtering is critical
for drone detection because:

1. **Frequency Selectivity**: Drones produce energy in specific frequency bands
   (propeller fundamentals: 50-250 Hz, harmonics up to 5 kHz). Filtering removes
   out-of-band noise from wind, traffic, birds, etc.

2. **Harmonic Enhancement**: Drone propellers generate strong harmonics at integer
   multiples of the fundamental frequency (f0, 2f0, 3f0, ...). Selective filtering
   can isolate these harmonics for better detection.

3. **SNR Improvement**: By focusing only on relevant frequency bands, we significantly
   improve Signal-to-Noise Ratio, leading to more reliable detection.

Key Techniques:
- **Coarse Band-Pass**: Broad frequency range (e.g., 100-5000 Hz) to reject
  very low frequencies (wind, building vibrations) and very high frequencies
  (electronic noise, ultrasonic interference)

- **Harmonic Selection**: Fine-tuned filtering around specific harmonics of a
  detected fundamental frequency. Useful when f0 is known or estimated.

- **Adaptive Bandwidth**: Adjustable bandwidth around each harmonic to account
  for frequency variations due to RPM changes, Doppler effect, etc.

References:
- Drone acoustic signatures are quasi-periodic with strong harmonics
- Typical fundamental: 80-200 Hz (depends on propeller size and RPM)
- Number of harmonics: 5-10 typically detectable above noise floor
- Bandwidth per harmonic: 10-50 Hz (accounts for modulation and Doppler)
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


# ============================================================================
# Input Validation Functions
# ============================================================================

def _validate_spectrum_and_freqs(
    spectrum: np.ndarray,
    freqs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate spectrum and frequency arrays for consistency.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum array (complex or real)
    freqs : np.ndarray
        Frequency bins [Hz]

    Returns
    -------
    spectrum, freqs : np.ndarray
        Validated arrays

    Raises
    ------
    ValueError
        If arrays are invalid or incompatible
    """
    spectrum = np.asarray(spectrum)
    freqs = np.asarray(freqs)

    # Check for empty arrays
    if spectrum.size == 0:
        raise ValueError("Spectrum array is empty")

    if freqs.size == 0:
        raise ValueError("Frequency array is empty")

    # Check frequency array is 1D
    if freqs.ndim != 1:
        raise ValueError(
            f"Frequency array must be 1D. Got shape {freqs.shape}."
        )

    # Check spectrum dimensions
    if spectrum.ndim == 1:
        # Single spectrum: reshape to (1, n_freqs)
        spectrum = spectrum.reshape(1, -1)
    elif spectrum.ndim != 2:
        raise ValueError(
            f"Spectrum must be 1D or 2D array. Got shape {spectrum.shape}."
        )

    n_frames, n_freqs_spectrum = spectrum.shape

    # Check that frequency array matches spectrum size
    if len(freqs) != n_freqs_spectrum:
        raise ValueError(
            f"Frequency array length ({len(freqs)}) must match spectrum "
            f"frequency dimension ({n_freqs_spectrum})."
        )

    # Check for NaN/Inf in spectrum
    if np.any(np.isnan(spectrum)):
        raise ValueError("Spectrum contains NaN values")

    if np.any(np.isinf(spectrum)):
        raise ValueError("Spectrum contains Inf values")

    # Check for NaN/Inf in freqs
    if np.any(np.isnan(freqs)):
        raise ValueError("Frequency array contains NaN values")

    if np.any(np.isinf(freqs)):
        raise ValueError("Frequency array contains Inf values")

    # Check frequency array is monotonic increasing
    if not np.all(np.diff(freqs) > 0):
        warnings.warn(
            "Frequency array is not monotonically increasing. "
            "This may cause unexpected filtering behavior.",
            UserWarning
        )

    # Check for negative frequencies
    if np.any(freqs < 0):
        raise ValueError(
            f"Negative frequencies detected. Freqs should be >= 0. "
            f"Min freq: {freqs.min():.2f} Hz."
        )

    return spectrum, freqs


def _validate_frequency_band(band_hz: Tuple[float, float], freqs: np.ndarray) -> None:
    """
    Validate frequency band parameters.

    Parameters
    ----------
    band_hz : tuple of (float, float)
        Frequency band (f_low, f_high) in Hz
    freqs : np.ndarray
        Available frequency bins [Hz]

    Raises
    ------
    ValueError
        If band is invalid
    """
    f_low, f_high = band_hz

    if f_low < 0 or f_high < 0:
        raise ValueError(
            f"Frequency band must be positive. Got ({f_low}, {f_high}) Hz."
        )

    if f_low >= f_high:
        raise ValueError(
            f"Lower frequency must be < higher frequency. Got ({f_low}, {f_high}) Hz."
        )

    # Warn if band is outside available frequency range
    freq_min, freq_max = freqs[0], freqs[-1]

    if f_high < freq_min or f_low > freq_max:
        warnings.warn(
            f"Frequency band ({f_low}-{f_high} Hz) is completely outside "
            f"available frequency range ({freq_min:.1f}-{freq_max:.1f} Hz). "
            f"Filter will have no effect.",
            UserWarning
        )

    if f_low < freq_min:
        warnings.warn(
            f"Lower frequency ({f_low} Hz) is below minimum available "
            f"frequency ({freq_min:.1f} Hz). Band will be clipped.",
            UserWarning
        )

    if f_high > freq_max:
        warnings.warn(
            f"Higher frequency ({f_high} Hz) exceeds maximum available "
            f"frequency ({freq_max:.1f} Hz). Band will be clipped.",
            UserWarning
        )


def _validate_harmonic_parameters(
    f0: float,
    n_harmonics: int,
    bw_hz: float,
    freqs: np.ndarray
) -> None:
    """
    Validate harmonic filtering parameters.

    Parameters
    ----------
    f0 : float
        Fundamental frequency [Hz]
    n_harmonics : int
        Number of harmonics to keep
    bw_hz : float
        Bandwidth per harmonic [Hz]
    freqs : np.ndarray
        Available frequency bins [Hz]

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if f0 <= 0:
        raise ValueError(f"Fundamental frequency must be positive. Got {f0} Hz.")

    if f0 < 10:
        warnings.warn(
            f"Very low fundamental frequency ({f0} Hz). "
            f"Typical drone propeller frequencies are 50-250 Hz.",
            UserWarning
        )

    if f0 > 500:
        warnings.warn(
            f"Very high fundamental frequency ({f0} Hz). "
            f"Typical drone propeller frequencies are 50-250 Hz. "
            f"This may be a motor frequency or harmonic.",
            UserWarning
        )

    if n_harmonics <= 0:
        raise ValueError(
            f"Number of harmonics must be positive. Got {n_harmonics}."
        )

    if n_harmonics > 20:
        warnings.warn(
            f"Very high number of harmonics ({n_harmonics}). "
            f"Typical drones have 5-10 detectable harmonics above noise floor.",
            UserWarning
        )

    if bw_hz <= 0:
        raise ValueError(f"Bandwidth must be positive. Got {bw_hz} Hz.")

    if bw_hz < 5:
        warnings.warn(
            f"Very narrow bandwidth ({bw_hz} Hz per harmonic). "
            f"This may miss frequency variations due to RPM changes or Doppler effect. "
            f"Consider 10-50 Hz bandwidth.",
            UserWarning
        )

    if bw_hz > 200:
        warnings.warn(
            f"Very wide bandwidth ({bw_hz} Hz per harmonic). "
            f"This may include excessive noise. Consider narrower bandwidth.",
            UserWarning
        )

    # Check if highest harmonic is within frequency range
    highest_harmonic_freq = f0 * n_harmonics
    freq_max = freqs[-1]

    if highest_harmonic_freq > freq_max:
        warnings.warn(
            f"Highest harmonic ({n_harmonics}*{f0:.1f} = {highest_harmonic_freq:.1f} Hz) "
            f"exceeds maximum available frequency ({freq_max:.1f} Hz). "
            f"Some harmonics will be outside the spectrum.",
            UserWarning
        )

    # Check frequency resolution
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 0
    if bw_hz < 2 * freq_resolution:
        warnings.warn(
            f"Bandwidth ({bw_hz} Hz) is too small relative to frequency "
            f"resolution ({freq_resolution:.2f} Hz). Each harmonic band may "
            f"contain very few bins. Consider increasing FFT size or bandwidth.",
            UserWarning
        )


# ============================================================================
# Coarse Band-Pass Filtering
# ============================================================================

def make_bandpass_mask(
    freqs: np.ndarray,
    band_hz: Tuple[float, float]
) -> np.ndarray:
    """
    Create a boolean mask selecting frequencies within a specified band.

    This is used for coarse frequency filtering to remove out-of-band noise
    while retaining the frequency range where drone acoustic signatures exist.

    Parameters
    ----------
    freqs : np.ndarray
        1D array of frequency bins [Hz], shape [n_freqs].
        Typically obtained from FFT (e.g., np.fft.rfftfreq).
    band_hz : tuple of (float, float)
        Frequency band (f_low, f_high) in Hz.
        Only frequencies within [f_low, f_high] will be True in the mask.

    Returns
    -------
    mask : np.ndarray
        Boolean array, shape [n_freqs].
        True for bins inside the band, False otherwise.

    Examples
    --------
    >>> # Create frequency bins
    >>> fs = 16000
    >>> n_fft = 1024
    >>> freqs = np.fft.rfftfreq(n_fft, 1/fs)
    >>>
    >>> # Drone detection band: 100-5000 Hz
    >>> mask = make_bandpass_mask(freqs, (100, 5000))
    >>> print(f"Selected {mask.sum()} of {len(mask)} bins")

    Notes
    -----
    - Typical drone detection band: 100-5000 Hz or 50-8000 Hz
    - Lower bound removes DC, very low frequency wind noise, building vibrations
    - Upper bound removes electronic noise, ultrasonic interference
    - For more selective filtering, use harmonic selection after band-pass
    """
    freqs = np.asarray(freqs)
    _validate_frequency_band(band_hz, freqs)

    f_low, f_high = band_hz

    # Create boolean mask
    mask = (freqs >= f_low) & (freqs <= f_high)

    # Warn if no bins selected
    if not np.any(mask):
        warnings.warn(
            f"No frequency bins found in band ({f_low}, {f_high}) Hz. "
            f"Frequency range is {freqs[0]:.1f} to {freqs[-1]:.1f} Hz. "
            f"Filter will remove all energy.",
            UserWarning
        )

    return mask


def apply_bandpass(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    band_hz: Tuple[float, float]
) -> np.ndarray:
    """
    Apply coarse band-pass filter in the frequency domain by zeroing out-of-band bins.

    This is a simple but effective rectangular filter that completely removes
    frequencies outside the specified band. Useful for initial noise reduction
    before more sophisticated processing.

    Parameters
    ----------
    spectrum : np.ndarray
        Complex spectrum array, shape [n_frames, n_freqs].
        Obtained from FFT (e.g., from fft.py compute_fft_per_frame()).
        Can also be 1D array [n_freqs] for single frame.
    freqs : np.ndarray
        1D array of frequency bins [Hz], shape [n_freqs].
    band_hz : tuple of (float, float)
        Frequency band (f_low, f_high) in Hz.

    Returns
    -------
    spectrum_bp : np.ndarray
        Band-pass filtered spectrum, same shape as input.
        Bins outside [f_low, f_high] are set to zero.

    Examples
    --------
    >>> # Get spectrum from FFT
    >>> from fft import compute_fft_per_frame
    >>> freqs, spectrum, _, _ = compute_fft_per_frame(frames, fs)
    >>>
    >>> # Apply drone detection band-pass
    >>> spectrum_filtered = apply_bandpass(spectrum, freqs, (100, 5000))
    >>>
    >>> # Check energy reduction
    >>> energy_before = np.sum(np.abs(spectrum)**2)
    >>> energy_after = np.sum(np.abs(spectrum_filtered)**2)
    >>> print(f"Energy retained: {energy_after/energy_before*100:.1f}%")

    Notes
    -----
    - This is a rectangular (brick-wall) filter with sharp cutoffs
    - No transition band, which may cause some spectral leakage
    - For smoother filtering, consider designing FIR/IIR filters in time domain
    - Filter operates on complex spectrum, preserving phase information
    - Useful for GCC-PHAT (needs phase) and inverse FFT operations
    """
    spectrum, freqs = _validate_spectrum_and_freqs(spectrum, freqs)
    mask = make_bandpass_mask(freqs, band_hz)

    # Copy spectrum to avoid modifying original
    spectrum_bp = spectrum.copy()

    # Zero out bins outside the band
    # mask is 1D [n_freqs], spectrum is 2D [n_frames, n_freqs]
    # Broadcasting: mask applies to all frames
    spectrum_bp[:, ~mask] = 0.0

    return spectrum_bp


# ============================================================================
# Harmonic Selection Filtering
# ============================================================================

def make_harmonic_mask(
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int,
    bw_hz: float,
    include_fundamental: bool = True
) -> np.ndarray:
    """
    Create a boolean mask selecting only frequency bins around harmonics of f0.

    This is the key technique for isolating drone propeller harmonic signatures.
    By focusing only on harmonics and rejecting everything else, we dramatically
    improve SNR for drone detection.

    Principle:
    - Drone propellers produce tones at f0, 2*f0, 3*f0, ..., n*f0
    - Each harmonic has some width due to modulation, Doppler, RPM variation
    - We create narrow bands around each harmonic: [k*f0 - bw/2, k*f0 + bw/2]

    Parameters
    ----------
    freqs : np.ndarray
        1D array of frequency bins [Hz], shape [n_freqs].
    f0 : float
        Fundamental frequency in Hz (propeller blade-pass frequency).
        This is the lowest frequency component of the periodic signal.
        For drones, typically 50-250 Hz depending on propeller size and RPM.
    n_harmonics : int
        Number of harmonics to keep (1..N).
        Example: n_harmonics=5 keeps harmonics at f0, 2f0, 3f0, 4f0, 5f0.
        Typical: 5-10 harmonics for drone detection.
    bw_hz : float
        Total bandwidth around each harmonic [Hz].
        Each harmonic band is [k*f0 - bw/2, k*f0 + bw/2].
        Accounts for:
        - RPM variations: ±5-10 Hz
        - Doppler effect: up to ±20 Hz for fast-moving drones
        - Frequency modulation from propeller load variations
        Typical: 20-50 Hz per harmonic
    include_fundamental : bool, default=True
        If True, include the fundamental (k=1, frequency=f0).
        If False, start from 2nd harmonic (k=2, frequency=2*f0).
        Set False if fundamental is weak or masked by low-frequency noise.

    Returns
    -------
    mask : np.ndarray
        Boolean array, shape [n_freqs].
        True = bin is inside one of the harmonic bands.
        False = bin is outside all harmonic bands (will be zeroed).

    Examples
    --------
    >>> # Setup
    >>> fs = 16000
    >>> n_fft = 1024
    >>> freqs = np.fft.rfftfreq(n_fft, 1/fs)
    >>>
    >>> # Drone with 150 Hz propeller frequency
    >>> f0 = 150  # Hz
    >>> mask = make_harmonic_mask(freqs, f0, n_harmonics=5, bw_hz=30)
    >>>
    >>> # Visualize harmonic bands
    >>> print(f"Harmonic 1: {f0-15:.1f} to {f0+15:.1f} Hz")
    >>> print(f"Harmonic 2: {2*f0-15:.1f} to {2*f0+15:.1f} Hz")
    >>> print(f"Total bins selected: {mask.sum()} of {len(mask)}")

    >>> # Exclude fundamental (only upper harmonics)
    >>> mask_no_f0 = make_harmonic_mask(
    ...     freqs, f0, n_harmonics=5, bw_hz=30, include_fundamental=False
    ... )
    >>> print(f"Without fundamental: {mask_no_f0.sum()} bins")

    Notes
    -----
    - Fundamental frequency f0 can be estimated from peak detection in spectrum
    - Bandwidth should be wide enough to handle RPM variations but narrow
      enough to reject noise between harmonics
    - Higher harmonics (5th, 6th, ...) are typically weaker in amplitude
    - Some drones have multiple propellers with slightly different f0, creating
      interleaved harmonics - consider wider bandwidth or multiple f0 values
    """
    freqs = np.asarray(freqs)
    _validate_harmonic_parameters(f0, n_harmonics, bw_hz, freqs)

    # Initialize mask to all False
    mask = np.zeros_like(freqs, dtype=bool)

    # Determine starting harmonic index
    start_k = 1 if include_fundamental else 2

    # Half bandwidth (symmetric around harmonic center)
    half_bw = bw_hz / 2.0

    # Create bands around each harmonic
    for k in range(start_k, n_harmonics + 1):
        center_freq = k * f0
        lower_bound = center_freq - half_bw
        upper_bound = center_freq + half_bw

        # Find bins within this harmonic band
        band_k = (freqs >= lower_bound) & (freqs <= upper_bound)

        # Combine with overall mask (logical OR)
        mask |= band_k

    # Warn if no bins selected
    if not np.any(mask):
        warnings.warn(
            f"No frequency bins found for any harmonics. "
            f"f0={f0} Hz, n_harmonics={n_harmonics}, bw={bw_hz} Hz. "
            f"Check if harmonics fall within frequency range "
            f"({freqs[0]:.1f} to {freqs[-1]:.1f} Hz).",
            UserWarning
        )

    return mask


def apply_harmonic_selection(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int,
    bw_hz: float,
    include_fundamental: bool = True
) -> np.ndarray:
    """
    Keep only frequency bins around harmonics of f0, zero-out everything else.

    This is the core harmonic filtering function for drone detection. It isolates
    the periodic harmonic structure of drone propeller noise while rejecting:
    - Broadband noise (wind, electronics)
    - Non-harmonic sounds (birds, speech, traffic)
    - Harmonics of other sources with different fundamental frequencies

    Process:
    1. Create harmonic mask (narrow bands around f0, 2f0, 3f0, ...)
    2. Zero out all spectrum bins not in the mask
    3. Return filtered spectrum with only harmonic content

    Parameters
    ----------
    spectrum : np.ndarray
        Complex spectrum array, shape [n_frames, n_freqs].
        Can also be 1D [n_freqs] for single frame.
    freqs : np.ndarray
        1D array of frequency bins [Hz], shape [n_freqs].
    f0 : float
        Fundamental frequency [Hz] (propeller blade-pass frequency).
    n_harmonics : int
        Number of harmonics to keep (1..N).
    bw_hz : float
        Bandwidth per harmonic [Hz].
    include_fundamental : bool, default=True
        If False, exclude fundamental and start from 2nd harmonic.

    Returns
    -------
    spectrum_harm : np.ndarray
        Harmonically filtered spectrum, same shape as input.
        Only harmonic bands are preserved, rest is zeroed.

    Examples
    --------
    >>> # Get spectrum from FFT
    >>> from fft import compute_fft_per_frame
    >>> freqs, spectrum, magnitude, _ = compute_fft_per_frame(frames, fs)
    >>>
    >>> # Estimate fundamental frequency (example: from peak detection)
    >>> from fft import find_spectral_peaks
    >>> peak_freqs, _ = find_spectral_peaks(magnitude, freqs, n_peaks=1)
    >>> f0_estimated = peak_freqs[0, 0]  # First frame, first peak
    >>>
    >>> # Apply harmonic filtering
    >>> spectrum_clean = apply_harmonic_selection(
    ...     spectrum, freqs,
    ...     f0=f0_estimated,
    ...     n_harmonics=7,
    ...     bw_hz=40,
    ...     include_fundamental=True
    ... )
    >>>
    >>> # Calculate SNR improvement
    >>> noise_power = np.sum(np.abs(spectrum - spectrum_clean)**2)
    >>> signal_power = np.sum(np.abs(spectrum_clean)**2)
    >>> snr_improvement_db = 10 * np.log10(signal_power / noise_power)
    >>> print(f"SNR improvement: {snr_improvement_db:.1f} dB")

    Notes
    -----
    - Fundamental frequency f0 must be known or estimated
    - For unknown f0, use peak detection or harmonic product spectrum
    - This filter is very aggressive - removes everything except harmonics
    - Works best when drone is dominant sound source
    - May fail if f0 is not correctly estimated (off by >bw_hz/2)
    - Consider adaptive f0 tracking for moving drones with varying RPM
    """
    spectrum, freqs = _validate_spectrum_and_freqs(spectrum, freqs)
    mask = make_harmonic_mask(freqs, f0, n_harmonics, bw_hz, include_fundamental)

    # Copy spectrum to avoid modifying original
    spectrum_harm = spectrum.copy()

    # Zero out bins outside harmonic bands
    spectrum_harm[:, ~mask] = 0.0

    return spectrum_harm


# ============================================================================
# Combined Filtering Function
# ============================================================================

def frequency_harmonic_filter(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    coarse_band_hz: Tuple[float, float],
    f0: Optional[float] = None,
    n_harmonics: int = 0,
    harmonic_bw_hz: float = 0.0,
    include_fundamental: bool = True,
) -> np.ndarray:
    """
    Combined frequency and harmonic filtering for drone detection.

    This is the main filtering function that combines two stages:
    1. Coarse band-pass filtering: Removes out-of-band noise
    2. Harmonic selection filtering: Isolates drone harmonic structure

    The two-stage approach provides:
    - Broad initial filtering (stage 1) for computational efficiency
    - Fine-tuned harmonic filtering (stage 2) for maximum selectivity

    Pipeline:
    Input spectrum → Band-pass filter → Harmonic filter → Output spectrum

    Parameters
    ----------
    spectrum : np.ndarray
        Complex spectrum array, shape [n_frames, n_freqs].
        Can also be 1D [n_freqs] for single frame.
        Obtained from FFT.
    freqs : np.ndarray
        1D array of frequency bins [Hz], shape [n_freqs].
    coarse_band_hz : tuple of (float, float)
        Coarse band-pass filter limits (f_low, f_high) in Hz.
        Example: (100, 5000) for typical drone detection.
        This is always applied (stage 1).
    f0 : float or None, default=None
        Fundamental frequency [Hz] for harmonic filtering.
        If None, harmonic filtering is skipped (only band-pass applied).
        If provided, must be positive.
    n_harmonics : int, default=0
        Number of harmonics to keep in harmonic filter (stage 2).
        If 0 or f0 is None, harmonic filtering is skipped.
    harmonic_bw_hz : float, default=0.0
        Bandwidth per harmonic [Hz] for harmonic filter.
        If 0 or f0 is None, harmonic filtering is skipped.
    include_fundamental : bool, default=True
        Whether to include fundamental frequency in harmonic filter.
        Only relevant if f0 is provided and n_harmonics > 0.

    Returns
    -------
    spectrum_filt : np.ndarray
        Filtered spectrum, same shape as input.
        If only band-pass: bins outside coarse_band_hz are zeroed.
        If harmonic filter also applied: only harmonic bands within
        coarse_band_hz are preserved.

    Examples
    --------
    >>> # Example 1: Band-pass only (f0 unknown)
    >>> spectrum_bp = frequency_harmonic_filter(
    ...     spectrum, freqs,
    ...     coarse_band_hz=(100, 5000)
    ... )

    >>> # Example 2: Band-pass + harmonic filter (f0 known)
    >>> spectrum_harm = frequency_harmonic_filter(
    ...     spectrum, freqs,
    ...     coarse_band_hz=(100, 5000),
    ...     f0=150,
    ...     n_harmonics=7,
    ...     harmonic_bw_hz=40
    ... )

    >>> # Example 3: Exclude fundamental (remove low-freq noise)
    >>> spectrum_no_f0 = frequency_harmonic_filter(
    ...     spectrum, freqs,
    ...     coarse_band_hz=(100, 5000),
    ...     f0=120,
    ...     n_harmonics=7,
    ...     harmonic_bw_hz=30,
    ...     include_fundamental=False
    ... )

    Notes
    -----
    - Always apply coarse band-pass first (fast, removes bulk of noise)
    - Harmonic filtering is optional and requires known/estimated f0
    - For unknown f0, use band-pass only in first pass, then estimate f0
      from filtered spectrum, then apply harmonic filter in second pass
    - Harmonic filtering provides significant SNR gain (10-20 dB typical)
    - Filtered spectrum can be used for inverse FFT, feature extraction, or GCC-PHAT

    Typical Usage Pipeline:
    1. Acquire audio → Frame and window → FFT
    2. Apply band-pass filter (this function, f0=None)
    3. Estimate f0 from filtered spectrum (peak detection)
    4. Apply band-pass + harmonic filter (this function, with f0)
    5. Extract features or perform detection
    """
    spectrum, freqs = _validate_spectrum_and_freqs(spectrum, freqs)

    # ========================================================================
    # Step 1: Coarse band-pass filter (always applied)
    # ========================================================================

    spec_bp = apply_bandpass(spectrum, freqs, coarse_band_hz)

    # ========================================================================
    # Step 2: Harmonic selection filter (optional, if f0 provided)
    # ========================================================================

    # Check if harmonic filtering should be applied
    apply_harmonic = (
        f0 is not None and
        n_harmonics > 0 and
        harmonic_bw_hz > 0
    )

    if apply_harmonic:
        spec_filt = apply_harmonic_selection(
            spec_bp,
            freqs,
            f0=f0,
            n_harmonics=n_harmonics,
            bw_hz=harmonic_bw_hz,
            include_fundamental=include_fundamental,
        )
    else:
        # No harmonic filtering, return band-pass result
        spec_filt = spec_bp

    return spec_filt


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_snr_improvement(
    spectrum_original: np.ndarray,
    spectrum_filtered: np.ndarray,
    harmonic_mask: Optional[np.ndarray] = None
) -> float:
    """
    Estimate SNR improvement from filtering (dB).

    Compares signal power in harmonic bands vs noise power in rejected bands.

    Parameters
    ----------
    spectrum_original : np.ndarray
        Original spectrum before filtering
    spectrum_filtered : np.ndarray
        Filtered spectrum
    harmonic_mask : np.ndarray, optional
        Boolean mask indicating harmonic bins (True = signal, False = noise).
        If None, estimates from non-zero bins in filtered spectrum.

    Returns
    -------
    snr_improvement_db : float
        SNR improvement in dB

    Examples
    --------
    >>> # Apply filter and estimate improvement
    >>> mask = make_harmonic_mask(freqs, f0=150, n_harmonics=5, bw_hz=30)
    >>> spectrum_filtered = apply_harmonic_selection(spectrum, freqs, f0=150, n_harmonics=5, bw_hz=30)
    >>> snr_gain = estimate_snr_improvement(spectrum, spectrum_filtered, mask)
    >>> print(f"SNR improvement: {snr_gain:.1f} dB")

    Notes
    -----
    - This is an approximation assuming noise is evenly distributed
    - Actual SNR depends on specific noise characteristics
    - Higher values indicate more effective filtering
    """
    # If no mask provided, infer from filtered spectrum
    if harmonic_mask is None:
        # Bins that are non-zero in filtered spectrum are signal
        harmonic_mask = np.any(np.abs(spectrum_filtered) > 1e-10, axis=0)

    # Calculate power in signal bands (harmonics)
    signal_power = np.sum(np.abs(spectrum_filtered) ** 2)

    # Calculate power in noise bands (rejected by filter)
    noise_spectrum = spectrum_original.copy()
    noise_spectrum[:, harmonic_mask] = 0  # Remove signal bands
    noise_power = np.sum(np.abs(noise_spectrum) ** 2)

    # Avoid division by zero
    if noise_power < 1e-20:
        return float('inf')  # Perfect filtering (no noise)

    # SNR improvement in dB
    snr_improvement_db = 10 * np.log10(signal_power / noise_power + 1e-20)

    return snr_improvement_db


def get_harmonic_energies(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int,
    bw_hz: float
) -> np.ndarray:
    """
    Extract energy of each harmonic from spectrum.

    Useful for analyzing harmonic structure and detecting missing/weak harmonics.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum array [n_frames, n_freqs]
    freqs : np.ndarray
        Frequency bins [Hz]
    f0 : float
        Fundamental frequency [Hz]
    n_harmonics : int
        Number of harmonics to analyze
    bw_hz : float
        Bandwidth around each harmonic [Hz]

    Returns
    -------
    harmonic_energies : np.ndarray
        Energy of each harmonic, shape [n_frames, n_harmonics]

    Examples
    --------
    >>> energies = get_harmonic_energies(spectrum, freqs, f0=150, n_harmonics=5, bw_hz=30)
    >>> print("Harmonic energies (average across frames):")
    >>> for k in range(5):
    ...     print(f"  Harmonic {k+1}: {energies[:, k].mean():.2f}")
    """
    spectrum, freqs = _validate_spectrum_and_freqs(spectrum, freqs)
    n_frames = spectrum.shape[0]

    harmonic_energies = np.zeros((n_frames, n_harmonics))

    half_bw = bw_hz / 2.0

    for k in range(1, n_harmonics + 1):
        center = k * f0
        mask = (freqs >= center - half_bw) & (freqs <= center + half_bw)

        # Sum energy in this harmonic band
        harmonic_energies[:, k-1] = np.sum(np.abs(spectrum[:, mask]) ** 2, axis=1)

    return harmonic_energies


# ============================================================================
# Demo / Test Code
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of frequency and harmonic filtering for drone detection.

    This demo:
    1. Generates synthetic drone signal with harmonics plus noise
    2. Computes spectrum
    3. Applies band-pass and harmonic filters
    4. Visualizes filtering effects
    5. Estimates SNR improvement
    """
    import matplotlib.pyplot as plt
    import sys

    # Import required modules
    try:
        from framing_windowing import frame_and_window
        from fft import compute_fft_per_frame
    except ImportError as e:
        print(f"Error: Cannot import required modules: {e}")
        print("Make sure framing_windowing.py and fft.py are in the same directory.")
        sys.exit(1)

    print("=" * 70)
    print("Acoustic Drone Detection: Harmonic Filtering Demo")
    print("=" * 70)

    # ========================================================================
    # Generate synthetic signal
    # ========================================================================

    fs = 16000  # 16 kHz
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Drone parameters
    f0 = 150  # Hz (propeller fundamental)
    n_harm = 5  # Number of harmonics

    # Generate clean drone signal (harmonics)
    drone_signal = np.zeros_like(t)
    harmonic_amplitudes = [0.5, 0.3, 0.2, 0.1, 0.05]  # Decreasing with harmonic number
    for k in range(1, n_harm + 1):
        drone_signal += harmonic_amplitudes[k-1] * np.sin(2 * np.pi * k * f0 * t)

    # Add broadband noise
    noise_level = 0.3
    noise = noise_level * np.random.randn(len(t))

    # Add low-frequency interference (wind-like)
    interference = 0.4 * np.sin(2 * np.pi * 10 * t)  # 10 Hz

    # Combined signal
    signal = drone_signal + noise + interference

    print(f"\nSignal Generation:")
    print(f"  Duration: {duration} s")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Drone fundamental: {f0} Hz")
    print(f"  Number of harmonics: {n_harm}")
    print(f"  Noise level: {noise_level}")

    # ========================================================================
    # Frame, window, and compute FFT
    # ========================================================================

    frames, frame_times = frame_and_window(
        signal, fs,
        frame_length_ms=64.0,
        hop_length_ms=32.0,
        window_type='hann'
    )

    freqs, spectrum, magnitude, magnitude_db = compute_fft_per_frame(
        frames, fs,
        nfft=1024
    )

    print(f"\nSpectrum:")
    print(f"  Frames: {spectrum.shape[0]}")
    print(f"  Frequency bins: {spectrum.shape[1]}")
    print(f"  Frequency range: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")

    # ========================================================================
    # Apply filters
    # ========================================================================

    # Band-pass only
    spectrum_bp = frequency_harmonic_filter(
        spectrum, freqs,
        coarse_band_hz=(50, 1000)
    )

    # Band-pass + harmonic filter
    spectrum_harm = frequency_harmonic_filter(
        spectrum, freqs,
        coarse_band_hz=(50, 1000),
        f0=f0,
        n_harmonics=n_harm,
        harmonic_bw_hz=40,
        include_fundamental=True
    )

    # Calculate magnitudes
    mag_bp = np.abs(spectrum_bp)
    mag_harm = np.abs(spectrum_harm)

    # Estimate SNR improvement
    harmonic_mask = make_harmonic_mask(freqs, f0, n_harm, 40)
    snr_gain = estimate_snr_improvement(spectrum, spectrum_harm, harmonic_mask)

    print(f"\nFiltering Results:")
    print(f"  Band-pass: 50-1000 Hz")
    print(f"  Harmonic filter: {n_harm} harmonics, 40 Hz bandwidth each")
    print(f"  SNR improvement: {snr_gain:.1f} dB")

    # ========================================================================
    # Visualization
    # ========================================================================

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    frame_idx = len(frames) // 2  # Middle frame

    # Plot 1: Original spectrum
    axes[0].plot(freqs, magnitude_db[frame_idx], linewidth=1, label='Original')
    axes[0].set_title('Original Spectrum (with noise)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_xlim([0, 1000])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Mark true harmonics
    for k in range(1, n_harm + 1):
        axes[0].axvline(k * f0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 2: Band-pass filtered
    mag_bp_db = 20 * np.log10(mag_bp[frame_idx] + 1e-12)
    axes[1].plot(freqs, mag_bp_db, linewidth=1, label='Band-pass filtered', color='orange')
    axes[1].set_title('Band-Pass Filtered (50-1000 Hz)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_xlim([0, 1000])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Mark harmonics
    for k in range(1, n_harm + 1):
        axes[1].axvline(k * f0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 3: Harmonic filtered
    mag_harm_db = 20 * np.log10(mag_harm[frame_idx] + 1e-12)
    axes[2].plot(freqs, mag_harm_db, linewidth=1, label='Harmonic filtered', color='green')
    axes[2].set_title(f'Harmonic Filtered ({n_harm} harmonics, SNR gain: {snr_gain:.1f} dB)',
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude (dB)')
    axes[2].set_xlim([0, 1000])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Mark harmonics with bandwidth
    bw = 40
    for k in range(1, n_harm + 1):
        center = k * f0
        axes[2].axvline(center, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[2].axvspan(center - bw/2, center + bw/2, alpha=0.2, color='green')

    plt.tight_layout()

    output_path = '/mnt/d/edth/harmonic_filter_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
