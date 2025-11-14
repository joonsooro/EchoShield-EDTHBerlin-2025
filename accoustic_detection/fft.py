"""
fft.py

FFT Analysis for Acoustic Drone Detection
==========================================

This module provides robust Fast Fourier Transform (FFT) operations for converting
time-domain acoustic signals into frequency-domain representations. This is essential
for drone detection because:

1. **Frequency Signatures**: Drones produce distinctive frequency patterns:
   - Propeller blade-pass frequency (fundamental): 50-250 Hz
   - Motor harmonics: Multiples of fundamental frequency
   - Broadband motor noise: 1-5 kHz range

2. **Spectral Analysis**: FFT reveals these frequency components that are difficult
   to detect in the time domain

3. **Feature Extraction**: Frequency-domain features (spectral peaks, energy distribution)
   are more discriminative for drone classification

Key Operations:
- Per-frame FFT computation (from windowed frames)
- Magnitude and phase extraction
- dB conversion for dynamic range compression
- Frequency bin generation for proper interpretation

References:
- Drones typically exhibit strong harmonics at multiples of blade-pass frequency
- Most drone acoustic energy is concentrated below 5 kHz
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


# ============================================================================
# Input Validation Functions
# ============================================================================

def _validate_frames(frames: np.ndarray) -> np.ndarray:
    """
    Validate input frames array.

    Parameters
    ----------
    frames : np.ndarray
        Input frames array

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
    if frames.ndim == 1:
        # Single frame: reshape to (1, frame_length)
        frames = frames.reshape(1, -1)
    elif frames.ndim != 2:
        raise ValueError(
            f"Frames must be 1D or 2D array. Got shape {frames.shape} with {frames.ndim} dimensions."
        )

    # Check for empty data
    if frames.size == 0:
        raise ValueError("Frames array is empty")

    n_frames, frame_length = frames.shape

    if frame_length < 2:
        raise ValueError(
            f"Frame length must be at least 2 samples for FFT. Got {frame_length}."
        )

    # Check for NaN/Inf values
    if np.any(np.isnan(frames)):
        raise ValueError(
            "Frames contain NaN values. Check your framing/windowing step."
        )

    if np.any(np.isinf(frames)):
        raise ValueError(
            "Frames contain Inf values. Check for numerical overflow in preprocessing."
        )

    return frames


def _validate_sampling_rate(fs: Union[int, float]) -> float:
    """
    Validate sampling rate.

    Parameters
    ----------
    fs : int or float
        Sampling rate in Hz

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

    if fs < 1000:
        warnings.warn(
            f"Very low sampling rate ({fs} Hz). "
            f"Drone detection typically requires ≥8 kHz to capture propeller harmonics.",
            UserWarning
        )

    return fs


def _validate_nfft(nfft: Optional[int], frame_length: int) -> int:
    """
    Validate and set FFT size.

    Parameters
    ----------
    nfft : int or None
        Desired FFT size (must be >= frame_length)
    frame_length : int
        Length of input frames

    Returns
    -------
    nfft : int
        Validated FFT size

    Raises
    ------
    ValueError
        If nfft is invalid
    """
    if nfft is None:
        return frame_length

    nfft = int(nfft)

    if nfft < frame_length:
        raise ValueError(
            f"FFT size ({nfft}) must be >= frame length ({frame_length}). "
            f"Use zero-padding to increase FFT size."
        )

    if nfft <= 0:
        raise ValueError(f"FFT size must be positive. Got {nfft}.")

    # Warn if not power of 2 (slower FFT)
    if nfft > 0 and (nfft & (nfft - 1)) != 0:
        next_pow2 = 2 ** np.ceil(np.log2(nfft)).astype(int)
        warnings.warn(
            f"FFT size ({nfft}) is not a power of 2. "
            f"Consider using {next_pow2} for faster computation.",
            UserWarning
        )

    return nfft


# ============================================================================
# Main FFT Functions
# ============================================================================

def compute_fft_per_frame(
    frames: np.ndarray,
    fs: Union[int, float],
    nfft: Optional[int] = None,
    remove_dc: bool = True,
    window_compensation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FFT for each time frame, converting to frequency domain.

    This is a critical step in acoustic drone detection pipelines, transforming
    windowed time-domain frames into frequency-domain representations where
    drone-specific spectral signatures can be analyzed.

    Process Flow:
    1. Validate input frames and parameters
    2. Optionally remove DC offset (recommended for acoustic analysis)
    3. Compute real FFT (positive frequencies only, more efficient)
    4. Extract magnitude and phase information
    5. Convert to dB scale for better visualization and dynamic range
    6. Generate frequency axis for proper interpretation

    Parameters
    ----------
    frames : np.ndarray
        Array of shape [n_frames, frame_length] containing windowed frames
        from your framing+windowing step. Can also be 1D array for single frame.
        Frames should already be windowed (Hann, Hamming, etc.) to reduce
        spectral leakage.
    fs : int or float
        Sampling rate in Hz (e.g., 16000 for 16 kHz, 44100 for 44.1 kHz).
        Must match the sampling rate used during audio acquisition.
    nfft : int, optional
        FFT size (number of frequency bins). If None, uses frame_length.
        If nfft > frame_length, frames are zero-padded, which:
        - Increases frequency resolution (interpolation)
        - Does NOT increase actual spectral information
        - Useful for visualization and peak detection
        Power-of-2 values (256, 512, 1024, etc.) are fastest.
    remove_dc : bool, default=True
        If True, remove DC offset (mean) from each frame before FFT.
        Recommended for acoustic signals to:
        - Remove microphone/sensor bias
        - Prevent DC bin from dominating the spectrum
        - Improve numerical stability
    window_compensation : bool, default=False
        If True, compensate for window function amplitude loss.
        Multiplies spectrum by 2/sum(window) to restore original amplitudes.
        Only useful if you need absolute power measurements.

    Returns
    -------
    freqs : np.ndarray
        1D array of frequency bins [Hz], shape [n_freqs].
        For real signals (rfft), contains only positive frequencies: [0, fs/2].
        Same for all frames, so returned as 1D array.
        Example: For fs=16000 Hz, frame_length=512, freqs spans 0 to 8000 Hz.
    spectrum : np.ndarray
        Complex array [n_frames, n_freqs] containing full FFT output.
        Contains both magnitude and phase information.
        Use this for:
        - Phase-sensitive operations (GCC-PHAT, beamforming)
        - Inverse FFT reconstruction
        - Cross-spectral analysis
    magnitude : np.ndarray
        Linear magnitude array [n_frames, n_freqs].
        Magnitude = sqrt(real^2 + imag^2) = abs(spectrum).
        Represents spectral energy at each frequency bin.
        Useful for:
        - Peak detection (finding fundamental frequency)
        - Energy distribution analysis
        - Direct visualization (linear scale)
    magnitude_db : np.ndarray
        Magnitude in decibels [n_frames, n_freqs].
        dB = 20 * log10(magnitude + epsilon).
        Benefits:
        - Compresses dynamic range (human hearing is logarithmic)
        - Reveals weak spectral components
        - Standard for audio visualization (spectrograms)
        Range: typically -120 dB (noise floor) to 0 dB (peak)

    Raises
    ------
    ValueError
        If input parameters are invalid or frames contain NaN/Inf

    Examples
    --------
    >>> # Single frame FFT
    >>> import numpy as np
    >>> fs = 16000
    >>> frame = np.random.randn(512)  # Single windowed frame
    >>> freqs, spec, mag, mag_db = compute_fft_per_frame(frame, fs)
    >>> print(f"Frequency bins: {len(freqs)}")
    >>> print(f"Frequency range: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")

    >>> # Multiple frames FFT
    >>> frames = np.random.randn(100, 512)  # 100 frames
    >>> freqs, spec, mag, mag_db = compute_fft_per_frame(frames, fs)
    >>> print(f"Spectrum shape: {spec.shape}")  # (100, 257)
    >>> print(f"Max energy at: {freqs[mag[0].argmax()]:.1f} Hz")  # Frame 0 peak

    >>> # Higher frequency resolution with zero-padding
    >>> freqs, spec, mag, mag_db = compute_fft_per_frame(
    ...     frames, fs, nfft=2048  # 4x zero-padding
    ... )
    >>> print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")

    Notes
    -----
    - For drone detection, focus on frequency range 50-5000 Hz
    - Typical drone propeller fundamentals are 80-200 Hz
    - Look for harmonics at 2f0, 3f0, 4f0, etc.
    - Use magnitude_db for visualization (better dynamic range)
    - Use spectrum (complex) for phase-based algorithms (GCC-PHAT)
    - Higher nfft improves peak detection accuracy but doesn't add information

    Frequency Resolution:
    - Δf = fs / nfft (Hz per bin)
    - Example: fs=16000, nfft=512 → Δf = 31.25 Hz
    - Trade-off: Lower Δf needs longer frames (worse temporal resolution)
    """

    # ========================================================================
    # Step 1: Validate inputs
    # ========================================================================

    frames = _validate_frames(frames)
    fs = _validate_sampling_rate(fs)
    n_frames, frame_length = frames.shape
    nfft = _validate_nfft(nfft, frame_length)

    # ========================================================================
    # Step 2: Optional DC removal
    # ========================================================================

    if remove_dc:
        # Remove mean from each frame (DC offset / bias removal)
        # This is important for acoustic signals to prevent DC bin dominance
        frames = frames - np.mean(frames, axis=1, keepdims=True)

    # ========================================================================
    # Step 3: Compute real FFT (positive frequencies only)
    # ========================================================================

    # Real FFT is more efficient than full FFT for real-valued signals
    # Output shape: [n_frames, n_freqs] where n_freqs = nfft//2 + 1
    # Only computes positive frequencies: [0, fs/2]
    try:
        spectrum = np.fft.rfft(frames, n=nfft, axis=1)
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {e}") from e

    n_freqs = spectrum.shape[1]

    # ========================================================================
    # Step 4: Generate frequency axis
    # ========================================================================

    # Frequency bins for real FFT: from 0 Hz to Nyquist frequency (fs/2)
    # Shape: [n_freqs]
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

    # ========================================================================
    # Step 5: Extract magnitude
    # ========================================================================

    # Linear magnitude: |spectrum| = sqrt(real^2 + imag^2)
    magnitude = np.abs(spectrum)

    # ========================================================================
    # Step 6: Optional window compensation
    # ========================================================================

    if window_compensation:
        # Compensate for window function amplitude loss
        # This is approximate and assumes Hann/Hamming window
        # For exact compensation, you'd need to pass the actual window used
        compensation_factor = 2.0  # Typical for Hann window
        magnitude = magnitude * compensation_factor
        spectrum = spectrum * compensation_factor

    # ========================================================================
    # Step 7: Convert to dB scale
    # ========================================================================

    # Add small epsilon to avoid log(0) = -inf
    # epsilon = 1e-12 represents ~-240 dB (well below noise floor)
    eps = 1e-12

    # dB conversion: dB = 20 * log10(magnitude)
    # Factor of 20 (not 10) because magnitude is amplitude, not power
    magnitude_db = 20.0 * np.log10(magnitude + eps)

    # ========================================================================
    # Step 8: Validate output
    # ========================================================================

    # Sanity check: no NaN or Inf in outputs
    if np.any(np.isnan(magnitude_db)) or np.any(np.isinf(magnitude_db)):
        warnings.warn(
            "Output contains NaN or Inf values. This may indicate numerical issues.",
            UserWarning
        )

    return freqs, spectrum, magnitude, magnitude_db


def compute_power_spectrum(
    frames: np.ndarray,
    fs: Union[int, float],
    nfft: Optional[int] = None,
    remove_dc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power spectral density (PSD) for each frame.

    Power spectrum represents energy distribution across frequencies.
    Useful for:
    - Energy-based drone detection
    - Spectral energy features
    - Comparing energy across frequency bands

    Parameters
    ----------
    frames : np.ndarray
        Array of shape [n_frames, frame_length] containing windowed frames
    fs : int or float
        Sampling rate in Hz
    nfft : int, optional
        FFT size. If None, uses frame_length
    remove_dc : bool, default=True
        Remove DC offset before computing FFT

    Returns
    -------
    freqs : np.ndarray
        1D array of frequency bins [Hz]
    power : np.ndarray
        Power spectrum [n_frames, n_freqs], linear scale
        Power = |spectrum|^2 (magnitude squared)
    power_db : np.ndarray
        Power spectrum in dB [n_frames, n_freqs]
        dB = 10 * log10(power)

    Examples
    --------
    >>> frames = np.random.randn(100, 512)
    >>> fs = 16000
    >>> freqs, power, power_db = compute_power_spectrum(frames, fs)
    >>> print(f"Power spectrum shape: {power.shape}")
    >>> print(f"Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz")

    Notes
    -----
    - Power = magnitude^2 (energy, not amplitude)
    - dB conversion uses factor of 10 (not 20) for power
    - Useful for detecting acoustic energy in specific frequency bands
    """

    # Compute FFT
    freqs, spectrum, magnitude, _ = compute_fft_per_frame(
        frames, fs, nfft=nfft, remove_dc=remove_dc
    )

    # Compute power: |spectrum|^2
    power = magnitude ** 2

    # Convert to dB (factor of 10 for power, not 20)
    eps = 1e-24  # epsilon for power (square of amplitude epsilon)
    power_db = 10.0 * np.log10(power + eps)

    return freqs, power, power_db


# ============================================================================
# Utility Functions
# ============================================================================

def get_frequency_band_energy(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    band_hz: Tuple[float, float],
) -> np.ndarray:
    """
    Calculate total energy within a specific frequency band for each frame.

    Useful for drone detection features:
    - Low-frequency energy (50-200 Hz): Propeller fundamental
    - Mid-frequency energy (200-1000 Hz): Motor harmonics
    - High-frequency energy (1-5 kHz): Broadband motor noise

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrum [n_frames, n_freqs]
    freqs : np.ndarray
        Frequency bins [Hz], shape [n_freqs]
    band_hz : tuple of (float, float)
        Frequency band limits (f_low, f_high) in Hz

    Returns
    -------
    energy : np.ndarray
        Total energy in the band for each frame, shape [n_frames]

    Examples
    --------
    >>> # Calculate propeller fundamental energy (80-200 Hz)
    >>> prop_energy = get_frequency_band_energy(magnitude, freqs, (80, 200))
    >>> print(f"Average propeller energy: {prop_energy.mean():.2f}")

    >>> # Calculate motor harmonic energy (200-1000 Hz)
    >>> motor_energy = get_frequency_band_energy(magnitude, freqs, (200, 1000))
    >>> print(f"Motor/Propeller ratio: {motor_energy.mean() / prop_energy.mean():.2f}")
    """
    f_low, f_high = band_hz

    if f_low < 0 or f_high < 0:
        raise ValueError(f"Frequency band must be positive. Got ({f_low}, {f_high}).")

    if f_low >= f_high:
        raise ValueError(
            f"Lower frequency must be < higher frequency. Got ({f_low}, {f_high})."
        )

    # Find bins within the band
    band_mask = (freqs >= f_low) & (freqs <= f_high)

    if not np.any(band_mask):
        warnings.warn(
            f"No frequency bins found in band ({f_low}, {f_high}) Hz. "
            f"Frequency range is {freqs[0]:.1f} to {freqs[-1]:.1f} Hz.",
            UserWarning
        )
        return np.zeros(magnitude.shape[0])

    # Sum energy across frequency bins in the band
    energy = np.sum(magnitude[:, band_mask], axis=1)

    return energy


def find_spectral_peaks(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    n_peaks: int = 5,
    min_freq: float = 50.0,
    max_freq: float = 5000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the strongest spectral peaks in each frame.

    Useful for identifying drone propeller fundamental frequency and harmonics.

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrum [n_frames, n_freqs]
    freqs : np.ndarray
        Frequency bins [Hz]
    n_peaks : int, default=5
        Number of peaks to find per frame
    min_freq : float, default=50.0
        Minimum frequency to search for peaks (Hz)
    max_freq : float, default=5000.0
        Maximum frequency to search for peaks (Hz)

    Returns
    -------
    peak_freqs : np.ndarray
        Frequencies of peaks [n_frames, n_peaks] in Hz
    peak_mags : np.ndarray
        Magnitudes of peaks [n_frames, n_peaks]

    Examples
    --------
    >>> # Find top 3 spectral peaks for drone detection
    >>> peak_freqs, peak_mags = find_spectral_peaks(magnitude, freqs, n_peaks=3)
    >>> print(f"Frame 0 fundamental: {peak_freqs[0, 0]:.1f} Hz")
    >>> print(f"Frame 0 harmonics: {peak_freqs[0, 1:]}")
    """
    # Limit search to specified frequency range
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    search_freqs = freqs[freq_mask]
    search_mag = magnitude[:, freq_mask]

    n_frames = magnitude.shape[0]
    peak_freqs = np.zeros((n_frames, n_peaks))
    peak_mags = np.zeros((n_frames, n_peaks))

    for i in range(n_frames):
        # Get indices of top n_peaks
        top_indices = np.argsort(search_mag[i])[-n_peaks:][::-1]

        peak_freqs[i] = search_freqs[top_indices]
        peak_mags[i] = search_mag[i, top_indices]

    return peak_freqs, peak_mags


# ============================================================================
# Demo / Test Code
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of FFT analysis for acoustic drone detection.

    This demo:
    1. Generates synthetic drone-like signal with harmonics
    2. Applies framing and windowing
    3. Computes FFT
    4. Visualizes frequency spectrum
    5. Demonstrates spectral analysis features
    """
    import matplotlib.pyplot as plt
    import sys
    import os

    # Import framing module (assumes it's in the same directory)
    try:
        from framing_windowing import frame_and_window
    except ImportError:
        print("Error: Cannot import framing_windowing module.")
        print("Make sure framing_windowing.py is in the same directory.")
        sys.exit(1)

    print("=" * 70)
    print("Acoustic Drone Detection: FFT Analysis Demo")
    print("=" * 70)

    # ========================================================================
    # Generate synthetic drone signal
    # ========================================================================

    fs = 16000  # 16 kHz sampling rate
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Drone propeller characteristics
    fundamental = 150  # Hz (typical quadcopter propeller)

    # Generate multi-harmonic signal
    signal = (
        0.5 * np.sin(2 * np.pi * fundamental * t) +          # Fundamental
        0.3 * np.sin(2 * np.pi * 2 * fundamental * t) +      # 2nd harmonic
        0.2 * np.sin(2 * np.pi * 3 * fundamental * t) +      # 3rd harmonic
        0.1 * np.sin(2 * np.pi * 4 * fundamental * t) +      # 4th harmonic
        0.05 * np.sin(2 * np.pi * 5 * fundamental * t) +     # 5th harmonic
        0.03 * np.random.randn(len(t))                       # Noise
    )

    print(f"\nGenerated synthetic drone signal:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Propeller fundamental: {fundamental} Hz")
    print(f"  Expected harmonics: {[i*fundamental for i in range(1, 6)]} Hz")

    # ========================================================================
    # Frame and window the signal
    # ========================================================================

    frames, frame_times = frame_and_window(
        signal, fs,
        frame_length_ms=64.0,  # 64ms frames for good frequency resolution
        hop_length_ms=32.0,    # 50% overlap
        window_type='hann'
    )

    print(f"\nFraming:")
    print(f"  Number of frames: {frames.shape[0]}")
    print(f"  Samples per frame: {frames.shape[1]}")

    # ========================================================================
    # Compute FFT
    # ========================================================================

    freqs, spectrum, magnitude, magnitude_db = compute_fft_per_frame(
        frames, fs,
        nfft=1024,  # Zero-pad for better frequency resolution
        remove_dc=True
    )

    print(f"\nFFT Analysis:")
    print(f"  Frequency bins: {len(freqs)}")
    print(f"  Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz/bin")
    print(f"  Frequency range: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
    print(f"  Spectrum shape: {spectrum.shape}")

    # ========================================================================
    # Spectral analysis
    # ========================================================================

    # Find peaks in first frame
    peak_freqs, peak_mags = find_spectral_peaks(
        magnitude, freqs,
        n_peaks=5,
        min_freq=50,
        max_freq=1000
    )

    print(f"\nSpectral peaks in frame 0:")
    for i, (freq, mag) in enumerate(zip(peak_freqs[0], peak_mags[0])):
        print(f"  Peak {i+1}: {freq:.1f} Hz (magnitude: {mag:.2f})")

    # Calculate energy in different bands
    prop_energy = get_frequency_band_energy(magnitude, freqs, (100, 200))
    motor_energy = get_frequency_band_energy(magnitude, freqs, (200, 1000))

    print(f"\nFrequency band energy (average across frames):")
    print(f"  Propeller band (100-200 Hz): {prop_energy.mean():.2f}")
    print(f"  Motor harmonics (200-1000 Hz): {motor_energy.mean():.2f}")
    print(f"  Ratio: {motor_energy.mean() / prop_energy.mean():.2f}")

    # ========================================================================
    # Visualization
    # ========================================================================

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Magnitude spectrum (linear scale) - first frame
    frame_idx = len(frames) // 2  # Middle frame
    axes[0].plot(freqs, magnitude[frame_idx], linewidth=1.5)
    axes[0].set_title(f'Linear Magnitude Spectrum (Frame {frame_idx})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_xlim([0, 1000])  # Focus on drone frequency range
    axes[0].grid(True, alpha=0.3)

    # Mark expected harmonics
    for i in range(1, 6):
        axes[0].axvline(i * fundamental, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 2: Magnitude spectrum (dB scale)
    axes[1].plot(freqs, magnitude_db[frame_idx], linewidth=1.5)
    axes[1].set_title(f'Magnitude Spectrum in dB (Frame {frame_idx})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_xlim([0, 1000])
    axes[1].set_ylim([-80, np.max(magnitude_db[frame_idx]) + 5])
    axes[1].grid(True, alpha=0.3)

    # Mark expected harmonics
    for i in range(1, 6):
        axes[1].axvline(i * fundamental, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 3: Spectrogram (time-frequency representation)
    # Limit frequency range for better visualization
    freq_limit = 1000  # Hz
    freq_mask = freqs <= freq_limit

    im = axes[2].imshow(
        magnitude_db[:, freq_mask].T,
        aspect='auto',
        origin='lower',
        extent=[frame_times[0], frame_times[-1], 0, freq_limit],
        cmap='viridis',
        vmin=np.percentile(magnitude_db, 10),
        vmax=np.percentile(magnitude_db, 99)
    )
    axes[2].set_title('Spectrogram (dB)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Frequency (Hz)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20)

    # Mark expected harmonics on spectrogram
    for i in range(1, 6):
        axes[2].axhline(i * fundamental, color='red', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()

    output_path = '/mnt/d/edth/fft_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
