"""
framing_windowing.py

Framing & Windowing for Acoustic Drone Detection Signals
=========================================================

This module provides robust signal framing and windowing functions specifically
designed for acoustic-based drone detection systems. Proper framing and windowing
are critical for:

1. **Time-Frequency Analysis**: Drones produce distinctive acoustic signatures
   in specific frequency bands (typically 100-5000 Hz for propellers and motors)

2. **Spectral Leakage Reduction**: Window functions minimize spectral leakage
   during FFT/STFT analysis, improving frequency resolution

3. **Temporal Resolution**: Short frames (20-50ms) capture rapid changes in
   drone acoustic patterns during acceleration, hovering, or maneuvering

Key Concepts:
- Frame Length: Duration of each analysis window (trade-off: frequency vs time resolution)
- Hop Length: Shift between consecutive frames (controls overlap and processing load)
- Windowing: Tapering function to reduce edge discontinuities

References:
- Acoustic drone detection typically uses 50-75% overlap for robust detection
- Hann window provides good frequency selectivity for harmonic analysis
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


# ============================================================================
# Window Function Generators
# ============================================================================

def get_window(window_type: str, window_length: int) -> np.ndarray:
    """
    Generate a window function for signal processing.

    Different window types offer trade-offs between main lobe width and
    side lobe attenuation. For drone detection:
    - Hann: Good general purpose, moderate frequency resolution
    - Hamming: Better frequency resolution, slightly worse side lobes
    - Blackman: Excellent side lobe suppression, wider main lobe

    Parameters
    ----------
    window_type : str
        Type of window: 'hann', 'hamming', 'blackman', 'rectangular'
    window_length : int
        Length of the window in samples

    Returns
    -------
    window : np.ndarray
        1D array containing the window coefficients

    Raises
    ------
    ValueError
        If window_type is not recognized or window_length <= 0
    """
    if window_length <= 0:
        raise ValueError(f"Window length must be positive, got {window_length}")

    window_type = window_type.lower().strip()

    if window_type == 'hann':
        return np.hanning(window_length)
    elif window_type == 'hamming':
        return np.hamming(window_length)
    elif window_type == 'blackman':
        return np.blackman(window_length)
    elif window_type == 'rectangular' or window_type == 'rect':
        return np.ones(window_length)
    else:
        raise ValueError(
            f"Unknown window type '{window_type}'. "
            f"Supported: 'hann', 'hamming', 'blackman', 'rectangular'"
        )


# ============================================================================
# Input Validation Functions
# ============================================================================

def _validate_audio_input(audio: np.ndarray) -> np.ndarray:
    """
    Validate and sanitize audio input.

    Ensures the audio signal is suitable for processing:
    - Converts to numpy array
    - Flattens to 1D (for mono processing)
    - Checks for NaN/Inf values
    - Converts to float for numerical stability

    Parameters
    ----------
    audio : array-like
        Input audio signal

    Returns
    -------
    audio_clean : np.ndarray
        Validated and sanitized 1D audio array

    Raises
    ------
    ValueError
        If audio contains NaN/Inf or is empty
    """
    # Convert to numpy array and flatten to 1D
    audio = np.asarray(audio).astype(float).flatten()

    # Check for empty array
    if audio.size == 0:
        raise ValueError("Audio array is empty")

    # Check for invalid values (NaN or Inf)
    if np.any(np.isnan(audio)):
        raise ValueError(
            "Audio contains NaN values. Please clean your input signal."
        )

    if np.any(np.isinf(audio)):
        raise ValueError(
            "Audio contains Inf values. Please check for amplitude clipping or errors."
        )

    return audio


def _validate_sampling_rate(fs: Union[int, float]) -> float:
    """
    Validate sampling rate.

    Typical acoustic sensors for drone detection use:
    - 16 kHz (common for speech-quality microphones)
    - 44.1 kHz (standard audio)
    - 48 kHz (professional audio)
    - Up to 96 kHz (high-quality research)

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
        raise ValueError(f"Sampling rate must be positive, got {fs} Hz")

    if fs < 1000:
        warnings.warn(
            f"Unusually low sampling rate ({fs} Hz). "
            f"Drone acoustic detection typically requires ≥8 kHz for propeller harmonics.",
            UserWarning
        )

    if fs > 200000:
        warnings.warn(
            f"Very high sampling rate ({fs} Hz). "
            f"This may be unnecessary for drone detection and increases computation.",
            UserWarning
        )

    return fs


def _validate_frame_parameters(
    frame_length_ms: float,
    hop_length_ms: float,
    fs: float,
    audio_length: int
) -> Tuple[int, int]:
    """
    Validate and convert frame parameters from milliseconds to samples.

    Recommended values for drone detection:
    - Frame length: 20-50 ms (balances time/frequency resolution)
    - Hop length: 10-25 ms (50-75% overlap for robust detection)

    Parameters
    ----------
    frame_length_ms : float
        Frame length in milliseconds
    hop_length_ms : float
        Hop size in milliseconds
    fs : float
        Sampling rate in Hz
    audio_length : int
        Length of audio signal in samples

    Returns
    -------
    frame_length : int
        Frame length in samples
    hop_length : int
        Hop length in samples

    Raises
    ------
    ValueError
        If parameters are invalid or inconsistent
    """
    # Check for positive values
    if frame_length_ms <= 0:
        raise ValueError(f"frame_length_ms must be positive, got {frame_length_ms}")

    if hop_length_ms <= 0:
        raise ValueError(f"hop_length_ms must be positive, got {hop_length_ms}")

    # Convert to samples
    frame_length = int(round(fs * frame_length_ms / 1000.0))
    hop_length = int(round(fs * hop_length_ms / 1000.0))

    # Verify conversion resulted in valid sample counts
    if frame_length <= 0:
        raise ValueError(
            f"frame_length_ms={frame_length_ms} at fs={fs} Hz results in "
            f"frame_length=0 samples. Increase frame_length_ms."
        )

    if hop_length <= 0:
        raise ValueError(
            f"hop_length_ms={hop_length_ms} at fs={fs} Hz results in "
            f"hop_length=0 samples. Increase hop_length_ms."
        )

    # Check if hop_length is larger than frame_length (no overlap, possible gap)
    if hop_length > frame_length:
        warnings.warn(
            f"hop_length ({hop_length} samples, {hop_length_ms} ms) > "
            f"frame_length ({frame_length} samples, {frame_length_ms} ms). "
            f"This creates gaps between frames and may miss transient drone events.",
            UserWarning
        )

    # Check for very high overlap (>90%)
    overlap_ratio = 1.0 - (hop_length / frame_length)
    if overlap_ratio > 0.9:
        warnings.warn(
            f"Very high overlap ({overlap_ratio*100:.1f}%). "
            f"This increases computation without much benefit for drone detection.",
            UserWarning
        )

    # Check if audio is long enough for at least one frame
    if audio_length < frame_length:
        raise ValueError(
            f"Audio too short ({audio_length} samples = {audio_length/fs*1000:.1f} ms) "
            f"for frame length {frame_length} samples ({frame_length_ms} ms). "
            f"Need at least {frame_length} samples."
        )

    return frame_length, hop_length


# ============================================================================
# Main Framing & Windowing Function
# ============================================================================

def frame_and_window(
    audio: np.ndarray,
    fs: Union[int, float],
    frame_length_ms: float = 32.0,
    hop_length_ms: float = 16.0,
    window_type: str = 'hann',
    zero_pad: bool = False,
    normalize_energy: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split audio signal into overlapping frames and apply windowing.

    This is the core preprocessing step for acoustic drone detection. It converts
    a continuous audio stream into a sequence of short-time frames suitable for:
    - Short-Time Fourier Transform (STFT)
    - Mel-Frequency Cepstral Coefficients (MFCC)
    - Spectral feature extraction
    - Deep learning input preparation

    Process Flow:
    1. Validate input audio and parameters
    2. Calculate frame boundaries with specified overlap
    3. Extract overlapping frames from audio
    4. Apply window function to each frame (reduces spectral leakage)
    5. Optionally normalize energy for consistent analysis

    Parameters
    ----------
    audio : np.ndarray or array-like
        1D numpy array containing mono audio samples (float or int).
        For multi-channel recordings, extract single channel before calling.
    fs : int or float
        Sampling rate in Hz (e.g., 16000 for 16 kHz, 44100 for 44.1 kHz).
    frame_length_ms : float, default=32.0
        Frame length in milliseconds. Typical values:
        - 20-25 ms: High temporal resolution, lower frequency resolution
        - 32-40 ms: Balanced (recommended for drone detection)
        - 50-64 ms: High frequency resolution, lower temporal resolution
    hop_length_ms : float, default=16.0
        Hop size (frame shift) in milliseconds. Determines overlap:
        - 16 ms with 32 ms frame → 50% overlap (recommended)
        - 8 ms with 32 ms frame → 75% overlap (high overlap for tracking)
        - 32 ms with 32 ms frame → 0% overlap (no overlap, fast processing)
    window_type : str, default='hann'
        Window function to apply. Options:
        - 'hann': Good general-purpose window, smooth frequency response
        - 'hamming': Slightly better frequency resolution
        - 'blackman': Best side-lobe suppression, wider main lobe
        - 'rectangular': No windowing (not recommended, causes spectral leakage)
    zero_pad : bool, default=False
        If True, zero-pad the audio to ensure all samples are included in frames.
        Useful when you need to process every sample without loss.
    normalize_energy : bool, default=False
        If True, normalize each frame to have unit energy (L2 norm = 1).
        Useful for making analysis robust to amplitude variations.

    Returns
    -------
    frames_win : np.ndarray
        2D array of shape (n_frames, frame_length_samples) containing
        windowed frames ready for spectral analysis. Each row is one frame.
    frame_times : np.ndarray
        1D array of length n_frames giving the center time (in seconds)
        of each frame. Useful for temporal alignment and visualization.

    Raises
    ------
    ValueError
        If input parameters are invalid or audio is too short

    Examples
    --------
    >>> # Process 1 second of audio at 16 kHz
    >>> fs = 16000
    >>> audio = np.random.randn(16000)  # Example: noise
    >>>
    >>> # Standard settings for drone detection
    >>> frames, times = frame_and_window(
    ...     audio, fs,
    ...     frame_length_ms=32.0,
    ...     hop_length_ms=16.0,
    ...     window_type='hann'
    ... )
    >>>
    >>> print(f"Generated {frames.shape[0]} frames")
    >>> print(f"Each frame is {frames.shape[1]} samples")
    >>> print(f"Frame times: {times[0]:.3f}s to {times[-1]:.3f}s")

    Notes
    -----
    - For real-time drone detection, use smaller hop sizes for lower latency
    - For offline analysis, larger overlaps (75%) improve detection robustness
    - Always use windowing (Hann or better) to reduce spectral artifacts
    - Consider energy normalization if microphone gain varies during recording
    """

    # ========================================================================
    # Step 1: Validate and sanitize inputs
    # ========================================================================

    audio = _validate_audio_input(audio)
    fs = _validate_sampling_rate(fs)
    frame_length, hop_length = _validate_frame_parameters(
        frame_length_ms, hop_length_ms, fs, len(audio)
    )

    # ========================================================================
    # Step 2: Optional zero-padding
    # ========================================================================

    if zero_pad:
        # Calculate how many samples would be left out
        samples_after_frames = (len(audio) - frame_length) % hop_length
        if samples_after_frames > 0:
            pad_length = hop_length - samples_after_frames
            audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    # ========================================================================
    # Step 3: Calculate number of frames
    # ========================================================================

    # Number of complete frames that fit in the audio
    # Formula: n_frames = floor((audio_length - frame_length) / hop_length) + 1
    n_frames = 1 + (len(audio) - frame_length) // hop_length

    if n_frames <= 0:
        raise ValueError(
            f"Cannot create any frames with current parameters. "
            f"Audio length: {len(audio)} samples, Frame length: {frame_length} samples"
        )

    # Log information about frame coverage
    samples_used = (n_frames - 1) * hop_length + frame_length
    samples_unused = len(audio) - samples_used
    coverage_percent = (samples_used / len(audio)) * 100

    if samples_unused > frame_length // 2:
        warnings.warn(
            f"Only {coverage_percent:.1f}% of audio is covered by frames. "
            f"{samples_unused} samples ({samples_unused/fs*1000:.1f} ms) will be unused. "
            f"Consider using zero_pad=True or adjusting hop_length.",
            UserWarning
        )

    # ========================================================================
    # Step 4: Extract frames from audio
    # ========================================================================

    # Pre-allocate array for efficiency: (n_frames, frame_length)
    frames = np.zeros((n_frames, frame_length), dtype=np.float64)

    # Slice audio into overlapping frames
    # Each frame starts at index: i * hop_length
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        frames[i, :] = audio[start_idx:end_idx]

    # ========================================================================
    # Step 5: Generate and apply window function
    # ========================================================================

    window = get_window(window_type, frame_length)

    # Apply window to all frames using broadcasting
    # Shape: (n_frames, frame_length) * (1, frame_length) → (n_frames, frame_length)
    frames_win = frames * window[None, :]

    # ========================================================================
    # Step 6: Optional energy normalization
    # ========================================================================

    if normalize_energy:
        # Calculate L2 norm (energy) of each frame
        frame_energies = np.sqrt(np.sum(frames_win**2, axis=1, keepdims=True))

        # Avoid division by zero for silent frames
        # Replace zero energies with 1 (keeps silent frames as zero)
        frame_energies[frame_energies == 0] = 1.0

        # Normalize each frame to unit energy
        frames_win = frames_win / frame_energies

    # ========================================================================
    # Step 7: Calculate frame center times
    # ========================================================================

    # Time (in seconds) of each frame's center point
    # Frame i starts at: i * hop_length
    # Frame i center at: i * hop_length + frame_length / 2
    frame_centers_sec = (np.arange(n_frames) * hop_length + frame_length / 2.0) / fs

    return frames_win, frame_centers_sec


# ============================================================================
# Utility Functions
# ============================================================================

def get_frame_info(
    audio_length_samples: int,
    fs: float,
    frame_length_ms: float,
    hop_length_ms: float
) -> dict:
    """
    Get detailed information about framing parameters without processing audio.

    Useful for planning processing pipelines and understanding resource requirements.

    Parameters
    ----------
    audio_length_samples : int
        Length of audio in samples
    fs : float
        Sampling rate in Hz
    frame_length_ms : float
        Frame length in milliseconds
    hop_length_ms : float
        Hop size in milliseconds

    Returns
    -------
    info : dict
        Dictionary containing:
        - n_frames: Number of frames that will be generated
        - frame_length_samples: Frame length in samples
        - hop_length_samples: Hop length in samples
        - overlap_ratio: Fraction of overlap between frames (0-1)
        - coverage_percent: Percentage of audio covered by frames
        - audio_duration_sec: Total audio duration in seconds
        - frame_duration_sec: Duration of each frame in seconds
        - hop_duration_sec: Time shift between frames in seconds
    """
    fs = _validate_sampling_rate(fs)
    frame_length, hop_length = _validate_frame_parameters(
        frame_length_ms, hop_length_ms, fs, audio_length_samples
    )

    n_frames = max(0, 1 + (audio_length_samples - frame_length) // hop_length)
    overlap_ratio = 1.0 - (hop_length / frame_length) if frame_length > 0 else 0

    samples_used = (n_frames - 1) * hop_length + frame_length if n_frames > 0 else 0
    coverage_percent = (samples_used / audio_length_samples * 100) if audio_length_samples > 0 else 0

    return {
        'n_frames': n_frames,
        'frame_length_samples': frame_length,
        'hop_length_samples': hop_length,
        'overlap_ratio': overlap_ratio,
        'overlap_percent': overlap_ratio * 100,
        'coverage_percent': coverage_percent,
        'audio_duration_sec': audio_length_samples / fs,
        'frame_duration_sec': frame_length / fs,
        'hop_duration_sec': hop_length / fs,
        'samples_per_frame': frame_length,
        'samples_unused': audio_length_samples - samples_used,
    }


# ============================================================================
# Demo / Test Code
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of framing and windowing for acoustic drone detection.

    This demo:
    1. Generates a synthetic signal mimicking drone acoustic signature
    2. Applies framing and windowing
    3. Visualizes results
    4. Shows parameter analysis
    """
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Acoustic Drone Detection: Framing & Windowing Demo")
    print("=" * 70)

    # ========================================================================
    # Configuration (typical for drone detection)
    # ========================================================================

    fs = 16000  # 16 kHz sampling rate (common for acoustic monitoring)
    duration = 2.0  # 2 seconds of audio
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # ========================================================================
    # Generate synthetic drone-like acoustic signal
    # ========================================================================

    # Drone acoustic signatures typically contain:
    # 1. Fundamental propeller frequency (50-200 Hz)
    # 2. Harmonics of propeller frequency
    # 3. Motor noise (broadband component)

    fundamental = 120  # Hz (typical propeller frequency)

    # Create multi-harmonic signal (propeller and motor harmonics)
    test_signal = (
        0.3 * np.sin(2 * np.pi * fundamental * t) +          # Fundamental
        0.2 * np.sin(2 * np.pi * 2 * fundamental * t) +      # 2nd harmonic
        0.1 * np.sin(2 * np.pi * 3 * fundamental * t) +      # 3rd harmonic
        0.05 * np.sin(2 * np.pi * 4 * fundamental * t) +     # 4th harmonic
        0.02 * np.random.randn(len(t))                       # Background noise
    )

    # Add a brief amplitude modulation (simulating drone movement)
    modulation = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * t)
    test_signal *= modulation

    print(f"\nGenerated test signal:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Samples: {len(test_signal)}")
    print(f"  Fundamental frequency: {fundamental} Hz (typical propeller)")

    # ========================================================================
    # Apply framing and windowing
    # ========================================================================

    frame_length_ms = 32.0  # 32 ms frames (good balance for drone detection)
    hop_length_ms = 16.0    # 16 ms hop (50% overlap)

    print(f"\nFraming parameters:")
    print(f"  Frame length: {frame_length_ms} ms")
    print(f"  Hop length: {hop_length_ms} ms")
    print(f"  Overlap: {(1 - hop_length_ms/frame_length_ms)*100:.0f}%")

    # Get frame information before processing
    info = get_frame_info(len(test_signal), fs, frame_length_ms, hop_length_ms)
    print(f"\nExpected output:")
    print(f"  Number of frames: {info['n_frames']}")
    print(f"  Samples per frame: {info['samples_per_frame']}")
    print(f"  Coverage: {info['coverage_percent']:.1f}%")

    # Process with different window types for comparison
    frames_hann, frame_times = frame_and_window(
        test_signal, fs,
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
        window_type='hann',
        normalize_energy=False
    )

    print(f"\nActual output:")
    print(f"  Windowed frames shape: {frames_hann.shape}")
    print(f"  Frame times shape: {frame_times.shape}")
    print(f"  Time range: {frame_times[0]:.3f}s to {frame_times[-1]:.3f}s")

    # ========================================================================
    # Visualization
    # ========================================================================

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Original signal
    axes[0].plot(t, test_signal, linewidth=0.5)
    axes[0].set_title('Original Acoustic Signal (Synthetic Drone)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, duration])

    # Mark frame centers on original signal
    for ft in frame_times[::10]:  # Show every 10th frame to avoid clutter
        axes[0].axvline(ft, color='red', alpha=0.3, linestyle='--', linewidth=0.5)

    # Plot 2: Multiple windowed frames
    axes[1].set_title('Sample Windowed Frames (Hann Window)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sample index within frame')
    axes[1].set_ylabel('Amplitude')

    # Plot several frames to show the windowing effect
    frame_indices = [10, 30, 50, 70]
    colors = plt.cm.viridis(np.linspace(0, 1, len(frame_indices)))

    for idx, color in zip(frame_indices, colors):
        if idx < len(frames_hann):
            axes[1].plot(frames_hann[idx], label=f'Frame {idx} (t={frame_times[idx]:.2f}s)',
                        color=color, alpha=0.7)

    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Comparison of different window types
    axes[2].set_title('Window Function Comparison', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Sample index')
    axes[2].set_ylabel('Amplitude')

    frame_len = info['samples_per_frame']
    window_hann = get_window('hann', frame_len)
    window_hamming = get_window('hamming', frame_len)
    window_blackman = get_window('blackman', frame_len)

    axes[2].plot(window_hann, label='Hann (recommended)', linewidth=2)
    axes[2].plot(window_hamming, label='Hamming', linewidth=2, linestyle='--')
    axes[2].plot(window_blackman, label='Blackman', linewidth=2, linestyle=':')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/d/edth/framing_windowing_demo.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: /mnt/d/edth/framing_windowing_demo.png")
    plt.show()

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
