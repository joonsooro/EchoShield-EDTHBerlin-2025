#!/usr/bin/env python3
"""
main_drone_detector.py

Complete Acoustic Drone Detection Pipeline
==========================================

This script integrates all refactored modules into a complete end-to-end
drone detection system. It processes audio files through:

1. Audio loading and preprocessing (mono or stereo)
2. Framing and windowing
3. GCC-PHAT Direction of Arrival estimation (stereo only)
4. FFT analysis
5. Harmonic filtering (optional)
6. Multi-evidence detection (SNR + Harmonic + Temporal)
7. Results visualization and export

For stereo audio, the system automatically estimates the drone's direction
using GCC-PHAT algorithm. Ensure correct microphone spacing is specified
via --mic-spacing for accurate angle estimation.

Data Structure Expected:
-----------------------
data/
├── train/
│   ├── drone/       ← Drone audio samples
│   ├── helicopter/  ← Helicopter audio samples
│   └── background/  ← Background noise samples
└── val/
    ├── drone/
    ├── helicopter/
    └── background/

Usage Examples:
--------------
# Process single file (mono or stereo)
python main_drone_detector.py --file data/train/drone/0055b2bb.wav

# Process stereo file with custom microphone spacing (8 cm laptop mics)
python main_drone_detector.py --file audio_stereo.wav --mic-spacing 0.08

# Process entire directory
python main_drone_detector.py --dir data/train/drone

# Batch process with output
python main_drone_detector.py --dir data/train/drone --output results/drone_results.csv

# Process all classes
python main_drone_detector.py --batch --split train --output results/train_results.csv

# With visualization
python main_drone_detector.py --file data/train/drone/0055b2bb.wav --visualize --save-plots results/

# Stereo file with DOA estimation (uses default 14 cm spacing)
python main_drone_detector.py --file stereo_recording.wav --visualize

# Stereo file with custom mic spacing (20 cm array)
python main_drone_detector.py --file stereo_recording.wav --mic-spacing 0.20 --visualize
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt

# Audio I/O
try:
    import scipy.io.wavfile as wavfile
    AUDIO_LIB = 'scipy'
except ImportError:
    try:
        import soundfile as sf
        AUDIO_LIB = 'soundfile'
    except ImportError:
        print("ERROR: Neither scipy nor soundfile is installed.")
        print("Install one with: pip install scipy  OR  pip install soundfile")
        sys.exit(1)

# Import our refactored modules
try:
    from framing_windowing import frame_and_window, get_frame_info
    from fft import compute_fft_per_frame, find_spectral_peaks, get_frequency_band_energy
    from harmonic_filter import frequency_harmonic_filter, make_harmonic_mask, estimate_snr_improvement
    from energy_likelihood_detector import EnergyLikelihoodDetector
except ImportError as e:
    print(f"ERROR: Cannot import detection modules: {e}")
    print("Make sure all refactored .py files are in the same directory.")
    sys.exit(1)


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class DetectorConfig:
    """Configuration for drone detection pipeline."""

    # Audio preprocessing
    target_fs: int = 16000              # Resample to 16 kHz (if needed)
    max_duration: Optional[float] = 10.0  # Max audio duration in seconds (None = no limit)

    # Framing parameters
    frame_length_ms: float = 64.0       # 64 ms frames
    hop_length_ms: float = 32.0         # 32 ms hop (50% overlap)
    window_type: str = 'hann'           # Window function

    # FFT parameters
    nfft: int = 1024                    # FFT size

    # Filtering parameters
    use_harmonic_filter: bool = True   # Enable harmonic filtering
    coarse_band_hz: Tuple[float, float] = (100, 5000)  # Coarse band-pass
    f0_estimate: Optional[float] = None  # Fundamental frequency (auto-detect if None)
    n_harmonics: int = 7                # Number of harmonics
    harmonic_bw_hz: float = 40          # Bandwidth per harmonic

    # Detection parameters
    detector_f0: float = 150            # Expected fundamental for detector
    detector_n_harmonics: int = 7       # Harmonics for detection
    detector_band_hz: Tuple[float, float] = (100, 2000)
    detector_harmonic_bw: float = 40

    # Evidence weights
    weight_snr: float = 0.4
    weight_harmonic: float = 0.3
    weight_temporal: float = 0.3

    # Detection thresholds
    snr_range_db: Tuple[float, float] = (0.0, 30.0)
    harmonic_min_snr_db: float = 3.0
    temporal_window: int = 5
    confidence_threshold: float = 0.75

    # Stereo/DOA parameters
    mic_spacing_m: float = 0.14         # Microphone spacing in meters (14 cm default)
                                        # Typical values:
                                        # - Laptop/phone stereo: 0.05-0.10 m (5-10 cm)
                                        # - Standalone stereo mics: 0.10-0.30 m (10-30 cm)
                                        # - Custom arrays: measure actual spacing

    # Output control
    verbose: bool = True


# ============================================================================
# Audio Loading
# ============================================================================

def load_audio(file_path: str, target_fs: Optional[int] = None) -> Tuple[np.ndarray, int, bool]:
    """
    Load audio file and optionally resample.

    Parameters
    ----------
    file_path : str
        Path to audio file
    target_fs : int, optional
        Target sampling rate (resamples if different from original)

    Returns
    -------
    audio : np.ndarray
        Audio signal (mono: [samples] or stereo: [samples, 2])
    fs : int
        Sampling rate [Hz]
    is_stereo : bool
        True if audio is stereo, False if mono
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load audio
    if AUDIO_LIB == 'scipy':
        fs, audio = wavfile.read(file_path)
        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
    else:  # soundfile
        audio, fs = sf.read(file_path)

    # Keep stereo if available (for GCC-PHAT angle estimation)
    # audio.ndim == 1: mono [samples]
    # audio.ndim == 2: stereo [samples, channels]
    is_stereo = (audio.ndim > 1)

    if not is_stereo:
        # Mono audio - keep as is
        pass
    else:
        # Stereo audio - keep both channels
        # Ensure shape is [samples, channels]
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T

    # Resample if needed
    if target_fs is not None and fs != target_fs:
        # Simple resampling using scipy
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_fs / fs)
            audio = signal.resample(audio, num_samples)
            fs = target_fs
        except ImportError:
            warnings.warn(
                f"scipy.signal not available for resampling. Using original fs={fs} Hz",
                UserWarning
            )

    return audio, fs, is_stereo


# ============================================================================
# Pipeline Functions
# ============================================================================

def estimate_fundamental_frequency(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    search_range: Tuple[float, float] = (50, 300)
) -> float:
    """
    Estimate fundamental frequency from spectrum.

    Uses peak detection in expected drone frequency range.

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude spectrum [n_frames, n_freqs]
    freqs : np.ndarray
        Frequency bins [Hz]
    search_range : tuple
        Frequency range to search for f0 [Hz]

    Returns
    -------
    f0 : float
        Estimated fundamental frequency [Hz]
    """
    # Average magnitude across frames
    avg_magnitude = np.mean(magnitude, axis=0)

    # Find peaks in search range
    peak_freqs, peak_mags = find_spectral_peaks(
        avg_magnitude.reshape(1, -1),
        freqs,
        n_peaks=5,
        min_freq=search_range[0],
        max_freq=search_range[1]
    )

    # Return strongest peak as f0
    f0 = peak_freqs[0, 0]

    return f0


def frame_and_window_multichannel(
    audio: np.ndarray,
    fs: float,
    frame_length_ms: float = 64.0,
    hop_length_ms: float = 32.0,
    window_type: str = 'hann'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame and window multi-channel audio for stereo processing.

    Parameters
    ----------
    audio : np.ndarray
        Multi-channel audio [samples, n_channels]
    fs : float
        Sampling rate [Hz]
    frame_length_ms : float
        Frame length in milliseconds
    hop_length_ms : float
        Hop length in milliseconds
    window_type : str
        Window type ('hann', 'hamming', etc.)

    Returns
    -------
    frames : np.ndarray
        Framed audio [n_frames, frame_length, n_channels]
    frame_times : np.ndarray
        Frame center times [seconds]
    """
    from framing_windowing import get_window

    n_samples, n_channels = audio.shape
    frame_length = int(round(fs * frame_length_ms / 1000.0))
    hop_length = int(round(fs * hop_length_ms / 1000.0))

    # Calculate number of frames
    n_frames = 1 + (n_samples - frame_length) // hop_length

    # Pre-allocate
    frames = np.zeros((n_frames, frame_length, n_channels), dtype=np.float64)

    # Extract frames for each channel
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        frames[i, :, :] = audio[start_idx:end_idx, :]

    # Apply window
    window = get_window(window_type, frame_length)
    # Broadcast: [n_frames, frame_length, n_channels] * [frame_length, 1]
    frames = frames * window[None, :, None]

    # Calculate frame times
    frame_times = (np.arange(n_frames) * hop_length + frame_length / 2.0) / fs

    return frames, frame_times


def process_audio_file(
    file_path: str,
    config: DetectorConfig
) -> Dict:
    """
    Process single audio file through complete detection pipeline.

    Parameters
    ----------
    file_path : str
        Path to audio file
    config : DetectorConfig
        Detection configuration

    Returns
    -------
    results : dict
        Dictionary with detection results and metrics
    """
    start_time = time.time()

    # ========================================================================
    # Step 1: Load audio
    # ========================================================================
    if config.verbose:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(file_path)}")
        print(f"{'='*70}")

    audio, fs, is_stereo = load_audio(file_path, target_fs=config.target_fs)

    # Trim to max duration if specified
    if config.max_duration is not None:
        max_samples = int(config.max_duration * fs)
        if is_stereo:
            audio = audio[:max_samples, :]
        else:
            audio = audio[:max_samples]

    duration = len(audio) / fs if not is_stereo else audio.shape[0] / fs
    n_channels = 2 if is_stereo else 1

    if config.verbose:
        print(f"\n[1/6] Audio loaded:")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Channels: {'Stereo (2)' if is_stereo else 'Mono (1)'}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Samples: {len(audio) if not is_stereo else audio.shape[0]}")

    # ========================================================================
    # Step 2: Framing and windowing
    # ========================================================================
    doa_angles = None  # Will store DOA estimates if stereo

    if is_stereo:
        # Multi-channel framing for stereo
        frames_multichannel, frame_times = frame_and_window_multichannel(
            audio, fs,
            frame_length_ms=config.frame_length_ms,
            hop_length_ms=config.hop_length_ms,
            window_type=config.window_type
        )

        if config.verbose:
            print(f"\n[2/6] Framing completed (stereo):")
            print(f"  Frames: {frames_multichannel.shape[0]}")
            print(f"  Samples per frame: {frames_multichannel.shape[1]}")
            print(f"  Channels: {frames_multichannel.shape[2]}")
            print(f"  Overlap: {(1 - config.hop_length_ms/config.frame_length_ms)*100:.0f}%")

        # Perform GCC-PHAT for angle estimation
        try:
            from gcc_phat_doa import estimate_doa_gcc_phat

            # Use configured microphone spacing
            mic_spacing = config.mic_spacing_m

            if config.verbose:
                print(f"\n[2b/6] GCC-PHAT DOA estimation:")
                print(f"  Microphone spacing: {mic_spacing*100:.1f} cm ({mic_spacing:.3f} m)")
                print(f"  NOTE: Ensure mic_spacing_m is set correctly for accurate angle estimation!")

            doa_angles, tdoa_series = estimate_doa_gcc_phat(
                frames_multichannel, fs, mic_spacing,
                ref_channel=0,
                c=343.0,
                interp=16,
                robust_averaging=True
            )

            # Convert to degrees for readability
            doa_degrees = np.degrees(doa_angles)

            if config.verbose:
                print(f"  Mean angle: {doa_degrees.mean():.1f}°")
                print(f"  Angle std: {doa_degrees.std():.1f}°")
                print(f"  Angle range: [{doa_degrees.min():.1f}°, {doa_degrees.max():.1f}°]")
        except Exception as e:
            if config.verbose:
                print(f"\n[2b/6] GCC-PHAT failed: {e}. Continuing without angle estimation.")
            doa_angles = None

        # For detection, use one channel (or average both)
        # Let's average both channels after framing
        frames = np.mean(frames_multichannel, axis=2)  # [n_frames, frame_length]
    else:
        # Mono framing
        frames, frame_times = frame_and_window(
            audio, fs,
            frame_length_ms=config.frame_length_ms,
            hop_length_ms=config.hop_length_ms,
            window_type=config.window_type
        )

        if config.verbose:
            print(f"\n[2/6] Framing completed (mono):")
            print(f"  Frames: {frames.shape[0]}")
            print(f"  Samples per frame: {frames.shape[1]}")
            print(f"  Overlap: {(1 - config.hop_length_ms/config.frame_length_ms)*100:.0f}%")

    # ========================================================================
    # Step 3: FFT analysis
    # ========================================================================
    freqs, spectrum, magnitude, magnitude_db = compute_fft_per_frame(
        frames, fs,
        nfft=config.nfft,
        remove_dc=True
    )

    if config.verbose:
        print(f"\n[3/6] FFT computed:")
        print(f"  Frequency bins: {len(freqs)}")
        print(f"  Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        print(f"  Frequency resolution: {freqs[1] - freqs[0]:.2f} Hz/bin")

    # ========================================================================
    # Step 4: Optional harmonic filtering
    # ========================================================================
    spectrum_filtered = spectrum.copy()
    f0_estimated = None
    snr_improvement = 0.0

    if config.use_harmonic_filter:
        # Estimate f0 if not provided
        if config.f0_estimate is None:
            f0_estimated = estimate_fundamental_frequency(magnitude, freqs)
        else:
            f0_estimated = config.f0_estimate

        # Apply harmonic filter
        spectrum_filtered = frequency_harmonic_filter(
            spectrum, freqs,
            coarse_band_hz=config.coarse_band_hz,
            f0=f0_estimated,
            n_harmonics=config.n_harmonics,
            harmonic_bw_hz=config.harmonic_bw_hz
        )

        # Estimate SNR improvement
        harmonic_mask = make_harmonic_mask(
            freqs, f0_estimated, config.n_harmonics, config.harmonic_bw_hz
        )
        snr_improvement = estimate_snr_improvement(
            spectrum, spectrum_filtered, harmonic_mask
        )

        # Recompute magnitude after filtering
        magnitude = np.abs(spectrum_filtered)

        if config.verbose:
            print(f"\n[4/6] Harmonic filtering applied:")
            print(f"  Estimated f0: {f0_estimated:.1f} Hz")
            print(f"  Harmonics: {config.n_harmonics}")
            print(f"  SNR improvement: {snr_improvement:.1f} dB")
    else:
        if config.verbose:
            print(f"\n[4/6] Harmonic filtering: SKIPPED")

    # ========================================================================
    # Step 5: Multi-evidence detection
    # ========================================================================
    detector = EnergyLikelihoodDetector(
        f0=config.detector_f0,
        n_harmonics=config.detector_n_harmonics,
        coarse_band_hz=config.detector_band_hz,
        harmonic_bw_hz=config.detector_harmonic_bw,
        weight_snr=config.weight_snr,
        weight_harmonic=config.weight_harmonic,
        weight_temporal=config.weight_temporal,
        snr_range_db=config.snr_range_db,
        harmonic_min_snr_db=config.harmonic_min_snr_db,
        temporal_window=config.temporal_window,
        confidence_threshold=config.confidence_threshold
    )

    # Run detection frame-by-frame
    confidences = []
    detections = []
    snr_values = []
    harmonic_scores = []
    temporal_scores = []
    valid_harmonics_list = []

    for i in range(len(magnitude)):
        confidence, detected, details = detector.score_frame(freqs, magnitude[i])

        confidences.append(confidence)
        detections.append(detected)
        snr_values.append(details['snr_db'])
        harmonic_scores.append(details['harmonic_score'])
        temporal_scores.append(details['temporal_score'])
        valid_harmonics_list.append(details['valid_harmonics'])

    # Get detector statistics
    stats = detector.get_statistics()

    if config.verbose:
        print(f"\n[5/6] Detection completed:")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Detections: {stats['detections']}")
        print(f"  Detection rate: {stats['detection_rate']*100:.1f}%")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
        print(f"  Average SNR: {np.mean(snr_values):.1f} dB")

    # ========================================================================
    # Step 6: Compute summary metrics
    # ========================================================================
    # Overall decision: majority vote over frames
    detection_rate = stats['detection_rate']
    overall_detected = detection_rate >= 0.3  # At least 30% of frames

    # Confidence statistics
    mean_confidence = np.mean(confidences)
    max_confidence = np.max(confidences)
    std_confidence = np.std(confidences)

    # SNR statistics
    mean_snr = np.mean(snr_values)
    max_snr = np.max(snr_values)

    # Harmonic quality
    mean_harmonic_score = np.mean(harmonic_scores)
    mean_valid_harmonics = np.mean(valid_harmonics_list)

    processing_time = time.time() - start_time

    if config.verbose:
        print(f"\n[6/6] Summary:")
        print(f"  Overall Decision: {'DRONE DETECTED' if overall_detected else 'NO DRONE'}")
        print(f"  Confidence: {mean_confidence:.3f} (±{std_confidence:.3f})")
        if doa_angles is not None:
            doa_deg = np.degrees(doa_angles)
            print(f"  Direction of Arrival: {doa_deg.mean():.1f}° (±{doa_deg.std():.1f}°)")
            print(f"  Microphone Spacing Used: {config.mic_spacing_m*100:.1f} cm ({config.mic_spacing_m:.3f} m)")
        print(f"  Processing time: {processing_time:.2f} seconds")

    # ========================================================================
    # Package results
    # ========================================================================
    results = {
        # File info
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'duration_sec': duration,
        'sampling_rate': fs,
        'is_stereo': is_stereo,
        'n_channels': n_channels,

        # Detection results
        'overall_detected': overall_detected,
        'detection_rate': detection_rate,
        'num_detections': stats['detections'],
        'num_frames': stats['frames_processed'],

        # Confidence metrics
        'mean_confidence': mean_confidence,
        'max_confidence': max_confidence,
        'std_confidence': std_confidence,

        # SNR metrics
        'mean_snr_db': mean_snr,
        'max_snr_db': max_snr,

        # Harmonic metrics
        'mean_harmonic_score': mean_harmonic_score,
        'mean_valid_harmonics': mean_valid_harmonics,
        'estimated_f0': f0_estimated,
        'snr_improvement_db': snr_improvement,

        # DOA metrics (stereo only)
        'doa_angles': doa_angles if doa_angles is not None else None,
        'mean_doa_deg': np.degrees(doa_angles).mean() if doa_angles is not None else None,
        'std_doa_deg': np.degrees(doa_angles).std() if doa_angles is not None else None,
        'mic_spacing_m': config.mic_spacing_m if is_stereo else None,

        # Processing
        'processing_time_sec': processing_time,

        # Time series data (for plotting)
        'frame_times': frame_times,
        'confidences': np.array(confidences),
        'detections': np.array(detections),
        'snr_values': np.array(snr_values),
        'harmonic_scores': np.array(harmonic_scores),
        'temporal_scores': np.array(temporal_scores),

        # Spectra (for visualization)
        'freqs': freqs,
        'magnitude': magnitude,
        'magnitude_db': magnitude_db,
    }

    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_detection_results(results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive visualization of detection results.

    Parameters
    ----------
    results : dict
        Results from process_audio_file()
    save_path : str, optional
        Path to save plot (if None, displays interactively)
    """
    # Check if DOA angles are available (stereo recording)
    has_doa = results.get('doa_angles') is not None
    n_plots = 5 if has_doa else 4

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 15 if has_doa else 12))

    file_name = results['file_name']
    frame_times = results['frame_times']
    confidences = results['confidences']
    detections = results['detections']
    snr_values = results['snr_values']
    harmonic_scores = results['harmonic_scores']
    temporal_scores = results['temporal_scores']

    freqs = results['freqs']
    magnitude_db = results['magnitude_db']

    # Plot 1: Spectrogram
    freq_limit = 2000  # Hz
    freq_mask = freqs <= freq_limit

    im = axes[0].imshow(
        magnitude_db[:, freq_mask].T,
        aspect='auto',
        origin='lower',
        extent=[frame_times[0], frame_times[-1], 0, freq_limit],
        cmap='viridis',
        vmin=np.percentile(magnitude_db, 10),
        vmax=np.percentile(magnitude_db, 99)
    )
    axes[0].set_title(f'Spectrogram: {file_name}', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')

    # Mark detection periods
    if np.any(detections):
        for i in range(len(detections)):
            if detections[i]:
                axes[0].axvline(frame_times[i], color='red', alpha=0.3, linewidth=0.5)

    plt.colorbar(im, ax=axes[0], label='Magnitude (dB)')

    # Plot 2: Confidence and components
    axes[1].plot(frame_times, confidences, label='Final Confidence',
                linewidth=2, color='black')
    axes[1].plot(frame_times, np.array(snr_values) / 30.0,
                label='SNR Score', alpha=0.7, linestyle='--')
    axes[1].plot(frame_times, harmonic_scores,
                label='Harmonic Score', alpha=0.7, linestyle='--')
    axes[1].plot(frame_times, temporal_scores,
                label='Temporal Score', alpha=0.7, linestyle='--')
    axes[1].axhline(0.75, color='orange', linestyle=':',
                   linewidth=2, label='Threshold')

    axes[1].set_title('Detection Confidence & Component Scores',
                     fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Score')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-0.05, 1.05])

    # Plot 3: SNR over time
    axes[2].plot(frame_times, snr_values, linewidth=1.5, color='green')
    axes[2].axhline(3.0, color='orange', linestyle='--',
                   label='Min SNR (3 dB)')
    axes[2].set_title('Signal-to-Noise Ratio', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('SNR (dB)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: DOA angles (if stereo)
    plot_idx = 3
    if has_doa:
        doa_angles = results['doa_angles']
        doa_degrees = np.degrees(doa_angles)
        mic_spacing = results.get('mic_spacing_m', 0.14)

        axes[3].plot(frame_times, doa_degrees, linewidth=2, color='purple', alpha=0.8)
        axes[3].axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        axes[3].fill_between(frame_times, doa_degrees - 5, doa_degrees + 5,
                            color='purple', alpha=0.2, label='±5° range')

        axes[3].set_title(f'Direction of Arrival (DOA) - Mic Spacing: {mic_spacing*100:.1f} cm',
                         fontweight='bold', fontsize=12)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Angle (degrees)')
        axes[3].set_ylim([-95, 95])
        axes[3].grid(True, alpha=0.3)

        # Add angle interpretation
        mean_angle = doa_degrees.mean()
        std_angle = doa_degrees.std()
        axes[3].text(0.02, 0.95, f'Mean: {mean_angle:.1f}° (±{std_angle:.1f}°)',
                    transform=axes[3].transAxes,
                    fontsize=11, fontweight='bold', color='purple',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plot_idx = 4

    # Plot 5 (or 4): Detection timeline
    axes[plot_idx].plot(frame_times, confidences, linewidth=2,
                label='Confidence', color='blue')

    # Mark detections
    if np.any(detections):
        detection_times = frame_times[detections]
        detection_confs = confidences[detections]
        axes[plot_idx].scatter(detection_times, detection_confs,
                       color='red', s=50, label='Detection', zorder=3)

    axes[plot_idx].axhline(0.75, color='orange', linestyle='--',
                   linewidth=2, label='Threshold')

    # Overall decision annotation
    decision_text = 'DRONE DETECTED' if results['overall_detected'] else 'NO DRONE'
    decision_color = 'red' if results['overall_detected'] else 'green'
    axes[plot_idx].text(0.02, 0.95, decision_text,
                transform=axes[plot_idx].transAxes,
                fontsize=14, fontweight='bold', color=decision_color,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[plot_idx].set_title('Detection Timeline', fontweight='bold', fontsize=12)
    axes[plot_idx].set_xlabel('Time (s)')
    axes[plot_idx].set_ylabel('Confidence')
    axes[plot_idx].legend(loc='upper right')
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Batch Processing
# ============================================================================

def process_directory(
    dir_path: str,
    config: DetectorConfig,
    output_csv: Optional[str] = None,
    visualize: bool = False,
    plot_dir: Optional[str] = None
) -> List[Dict]:
    """
    Process all audio files in a directory.

    Parameters
    ----------
    dir_path : str
        Path to directory containing audio files
    config : DetectorConfig
        Detection configuration
    output_csv : str, optional
        Path to save results CSV
    visualize : bool
        Generate plots for each file
    plot_dir : str, optional
        Directory to save plots

    Returns
    -------
    results_list : list of dict
        Results for all files
    """
    # Find all WAV files
    audio_files = sorted(Path(dir_path).glob('*.wav'))

    if len(audio_files) == 0:
        print(f"No WAV files found in {dir_path}")
        return []

    print(f"\nFound {len(audio_files)} audio files in {dir_path}")

    results_list = []

    for i, file_path in enumerate(audio_files):
        print(f"\n[{i+1}/{len(audio_files)}] Processing {file_path.name}...")

        try:
            results = process_audio_file(str(file_path), config)
            results_list.append(results)

            # Optional visualization
            if visualize and plot_dir:
                plot_path = os.path.join(plot_dir, f"{file_path.stem}_detection.png")
                plot_detection_results(results, save_path=plot_path)

        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            continue

    # Save CSV if requested
    if output_csv and len(results_list) > 0:
        save_results_csv(results_list, output_csv)

    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total files processed: {len(results_list)}")

    num_detected = sum(1 for r in results_list if r['overall_detected'])
    print(f"Drones detected: {num_detected} ({num_detected/len(results_list)*100:.1f}%)")

    avg_confidence = np.mean([r['mean_confidence'] for r in results_list])
    print(f"Average confidence: {avg_confidence:.3f}")

    avg_snr = np.mean([r['mean_snr_db'] for r in results_list])
    print(f"Average SNR: {avg_snr:.1f} dB")

    return results_list


def save_results_csv(results_list: List[Dict], output_path: str):
    """
    Save results to CSV file.

    Parameters
    ----------
    results_list : list of dict
        Results from batch processing
    output_path : str
        Path to output CSV file
    """
    import csv

    # Select fields to save (exclude large arrays)
    fields = [
        'file_name', 'duration_sec', 'overall_detected',
        'detection_rate', 'num_detections', 'num_frames',
        'mean_confidence', 'max_confidence', 'std_confidence',
        'mean_snr_db', 'max_snr_db',
        'mean_harmonic_score', 'mean_valid_harmonics',
        'estimated_f0', 'snr_improvement_db',
        'processing_time_sec'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in results_list:
            row = {k: result[k] for k in fields if k in result}
            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Acoustic Drone Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', type=str,
                           help='Process single audio file')
    input_group.add_argument('--dir', type=str,
                           help='Process all WAV files in directory')
    input_group.add_argument('--batch', action='store_true',
                           help='Process all classes in data/train or data/val')

    # Batch processing options
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                       help='Data split for batch processing (default: train)')
    parser.add_argument('--classes', nargs='+',
                       default=['drone', 'helicopter', 'background'],
                       help='Classes to process (default: all)')

    # Output options
    parser.add_argument('--output', type=str,
                       help='Output CSV file for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-plots', type=str,
                       help='Directory to save plots')

    # Detection parameters
    parser.add_argument('--f0', type=float, default=150,
                       help='Expected fundamental frequency (Hz, default: 150)')
    parser.add_argument('--threshold', type=float, default=0.75,
                       help='Confidence threshold (default: 0.75)')
    parser.add_argument('--use-filter', action='store_true',
                       help='Enable harmonic filtering')

    # Stereo/DOA parameters
    parser.add_argument('--mic-spacing', type=float, default=0.14,
                       help='Microphone spacing in meters for stereo DOA estimation (default: 0.14 m / 14 cm). '
                            'Examples: laptop mics ~0.08 m, standalone stereo ~0.14 m')

    # Other options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Create configuration
    config = DetectorConfig(
        detector_f0=args.f0,
        confidence_threshold=args.threshold,
        use_harmonic_filter=args.use_filter,
        mic_spacing_m=args.mic_spacing,
        verbose=not args.quiet
    )

    # Create output directory if needed
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    # Process based on input mode
    if args.file:
        # Single file processing
        results = process_audio_file(args.file, config)

        if args.visualize or args.save_plots:
            if args.save_plots:
                file_stem = Path(args.file).stem
                plot_path = os.path.join(args.save_plots, f"{file_stem}_detection.png")
            else:
                plot_path = None
            plot_detection_results(results, save_path=plot_path)

        if args.output:
            save_results_csv([results], args.output)

    elif args.dir:
        # Directory processing
        results_list = process_directory(
            args.dir, config,
            output_csv=args.output,
            visualize=args.visualize,
            plot_dir=args.save_plots
        )

    elif args.batch:
        # Batch processing all classes
        data_dir = f"data/{args.split}"

        if not os.path.exists(data_dir):
            # Try absolute path
            data_dir = f"/mnt/d/edth/data/{args.split}"
            if not os.path.exists(data_dir):
                print(f"ERROR: Data directory not found: {data_dir}")
                sys.exit(1)

        all_results = []

        for class_name in args.classes:
            class_dir = os.path.join(data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"WARNING: Class directory not found: {class_dir}")
                continue

            print(f"\n{'='*70}")
            print(f"Processing class: {class_name}")
            print(f"{'='*70}")

            # Create class-specific plot directory
            if args.save_plots:
                class_plot_dir = os.path.join(args.save_plots, class_name)
                os.makedirs(class_plot_dir, exist_ok=True)
            else:
                class_plot_dir = None

            results_list = process_directory(
                class_dir, config,
                visualize=args.visualize,
                plot_dir=class_plot_dir
            )

            # Add class label to results
            for r in results_list:
                r['true_class'] = class_name

            all_results.extend(results_list)

        # Save combined results
        if args.output and len(all_results) > 0:
            save_results_csv(all_results, args.output)

        # Print class-wise summary
        print(f"\n{'='*70}")
        print("CLASS-WISE SUMMARY")
        print(f"{'='*70}")

        for class_name in args.classes:
            class_results = [r for r in all_results if r.get('true_class') == class_name]
            if len(class_results) == 0:
                continue

            num_detected = sum(1 for r in class_results if r['overall_detected'])
            avg_conf = np.mean([r['mean_confidence'] for r in class_results])

            print(f"\n{class_name}:")
            print(f"  Files: {len(class_results)}")
            print(f"  Detected: {num_detected} ({num_detected/len(class_results)*100:.1f}%)")
            print(f"  Avg confidence: {avg_conf:.3f}")


if __name__ == '__main__':
    main()
