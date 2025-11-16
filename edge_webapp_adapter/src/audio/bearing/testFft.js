// src/audio/bearing/testFft.js
import { computeFftPerFrame } from "./fft.js";

(function testFft() {
    const fs = 48000;
    const durationSec = 0.032;           // 32 ms
    const frameLength = Math.round(fs * durationSec); // ≈ 1536
    const freq = 1000;                   // 1 kHz test tone

    const frame = new Float32Array(frameLength);
    for (let n = 0; n < frameLength; n++) {
        frame[n] = Math.sin(2 * Math.PI * freq * (n / fs));
    }

    const frames = [frame];

    const result = computeFftPerFrame(frames, fs, { nfft: frameLength });

    console.log("=== FFT test result ===");
    console.log("freqs length:", result.freqs.length);
    console.log("magnitude[0] length:", result.magnitude[0].length);

    // 가장 큰 peak가 어디 있는지
    const mags = result.magnitude[0];
    let maxIdx = 0;
    for (let k = 1; k < mags.length; k++) {
        if (mags[k] > mags[maxIdx]) maxIdx = k;
    }
    console.log("peak bin index:", maxIdx);
    console.log("peak freq (Hz):", result.freqs[maxIdx]);
})();
