// src/audio/bearing/framingWindowing.js

/**
 * @typedef {Float32Array[]} MonoFrames  // [nFrames][frameLength]
 */

/**
 * @typedef {Object} FrameAndWindowOptions
 * @property {number} [frameLengthMs]  // default 32.0
 * @property {number} [hopLengthMs]    // default 16.0
 * @property {"hann"|"hamming"|"blackman"|"rectangular"} [windowType]
 * @property {boolean} [zeroPad]
 * @property {boolean} [normalizeEnergy]
 */

/**
 * @typedef {Object} FrameAndWindowResult
 * @property {MonoFrames} frames
 * @property {Float32Array} frameTimes
 */

/**
 * Hann / Hamming / Blackman / Rectangular window generator
 *
 * @param {string} windowType
 * @param {number} length
 * @returns {Float32Array}
 */
function getWindow(windowType, length) {
    if (length <= 0) {
        throw new Error(`getWindow: window length must be > 0, got ${length}`);
    }

    const win = new Float32Array(length);
    const t = windowType.toLowerCase().trim();

    if (t === "hann") {
        // np.hanning(N) equivalent
        if (length === 1) {
            win[0] = 1.0;
        } else {
            const N = length;
            for (let n = 0; n < N; n++) {
                win[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1));
            }
        }
    } else if (t === "hamming") {
        // np.hamming(N)
        if (length === 1) {
            win[0] = 1.0;
        } else {
            const N = length;
            for (let n = 0; n < N; n++) {
                win[n] = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (N - 1));
            }
        }
    } else if (t === "blackman") {
        // np.blackman(N)
        if (length === 1) {
            win[0] = 1.0;
        } else {
            const N = length;
            for (let n = 0; n < N; n++) {
                const a0 = 0.42;
                const a1 = 0.5;
                const a2 = 0.08;
                const phase = (2 * Math.PI * n) / (N - 1);
                win[n] =
                    a0 -
                    a1 * Math.cos(phase) +
                    a2 * Math.cos(2 * phase);
            }
        }
    } else if (t === "rectangular" || t === "rect") {
        for (let n = 0; n < length; n++) {
            win[n] = 1.0;
        }
    } else {
        throw new Error(
            `getWindow: unknown window type '${windowType}'.` +
            ` Supported: 'hann', 'hamming', 'blackman', 'rectangular'`
        );
    }

    return win;
}

/**
 * Very minimal sampling-rate sanity check (less strict than Python).
 *
 * @param {number} fs
 * @returns {number}
 */
function validateSamplingRate(fs) {
    if (!Number.isFinite(fs) || fs <= 0) {
        throw new Error(`frameAndWindow: sampling rate must be > 0, got ${fs}`);
    }
    // mandatory check
    return fs;
}

/**
 * Convert frame_length_ms / hop_length_ms → samples
 *
 * @param {number} frameLengthMs
 * @param {number} hopLengthMs
 * @param {number} fs
 * @returns {{ frameLength: number, hopLength: number }}
 */
function computeFrameParams(frameLengthMs, hopLengthMs, fs) {
    if (frameLengthMs <= 0) {
        throw new Error(`frameAndWindow: frameLengthMs must be > 0, got ${frameLengthMs}`);
    }
    if (hopLengthMs <= 0) {
        throw new Error(`frameAndWindow: hopLengthMs must be > 0, got ${hopLengthMs}`);
    }

    const frameLength = Math.round((fs * frameLengthMs) / 1000);
    const hopLength = Math.round((fs * hopLengthMs) / 1000);

    if (frameLength <= 0) {
        throw new Error(
            `frameAndWindow: frameLengthMs=${frameLengthMs} at fs=${fs} gives frameLength=0`
        );
    }
    if (hopLength <= 0) {
        throw new Error(
            `frameAndWindow: hopLengthMs=${hopLengthMs} at fs=${fs} gives hopLength=0`
        );
    }

    if (hopLength > frameLength) {
        console.warn(
            `frameAndWindow: hopLength (${hopLength}) > frameLength (${frameLength}).` +
            ` Gaps between frames may miss transient events.`
        );
    }

    const overlapRatio = 1.0 - hopLength / frameLength;
    if (overlapRatio > 0.9) {
        console.warn(
            `frameAndWindow: very high overlap (${(overlapRatio * 100).toFixed(
                1
            )}%). Computation cost may be high.`
        );
    }

    return { frameLength, hopLength };
}

/**
 * JS version of Python frame_and_window (mono).
 * Multi-channel: estimator will call per-channel to coordinate 3D tensor.
 *
 * @param {Float32Array} audio
 * @param {number} fs
 * @param {FrameAndWindowOptions} [options]
 * @returns {FrameAndWindowResult}
 */
function frameAndWindow(audio, fs, options = {}) {
    if (!(audio instanceof Float32Array)) {
        // 안전하게 복사 (Float32Array 아닌 경우 대응)
        audio = new Float32Array(audio);
    }

    if (audio.length === 0) {
        return {
            frames: [],
            frameTimes: new Float32Array(0),
        };
    }

    fs = validateSamplingRate(fs);

    const frameLengthMs = options.frameLengthMs ?? 32.0;
    const hopLengthMs = options.hopLengthMs ?? 16.0;
    const windowType = options.windowType ?? "hann";
    const zeroPad = options.zeroPad ?? false;
    const normalizeEnergy = options.normalizeEnergy ?? false;

    const { frameLength, hopLength } = computeFrameParams(
        frameLengthMs,
        hopLengthMs,
        fs
    );

    // === Step 2: optional zero-padding ===
    let padded = audio;
    if (zeroPad) {
        const remainder = (audio.length - frameLength) % hopLength;
        if (remainder > 0) {
            const padLength = hopLength - remainder;
            const tmp = new Float32Array(audio.length + padLength);
            tmp.set(audio, 0);
            // 남은 부분은 0으로 유지
            padded = tmp;
        }
    }

    // if audio length is shorter than 1 frame, in python, it's an error,
    // but in browser streaming, it usually means that "chunk is too short yet"
    // you return empty result
    if (padded.length < frameLength) {
        return {
            frames: [],
            frameTimes: new Float32Array(0),
        };
    }

    // === Step 3: number of frames ===
    const nFrames = 1 + Math.floor((padded.length - frameLength) / hopLength);
    /** @type {MonoFrames} */
    const frames = [];
    const frameTimes = new Float32Array(nFrames);

    // === Step 4: slice audio into overlapping frames ===
    for (let i = 0; i < nFrames; i++) {
        const start = i * hopLength;
        const end = start + frameLength;

        const frame = new Float32Array(frameLength);
        frame.set(padded.subarray(start, end));
        frames.push(frame);

        const center = start + frameLength / 2;
        frameTimes[i] = center / fs;
    }

    // === Step 5: apply window ===
    const window = getWindow(windowType, frameLength);
    for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        for (let j = 0; j < frameLength; j++) {
            frame[j] *= window[j];
        }
    }

    // === Step 6: optional energy normalization ===
    if (normalizeEnergy) {
        for (let i = 0; i < frames.length; i++) {
            const frame = frames[i];
            let energy = 0;
            for (let j = 0; j < frameLength; j++) {
                const v = frame[j];
                energy += v * v;
            }
            const norm = Math.sqrt(energy);
            const denom = norm === 0 ? 1.0 : norm;
            for (let j = 0; j < frameLength; j++) {
                frame[j] /= denom;
            }
        }
    }

    return { frames, frameTimes };
}
