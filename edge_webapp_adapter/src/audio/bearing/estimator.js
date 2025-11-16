// src/audio/bearing/estimator.js

console.log("[Bearing] estimator.js script loading (non-module)...");

// All dependencies are expected to be attached on window.*
// from types.js, framingWindowing.js, fft.js, gccPhatDoa.js, bearingTransform.js
var frameAndWindow = window.frameAndWindow;
var computeFftPerFrame = window.computeFftPerFrame;
var gccPhatPair = window.gccPhatPair;
var tdoaToDoaLinear = window.tdoaToDoaLinear;
var SOUND_SPEED = window.SOUND_SPEED;
var localToGlobalBearing = window.localToGlobalBearing;
var createInitialBearingState = window.createInitialBearingState;

/**
 * @typedef {import("./types.js").MicGeometry} MicGeometry
 * @typedef {import("./types.js").BearingState} BearingState
 * @typedef {import("./types.js").EstimateBearingResult} EstimateBearingResult
 *
 * MultiChannelChunk: [nSamples][nMics], each sample is number.
 * @typedef {number[][]} MultiChannelChunk
 */

/**
 * Ensure shape [nSamples][nMics].
 * For now we assume audioChunk already in this shape.
 *
 * @param {MultiChannelChunk} chunk
 * @returns {MultiChannelChunk}
 */
function normalizeAudioShape(chunk) {
    if (!chunk || chunk.length === 0) return chunk;

    const nSamples = chunk.length;
    const nMics = chunk[0].length;

    // If caller gave [nMics][nSamples] style and nMics <=4, we could transpose.
    // For now, assume [nSamples][nMics].
    if (nMics <= 4) {
        return chunk;
    }

    // If needed: transpose here. Not used right now.
    return chunk;
}

/**
 * Frame a multi-channel signal: [nSamples][nMics]
 * → [nFrames][frameLength][nMics]
 *
 * @param {MultiChannelChunk} x
 * @param {number} fs
 * @param {number} [frameLengthMs=32.0]
 * @param {number} [hopLengthMs=16.0]
 * @param {"hann"|"hamming"|"blackman"|"rectangular"} [window="hann"]
 * @returns {Float32Array[][]} FrameTensor
 */
function frameMultiChannel(
    x,
    fs,
    frameLengthMs = 32.0,
    hopLengthMs = 16.0,
    window = "hann"
) {
    const nSamples = x.length;
    if (nSamples === 0) return [];

    const nMics = x[0].length;
    if (nMics < 2) {
        throw new Error("Need at least 2 mics for bearing estimation");
    }

    /** @type {Float32Array[][]} */
    const framesPerCh = [];
    let minFrames = null;

    for (let ch = 0; ch < nMics; ch++) {
        const mono = new Float32Array(nSamples);
        for (let i = 0; i < nSamples; i++) {
            mono[i] = x[i][ch];
        }

        const { frames } = frameAndWindow(mono, fs, {
            frameLengthMs,
            hopLengthMs,
            windowType: window,
            zeroPad: false,
            normalizeEnergy: false,
        });

        if (minFrames === null) minFrames = frames.length;
        else minFrames = Math.min(minFrames, frames.length);

        framesPerCh.push(frames);
    }

    if (minFrames === null || minFrames === 0) return [];

    /** @type {Float32Array[][]} */
    const frames = [];

    for (let i = 0; i < minFrames; i++) {
        const frameLength = framesPerCh[0][i].length;
        /** @type {Float32Array[]} */
        const frame = [];
        for (let j = 0; j < frameLength; j++) {
            const sample = new Float32Array(nMics);
            for (let ch = 0; ch < nMics; ch++) {
                sample[ch] = framesPerCh[ch][i][j];
            }
            frame.push(sample);
        }
        frames.push(frame);
    }

    return frames;
}

/**
 * Energy-based frame selection using magnitude spectra.
 *
 * @param {Float32Array[]} magnitude
 * @param {number} [topK=8]
 * @returns {number[]} indices
 */
function selectActiveFramesFromSpectra(magnitude, topK = 8) {
    const nFrames = magnitude.length;
    if (nFrames === 0) return [];

    const energy = new Float32Array(nFrames);
    for (let i = 0; i < nFrames; i++) {
        const mag = magnitude[i];
        let sum = 0;
        for (let k = 0; k < mag.length; k++) {
            sum += mag[k];
        }
        energy[i] = mag.length > 0 ? sum / mag.length : 0;
    }

    // median threshold
    const sorted = Array.from(energy).sort((a, b) => a - b);
    const mid = sorted[Math.floor(sorted.length / 2)];

    /** @type {number[]} */
    const active = [];
    for (let i = 0; i < nFrames; i++) {
        if (energy[i] >= mid) active.push(i);
    }
    if (active.length === 0) return [];

    // sort active by descending energy
    active.sort((a, b) => energy[b] - energy[a]);

    return active.slice(0, Math.min(topK, active.length));
}

/**
 * Single-node bearing estimation (empty DSP skeleton).
 *
 * @param {MultiChannelChunk} audioChunk - [nSamples][nMics]
 * @param {number} fs                   - sample rate (e.g. 48000)
 * @param {MicGeometry} micGeometry
 * @param {BearingState} [state]
 * @returns {EstimateBearingResult}
 */
function estimateBearing(audioChunk, fs, micGeometry, state) {
    /** @type {BearingState} */
    let s = state ?? createInitialBearingState();

    // 1) Basic input shape diagnostics
    if (!audioChunk || audioChunk.length === 0) {
        console.log("[Bearing] audioChunk empty");
        return { bearingDeg: null, confidence: 0, state: s };
    }

    const firstRow = audioChunk[0];
    const nSamplesIn = audioChunk.length;
    const nMicsIn = Array.isArray(firstRow) ? firstRow.length : "unknown";

    console.log(
        "[Bearing] audioChunk (raw) shape nSamples=",
        nSamplesIn,
        "nMics=",
        nMicsIn
    );

    const x = normalizeAudioShape(audioChunk);

    if (!x || x.length === 0 || !Array.isArray(x[0]) || x[0].length < 2) {
        console.log(
            "[Bearing] after normalize: invalid shape, nSamples=",
            x ? x.length : 0,
            "nMics=",
            x && Array.isArray(x[0]) ? x[0].length : "unknown"
        );
        return { bearingDeg: null, confidence: 0, state: s };
    }

    const nSamplesNorm = x.length;
    const nMicsNorm = x[0].length;
    console.log(
        "[Bearing] normalized shape nSamples=",
        nSamplesNorm,
        "nMics=",
        nMicsNorm
    );

    // 2) Multi-channel framing
    const frames = frameMultiChannel(
        x,
        fs,
        32.0,   // frameLengthMs
        16.0,   // hopLengthMs
        "hann" // window
    );

    const nFrames = frames ? frames.length : 0;
    console.log("[Bearing] frames.length =", nFrames);

    if (!frames || frames.length === 0) {
        console.log("[Bearing] no frames produced → returning null");
        return { bearingDeg: null, confidence: 0, state: s };
    }

    // 3) Build mono frames for FFT (reference channel = 0)
    /** @type {Float32Array[]} */
    const monoFrames = frames.map((frame, idx) => {
        const frameLength = frame.length;
        const out = new Float32Array(frameLength);
        for (let i = 0; i < frameLength; i++) {
            out[i] = frame[i][0];
        }
        return out;
    });

    const { magnitude } = computeFftPerFrame(monoFrames, fs);

    const nFramesMag = magnitude.length;
    const nBinsMag = nFramesMag > 0 && magnitude[0] ? magnitude[0].length : 0;
    console.log(
        "[Bearing] magnitude frames=",
        nFramesMag,
        "bins=",
        nBinsMag
    );

    const activeIdx = selectActiveFramesFromSpectra(magnitude, 8);
    console.log("[Bearing] activeIdx.length =", activeIdx.length);

    if (activeIdx.length === 0) {
        console.log("[Bearing] no active frames → returning null");
        return { bearingDeg: null, confidence: 0, state: s };
    }

    // Resolve mic spacing with explicit null checks (no ?? to support older iOS)
    let micSpacing = 0.15;
    if (micGeometry && micGeometry.mic_spacing_m != null) {
        micSpacing = micGeometry.mic_spacing_m;
    } else if (micGeometry && micGeometry.d != null) {
        micSpacing = micGeometry.d;
    }

    // Resolve heading (deg) with explicit null checks
    let heading = 0.0;
    if (micGeometry && micGeometry.heading_deg != null) {
        heading = micGeometry.heading_deg;
    } else if (micGeometry && micGeometry.orientation_deg != null) {
        heading = micGeometry.orientation_deg;
    }
    const maxTau = micSpacing / SOUND_SPEED;

    /** @type {number[]} */
    const bearings = [];

    for (const idx of activeIdx) {
        const frame = frames[idx];
        const frameLength = frame.length;

        const sig0 = new Float32Array(frameLength);
        const sig1 = new Float32Array(frameLength);
        for (let i = 0; i < frameLength; i++) {
            sig0[i] = frame[i][0];
            sig1[i] = frame[i][1];
        }

        let tau;
        try {
            tau = gccPhatPair(sig0, sig1, fs, maxTau, 16);
        } catch (err) {
            console.warn("[Bearing] gccPhatPair error on frame", idx, err);
            continue;
        }

        const thetaRad = tdoaToDoaLinear(tau, micSpacing, SOUND_SPEED);
        const thetaDegLocal = (thetaRad * 180) / Math.PI;
        const bearingGlobal = localToGlobalBearing(thetaDegLocal, heading);

        if (!Number.isNaN(bearingGlobal)) {
            bearings.push(bearingGlobal);
        }
    }

    console.log("[Bearing] bearings.length =", bearings.length);

    if (bearings.length === 0) {
        console.log("[Bearing] no valid bearings → returning null");
        return { bearingDeg: null, confidence: 0, state: s };
    }

    const raw =
        bearings.reduce((acc, v) => acc + v, 0) / Math.max(1, bearings.length);

    s.lastBearings.push(raw);
    s.lastConfidences.push(1.0);

    if (s.lastBearings.length > s.maxHistory) {
        s.lastBearings = s.lastBearings.slice(-s.maxHistory);
        s.lastConfidences = s.lastConfidences.slice(-s.maxHistory);
    }

    const smooth =
        s.lastBearings.reduce((acc, v) => acc + v, 0) /
        Math.max(1, s.lastBearings.length);

    const confidence = Math.min(1, bearings.length / activeIdx.length);

    console.log(
        "[Bearing] RESULT bearingDeg=",
        smooth,
        "confidence=",
        confidence
    );

    return {
        bearingDeg: smooth,
        confidence,
        state: s,
    };
}


/**
 * Small helper so app.legacy.js can be used.
 *
 * @returns {BearingState}
 */
function createBearingState() {
    return createInitialBearingState();
}


// Expose functions on window for legacy app.legacy.js
// so maybeComputeBearing() can call window.estimateBearing(...)
if (typeof window !== "undefined") {
    // attach to global for legacy script
    // eslint-disable-next-line no-undef
    window.estimateBearing = estimateBearing;
    // eslint-disable-next-line no-undef
    window.createBearingState = createBearingState;
    // mark for debugging
    // eslint-disable-next-line no-undef
    window._bearingEstimatorLoaded = true;

    console.log("[Bearing] estimator.js: attached estimateBearing to window");
}

