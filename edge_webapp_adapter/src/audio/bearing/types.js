// src/audio/bearing/types.js

// JSDoc typedefs for editor/TS tooling only. No runtime logic depends on these.
/**
 * @typedef {Float32Array} SampleVec          // one time step, nMics long
 * @typedef {Float32Array[]} MonoFrames      // [nFrames][frameLength]
 * @typedef {number[][]} MultiChannelChunk   // [nSamples][nMics]
 * @typedef {SampleVec[]} Frame              // [frameLength][nMics]
 * @typedef {Frame[]} FrameTensor            // [nFrames][frameLength][nMics]
 */

/**
 * @typedef {Object} MicGeometry
 * @property {number} [mic_spacing_m]    // primary
 * @property {number} [d]                // fallback
 * @property {number} [heading_deg]      // primary
 * @property {number} [orientation_deg]  // fallback
 */

/**
 * @typedef {Object} BearingState
 * @property {number[]} lastBearings
 * @property {number[]} lastConfidences
 * @property {number}  maxHistory
 */

/**
 * @typedef {Object} EstimateBearingResult
 * @property {number|null} bearingDeg
 * @property {number}      confidence
 * @property {BearingState} state
 */

/**
 * Create a fresh BearingState.
 * @returns {BearingState}
 */
function createInitialBearingState() {
    return {
        lastBearings: [],
        lastConfidences: [],
        maxHistory: 10,
    };
}

