// src/audio/bearing/fft.js

/**
 * @typedef {Float32Array[]} MonoFrames  // [nFrames][frameLength]
 */

/**
 * @typedef {Object} ComputeFftPerFrameOptions
 * @property {number} [nfft]      // FFT size (>= frameLength), optional
 * @property {boolean} [removeDc] // default: true
 */

/**
 * @typedef {Object} ComputeFftPerFrameResult
 * @property {Float32Array} freqs           // [nFreqs]
 * @property {Float32Array[]} spectrum      // [nFrames][nFreqs*2] (re,im interleaved)
 * @property {Float32Array[]} magnitude     // [nFrames][nFreqs]
 * @property {Float32Array[]} magnitudeDb   // [nFrames][nFreqs]
 */

// ---------------------- 내부 유틸 함수들 ------------------------

/**
 * power-of-two 여부 체크
 * @param {number} n
 * @returns {boolean}
 */
function isPowerOfTwo(n) {
    return (n & (n - 1)) === 0 && n > 0;
}

/**
 * radix-2 Cooley–Tukey FFT (in-place)
 * re, im: 길이 n (power of two)
 * @param {Float64Array} re
 * @param {Float64Array} im
 */
function fftRadix2(re, im) {
    const n = re.length;
    if (!isPowerOfTwo(n)) {
        throw new Error(`fftRadix2: n=${n} is not power of two`);
    }

    // bit-reversal
    let j = 0;
    for (let i = 0; i < n; i++) {
        if (i < j) {
            const tmpRe = re[i];
            const tmpIm = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tmpRe;
            im[j] = tmpIm;
        }
        let m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // stages
    for (let len = 2; len <= n; len <<= 1) {
        const ang = -2 * Math.PI / len;
        const wLenRe = Math.cos(ang);
        const wLenIm = Math.sin(ang);

        for (let i = 0; i < n; i += len) {
            let wRe = 1.0;
            let wIm = 0.0;

            const half = len >> 1;
            for (let k = 0; k < half; k++) {
                const uRe = re[i + k];
                const uIm = im[i + k];

                const tRe = re[i + k + half];
                const tIm = im[i + k + half];

                const vRe = tRe * wRe - tIm * wIm;
                const vIm = tRe * wIm + tIm * wRe;

                re[i + k] = uRe + vRe;
                im[i + k] = uIm + vIm;

                re[i + k + half] = uRe - vRe;
                im[i + k + half] = uIm - vIm;

                const tmpRe = wRe * wLenRe - wIm * wLenIm;
                const tmpIm = wRe * wLenIm + wIm * wLenRe;
                wRe = tmpRe;
                wIm = tmpIm;
            }
        }
    }
}

/**
 * if not power-of-two, use naive DFT (O(N^2). Slow but provides actual value)
 * @param {Float64Array} outRe
 * @param {Float64Array} outIm
 * @param {Float64Array} inRe
 */
function dftNaive(outRe, outIm, inRe) {
    const n = inRe.length;
    for (let k = 0; k < n; k++) {
        let sumRe = 0.0;
        let sumIm = 0.0;
        const twoPiOverN = -2 * Math.PI * k / n;
        for (let t = 0; t < n; t++) {
            const angle = twoPiOverN * t;
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            const x = inRe[t];
            sumRe += x * c;
            sumIm += x * s;
        }
        outRe[k] = sumRe;
        outIm[k] = sumIm;
    }
}


/**
 * FFT on the real input (forward)
 * - input: real time-domain (Float32Array)
 * - output: re, im (Float64Array)
 *
 * @param {Float32Array} buf  // 길이 nfft
 * @param {number} nfft
 * @returns {{ re: Float64Array, im: Float64Array }}
 */
function realFft(buf, nfft) {
    const n = nfft;
    const re = new Float64Array(n);
    const im = new Float64Array(n);

    // copy + zero imaginary
    for (let i = 0; i < n; i++) {
        re[i] = buf[i];
        im[i] = 0.0;
    }

    if (isPowerOfTwo(n)) {
        fftRadix2(re, im);
    } else {
        // 느리지만 동작하는 fallback
        dftNaive(re, im, re.slice());
    }

    return { re, im };
}

// ---------------------- 메인 API ------------------------

/**
 * JS version that reacts to the Python compute_fft_per_frame
 *
 * frames: [nFrames][frameLength]
 *
 * @param {MonoFrames} frames
 * @param {number} fs
 * @param {ComputeFftPerFrameOptions} [options]
 * @returns {ComputeFftPerFrameResult}
 */
function computeFftPerFrame(frames, fs, options = {}) {
    const removeDc = options.removeDc ?? true;

    const nFrames = frames.length;
    if (nFrames === 0) {
        return {
            freqs: new Float32Array(0),
            spectrum: [],
            magnitude: [],
            magnitudeDb: [],
        };
    }

    const frameLength = frames[0].length;
    let nfft = options.nfft ?? frameLength;

    if (nfft < frameLength) {
        throw new Error(
            `computeFftPerFrame: nfft (${nfft}) must be >= frameLength (${frameLength})`,
        );
    }

    const nFreqs = Math.floor(nfft / 2) + 1;

    // frequency bins (0 .. fs/2)
    const freqs = new Float32Array(nFreqs);
    for (let k = 0; k < nFreqs; k++) {
        freqs[k] = (fs * k) / nfft;
    }

    /** @type {Float32Array[]} */
    const spectrum = [];
    /** @type {Float32Array[]} */
    const magnitude = [];
    /** @type {Float32Array[]} */
    const magnitudeDb = [];

    const eps = 1e-12;

    for (let i = 0; i < nFrames; i++) {
        const frame = frames[i];
        if (frame.length !== frameLength) {
            throw new Error(
                `computeFftPerFrame: inconsistent frame length at index ${i}: ` +
                `${frame.length} vs ${frameLength}`,
            );
        }

        // zero-padded buffer
        const buf = new Float32Array(nfft);
        buf.set(frame);

        if (removeDc) {
            let mean = 0.0;
            for (let j = 0; j < frameLength; j++) {
                mean += buf[j];
            }
            mean /= frameLength;
            for (let j = 0; j < frameLength; j++) {
                buf[j] -= mean;
            }
        }

        const { re, im } = realFft(buf, nfft);

        const spec = new Float32Array(nFreqs * 2);
        const mag = new Float32Array(nFreqs);
        const magDb = new Float32Array(nFreqs);

        for (let k = 0; k < nFreqs; k++) {
            const r = re[k];
            const imv = im[k];

            spec[2 * k] = r;
            spec[2 * k + 1] = imv;

            const m = Math.hypot(r, imv);
            mag[k] = m;
            magDb[k] = 20.0 * Math.log10(m + eps);
        }

        spectrum.push(spec);
        magnitude.push(mag);
        magnitudeDb.push(magDb);
    }

    return { freqs, spectrum, magnitude, magnitudeDb };
}
