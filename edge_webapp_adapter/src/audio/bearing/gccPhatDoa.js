// src/audio/bearing/gccPhatDoa.js

var SOUND_SPEED = 343.0;

function isPowerOfTwo(n) {
    return (n & (n - 1)) === 0 && n > 0;
}

function nextPowerOfTwo(n) {
    let p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

/**
 * 복소 DFT / IDFT (naive, O(N^2))
 *
 * inRe, inIm  -> 입력 신호
 * outRe, outIm -> 결과
 * inverse = false  -> DFT
 * inverse = true   -> IDFT (1/N 스케일링 포함)
 *
 * @param {Float64Array} inRe
 * @param {Float64Array} inIm
 * @param {Float64Array} outRe
 * @param {Float64Array} outIm
 * @param {boolean} inverse
 */
function dft(inRe, inIm, outRe, outIm, inverse = false) {
    const N = inRe.length;
    if (!isPowerOfTwo(N)) {
        throw new Error(`dft(FFT): length must be power of two, got ${N}`);
    }

    const real = new Float64Array(N);
    const imag = new Float64Array(N);

    // Copy input into working buffers
    for (let i = 0; i < N; i++) {
        real[i] = inRe[i];
        imag[i] = inIm[i];
    }

    // Bit-reversal permutation
    let j = 0;
    for (let i = 0; i < N; i++) {
        if (i < j) {
            const tmpRe = real[i];
            const tmpIm = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tmpRe;
            imag[j] = tmpIm;
        }
        let bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Iterative Cooley–Tukey FFT
    const sign = inverse ? 1.0 : -1.0;
    for (let size = 2; size <= N; size <<= 1) {
        const halfSize = size >> 1;
        const theta = (sign * 2.0 * Math.PI) / size;
        const wtempReBase = Math.cos(theta);
        const wtempImBase = Math.sin(theta);

        for (let start = 0; start < N; start += size) {
            let wRe = 1.0;
            let wIm = 0.0;

            for (let k = 0; k < halfSize; k++) {
                const iTop = start + k;
                const iBot = iTop + halfSize;

                const tRe = wRe * real[iBot] - wIm * imag[iBot];
                const tIm = wRe * imag[iBot] + wIm * real[iBot];

                real[iBot] = real[iTop] - tRe;
                imag[iBot] = imag[iTop] - tIm;
                real[iTop] += tRe;
                imag[iTop] += tIm;

                const tmpRe = wRe * wtempReBase - wIm * wtempImBase;
                const tmpIm = wRe * wtempImBase + wIm * wtempReBase;
                wRe = tmpRe;
                wIm = tmpIm;
            }
        }
    }

    const norm = inverse ? 1.0 / N : 1.0;
    for (let i = 0; i < N; i++) {
        outRe[i] = real[i] * norm;
        outIm[i] = imag[i] * norm;
    }
}

/**
 * GCC-PHAT으로 두 신호 x,y 사이의 TDOA (초 단위)를 추정.
 *
 * Python gcc_phat_pair(x, y, fs, max_tau, interp=16) 에 대응.
 * 여기서는 일단 interp는 무시하고 (1 샘플 해상도), 정확도 위주로 구현.
 *
 * @param {Float32Array} x
 * @param {Float32Array} y
 * @param {number} fs
 * @param {number} maxTau   // 허용 TDOA 최대 절대값 (초)
 * @param {number} [interp] // 현재 구현에서는 사용하지 않음 (해상도=1샘플)
 * @returns {number} tau (seconds)
 */
function gccPhatPair(
    x,
    y,
    fs,
    maxTau,
    interp = 1,
) {
    if (x.length !== y.length) {
        throw new Error(
            `gccPhatPair: signal length mismatch: ${x.length} vs ${y.length}`,
        );
    }

    const n = x.length;
    if (n === 0) {
        return 0.0;
    }

    // FFT 크기: 2 * n (zero-padding)
    const nfft = nextPowerOfTwo(2 * n);

    // 실수 입력을 복소 버퍼에 복사 (나머지는 0패딩)
    const xr = new Float64Array(nfft);
    const xi = new Float64Array(nfft);
    const yr = new Float64Array(nfft);
    const yi = new Float64Array(nfft);

    for (let i = 0; i < n; i++) {
        xr[i] = x[i];
        xi[i] = 0.0;
        yr[i] = y[i];
        yi[i] = 0.0;
    }

    // X = DFT(x), Y = DFT(y)
    const Xre = new Float64Array(nfft);
    const Xim = new Float64Array(nfft);
    const Yre = new Float64Array(nfft);
    const Yim = new Float64Array(nfft);

    dft(xr, xi, Xre, Xim, false);
    dft(yr, yi, Yre, Yim, false);

    // 크로스 스펙트럼 G = X * conj(Y) / |X * conj(Y)| (PHAT weighting)
    const Gre = new Float64Array(nfft);
    const Gim = new Float64Array(nfft);
    const eps = 1e-12;

    for (let k = 0; k < nfft; k++) {
        // X * conj(Y) = (Xr + jXi)*(Yr - jYi)
        const rRe = Xre[k] * Yre[k] + Xim[k] * Yim[k];
        const rIm = Xim[k] * Yre[k] - Xre[k] * Yim[k];

        const mag = Math.hypot(rRe, rIm) + eps;
        Gre[k] = rRe / mag;
        Gim[k] = rIm / mag;
    }

    // R = IDFT(G)  -> 크로스 코릴레이션 r[lag]
    const Rre = new Float64Array(nfft);
    const Rim = new Float64Array(nfft);
    dft(Gre, Gim, Rre, Rim, true); // inverse

    // 허용 lag 범위 (샘플 단위)
    const maxLagSamples = Math.min(
        Math.floor(Math.abs(maxTau) * fs),
        n - 1,
    );

    let bestLag = 0;
    let bestVal = -Infinity;

    // lag ∈ [-maxLagSamples, +maxLagSamples] 안에서 최대값 찾기
    for (let lag = -maxLagSamples; lag <= maxLagSamples; lag++) {
        // lag를 0..nfft-1 인덱스로 wrap
        const idx = (lag + nfft) % nfft;
        const v = Rre[idx]; // 이론상 실수여야 함

        if (v > bestVal) {
            bestVal = v;
            bestLag = lag;
        }
    }

    // NOTE:
    // We define tau > 0 when y is delayed relative to x (y arrives later than x).
    // The cross-correlation search above gives bestLag in the opposite sign
    // for our convention, so we flip the sign here.
    const tau = -bestLag / fs; // seconds
    return tau;
}

/**
 * TDOA -> 선형 2-mic array 기준 DOA (radians)
 *
 * Python tdoa_to_doa_linear 에 대응
 *
 * @param {number} tau
 * @param {number} d
 * @param {number} [c]
 * @returns {number} radians
 */
function tdoaToDoaLinear(
    tau,
    d,
    c = SOUND_SPEED,
) {
    if (d <= 0) {
        throw new Error(`tdoaToDoaLinear: d must be > 0, got ${d}`);
    }
    const arg = (c * tau) / d;
    const clipped = Math.max(-1, Math.min(1, arg));
    return Math.asin(clipped); // radians
}

// attach to window for non-module environments
if (typeof window !== "undefined") {
    window.frameAndWindow = frameAndWindow;
}
