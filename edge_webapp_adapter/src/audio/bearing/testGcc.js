// src/audio/bearing/testGcc.js (대체 버전 예시)
import { gccPhatPair, tdoaToDoaLinear, SOUND_SPEED } from "./gccPhatDoa.js";

function xcorrNaive(x, y, maxLagSamples) {
    const n = x.length;
    let bestLag = 0;
    let bestVal = -Infinity;

    for (let lag = -maxLagSamples; lag <= maxLagSamples; lag++) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
            const j = i - lag;
            if (j >= 0 && j < n) {
                sum += x[i] * y[j];
            }
        }
        if (sum > bestVal) {
            bestVal = sum;
            bestLag = lag;
        }
    }
    return { bestLag, bestVal };
}

(function testXcorrAndGccNoise() {
    const fs = 48000;
    const durationSec = 0.05;
    const n = Math.round(fs * durationSec);

    const d = 0.15;
    const c = SOUND_SPEED;
    const targetThetaDeg = 30;
    const targetThetaRad = (targetThetaDeg * Math.PI) / 180;

    const trueTau = (d * Math.sin(targetThetaRad)) / c;
    const delaySamples = Math.round(trueTau * fs);

    const x = new Float32Array(n);
    const y = new Float32Array(n);

    // broadband noise
    for (let i = 0; i < n; i++) {
        const noise = (Math.random() * 2 - 1); // [-1,1] uniform
        x[i] = noise;

        const j = i - delaySamples;
        y[i] = j >= 0 ? x[j] : 0;  // delayed copy
    }

    const maxLagSamples = 30;
    const { bestLag } = xcorrNaive(x, y, maxLagSamples);

    console.log("=== Naive XCorr (noise) ===");
    console.log("targetThetaDeg:", targetThetaDeg);
    console.log("trueTau       (s):", trueTau);
    console.log("true lag      (samples):", delaySamples);
    console.log("naive bestLag (samples):", bestLag);

    const maxTau = maxLagSamples / fs;
    const estTau = gccPhatPair(x, y, fs, maxTau);
    const estLag = estTau * fs;
    const thetaRad = tdoaToDoaLinear(estTau, d, SOUND_SPEED);
    const thetaDeg = (thetaRad * 180) / Math.PI;

    console.log("=== GCC-PHAT (noise) ===");
    console.log("estTau   (s):", estTau);
    console.log("est lag  (samples):", estLag);
    console.log("est theta(deg):", thetaDeg);
})();
