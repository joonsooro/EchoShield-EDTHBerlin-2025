// src/audio/bearing/testBearing.js

import { estimateBearing } from "./estimator.js";

/**
 * nSamples x nMics 형태 dummy 신호 만들기
 * 여기선 1초짜리 48kHz, 2채널 zero 신호
 */
function makeDummyChunk(nSamples = 48000, nMics = 2) {
    const chunk = new Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
        const vec = new Float32Array(nMics);
        // 일단 0으로 채워둠 (무음)
        vec[0] = 0.0;
        vec[1] = 0.0;
        chunk[i] = vec;
    }
    return chunk;
}

function runTest() {
    const fs = 48000;
    const dummyChunk = makeDummyChunk(fs, 2);

    const micGeometry = {
        mic_spacing_m: 0.15,
        heading_deg: 0,
    };

    const result = estimateBearing(dummyChunk, fs, micGeometry);

    console.log("=== Bearing test result ===");
    console.log("bearingDeg:", result.bearingDeg);
    console.log("confidence:", result.confidence);
    console.log("state:", result.state);
}

// 브라우저 로드 시 자동 실행
runTest();
