/***** Drone Classifier – App v5 (button-hardened + verbose logs) *****/

var DRONE_CLASS_NAME = "drone";
var ALERT_THRESHOLD = 0.80;          // 80%
var ALERT_COOLDOWN_MS = 3000;        // 3s debounce(Prevent SPAM)
var WEBHOOK_URL = "/webhook/edge"; // use same-origin; ngrok will proxy to adapter
var lastAlertAt = 0;
var SENSOR_NODE_ID = "NODE_IPHONE_01"; // Bind in UI if needed

// Alert state (hysteresis to avoid spam + double posts on jitter)
var prevAbove = false;           // last decision state over threshold
var ALERT_RELEASE_RATIO = 0.8;   // drop-out at 80% of threshold to re-arm

// ===== UI hooks =====
const logEl = document.getElementById('log');
const btn = document.getElementById('btnStart');
const statusPill = document.getElementById('status');
const probsTable = document.getElementById('probsTable');
const probsTbody = probsTable?.querySelector('tbody');

function log(msg) {
  const time = new Date().toISOString().split('T')[1].split('.')[0];
  if (logEl) {
    logEl.textContent += `[${time}] ${msg}\n`;
    logEl.scrollTop = logEl.scrollHeight;
  } else {
    console.log(msg);
  }
}
function setStatus(s) { if (statusPill) statusPill.textContent = s; }

// ===== On-page diagnostics at load =====
(function pageDiag() {
  log("App v5 loaded");
  const diag = {
    protocol: location.protocol,
    hasAudioContext: !!window.AudioContext,
    hasWebkitAudioContext: !!window.webkitAudioContext,
    hasMediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    ua: navigator.userAgent
  };
  diag.origin = location.origin;
  diag.href = location.href;
  log("Support diag: " + JSON.stringify(diag));
})();

// ===== Math/DSP helpers =====
function hannWindow(N) {
  const w = new Float32Array(N);
  for (let n = 0; n < N; n++) w[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
  return w;
}
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function powerSpectrum(frame) {
  const N = frame.length;
  const re = new Float32Array(N), im = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    let sr = 0, si = 0;
    for (let n = 0; n < N; n++) {
      const ang = -2 * Math.PI * k * n / N;
      sr += frame[n] * Math.cos(ang);
      si += frame[n] * Math.sin(ang);
    }
    re[k] = sr; im[k] = si;
  }
  const ps = new Float32Array(N / 2 + 1);
  for (let k = 0; k <= N / 2; k++) ps[k] = (re[k] * re[k] + im[k] * im[k]) / N;
  return ps;
}
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }
function melFilterBank(sr, nFft, nMels, fmin, fmax) {
  const fminMel = hzToMel(fmin), fmaxMel = hzToMel(fmax);
  const mels = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) mels[i] = fminMel + (i * (fmaxMel - fminMel)) / (nMels + 1);
  const hz = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) hz[i] = melToHz(mels[i]);
  const freqs = new Float32Array(nFft / 2 + 1);
  for (let i = 0; i < freqs.length; i++) freqs[i] = (sr / 2) * (i / (nFft / 2));
  const fb = Array.from({ length: nMels }, () => new Float32Array(nFft / 2 + 1));
  for (let m = 1; m <= nMels; m++) {
    const f0 = hz[m - 1], f1 = hz[m], f2 = hz[m + 1];
    for (let k = 0; k < freqs.length; k++) {
      const f = freqs[k];
      fb[m - 1][k] = (f >= f0 && f <= f1) ? (f - f0) / (f1 - f0)
        : (f > f1 && f <= f2) ? (f2 - f) / (f2 - f1)
          : 0;
    }
  }
  return { fb, freqs };
}
function dct2(x, K) {
  const N = x.length, out = new Float32Array(K);
  const s0 = Math.sqrt(1 / N), s = Math.sqrt(2 / N);
  for (let k = 0; k < K; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) sum += x[n] * Math.cos(Math.PI * (n + 0.5) * k / N);
    out[k] = (k === 0 ? s0 : s) * sum;
  }
  return out;
}
function spectralCentroid(freqs, S) {
  let num = 0, den = 0; for (let i = 0; i < S.length; i++) { num += freqs[i] * S[i]; den += S[i]; }
  return den > 0 ? num / den : 0;
}
function spectralBandwidth(freqs, S, c) {
  let num = 0, den = 0; for (let i = 0; i < S.length; i++) { const d = freqs[i] - c; num += d * d * S[i]; den += S[i]; }
  return den > 0 ? Math.sqrt(num / den) : 0;
}
function spectralFlatness(S) {
  let g = 0, a = 0, eps = 1e-12;
  for (let v of S) { v += eps; g += Math.log(v); a += v; }
  g = Math.exp(g / S.length); a /= S.length;
  return a > 0 ? g / a : 0;
}
function spectralRolloff(freqs, S, roll = 0.85) {
  let tot = 0; for (let v of S) tot += v;
  const th = roll * tot; let c = 0;
  for (let i = 0; i < S.length; i++) { c += S[i]; if (c >= th) return freqs[i]; }
  return freqs[S.length - 1];
}
function zeroCrossRate(x) { let z = 0; for (let i = 1; i < x.length; i++) { const a = x[i - 1], b = x[i]; if ((a >= 0 && b < 0) || (a < 0 && b >= 0)) z++; } return z / x.length; }
function spectralFlux(prevS, S) { let f = 0; for (let i = 0; i < S.length; i++) { const d = S[i] - (prevS ? prevS[i] : 0); f += Math.max(d, 0); } return f / S.length; }
function bandEnergyRatio(freqs, S, a, b) { let lo = 0, tot = 0; for (let i = 0; i < S.length; i++) { const v = S[i]; tot += v; if (freqs[i] >= a && freqs[i] <= b) lo += v; } return tot > 0 ? lo / tot : 0; }
function resampleLinear(x, srcRate, dstRate) {
  if (srcRate === dstRate) return x.slice();
  const ratio = dstRate / srcRate;
  const nOut = Math.round(x.length * ratio);
  const y = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) {
    const t = i / ratio;
    const i0 = Math.floor(t);
    const i1 = Math.min(i0 + 1, x.length - 1);
    const frac = t - i0;
    y[i] = x[i0] * (1 - frac) + x[i1] * frac;
  }
  return y;
}

// ===== Config =====
const SR_TARGET = 16000;
const N_FFT = 1024;
const HOP = 256;
const WIN_SEC = 0.8;
const N_SAMPLES_TARGET = Math.round(SR_TARGET * WIN_SEC);
const window = hannWindow(N_FFT);
const { fb: melFb, freqs } = melFilterBank(SR_TARGET, N_FFT, 40, 20, 8000);

let audioCtx, processor, mediaStream, ortSession, classes, scalerMean, scalerStd, inited = false;

async function init() {
  if (inited) return;
  log('Loading artifacts…');
  const art = await fetch('edge_artifacts/feature_norm_and_config.json').then(r => r.json());
  classes = art.classes;
  log('classes=' + JSON.stringify(classes));
  if (!classes || classes.indexOf(DRONE_CLASS_NAME) === -1) {
    log('WARNING: DRONE_CLASS_NAME "' + DRONE_CLASS_NAME + '" not found in classes; alerts will not trigger.');
  }
  scalerMean = new Float32Array(art.scaler_mean);
  scalerStd = new Float32Array(art.scaler_std);
  setStatus('loading model…');
  ortSession = await ort.InferenceSession.create('edge_artifacts/drone_33d_mlp.onnx', { executionProviders: ['wasm'] });
  setStatus('ready');
  log('Artifacts loaded');
  inited = true;
}

btn?.addEventListener('click', async () => {
  log('Enable Microphone clicked');
  setStatus('starting…');
  try {
    await init();
  } catch (e) {
    log('Init error: ' + e.message);
    setStatus('error');
    return;
  }
  try {
    await startMic();
  } catch (e) {
    log('Start error: ' + e.message);
    setStatus('error');
  }
});

// ===== Mic + inference loop (iOS-safe) =====
async function startMic() {
  // Support check
  const AC = window.AudioContext || window.webkitAudioContext;
  log(`Support — AudioContext:${!!window.AudioContext} webkitAudioContext:${!!window.webkitAudioContext} HTTPS:${location.protocol === 'https:'}`);
  if (!AC) {
    setStatus('unsupported');
    log('Web Audio not available. Open in the Safari app (not an in-app browser), ensure HTTPS, and enable Web Audio under Settings → Safari → Advanced → Experimental Features.');
    return;
  }

  // Ask mic
  setStatus('requesting mic…');
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });
    log('Microphone permission granted');
  } catch (err) {
    setStatus('mic blocked');
    log('Microphone permission error: ' + err.message);
    return;
  }

  // Create context (no options), then resume if suspended
  let ctx;
  try {
    ctx = new AC();
  } catch (e) {
    setStatus('unsupported');
    log('AudioContext creation failed: ' + e.message);
    return;
  }
  audioCtx = ctx;
  if (audioCtx.state === 'suspended') {
    log('AudioContext suspended; attempting resume…');
    try { await audioCtx.resume(); log('AudioContext resumed'); } catch (e) { log('Resume failed: ' + e.message); }
  }
  const SR_CTX = audioCtx.sampleRate;
  log('AudioContext rate: ' + SR_CTX + ' Hz');

  const src = audioCtx.createMediaStreamSource(stream);
  mediaStream = stream;

  // Ring buffer @ context rate
  const N_SAMPLES_CTX = Math.round(SR_CTX * WIN_SEC);
  const ring = new Float32Array(N_SAMPLES_CTX);
  let idx = 0;

  // Capture
  const workSize = 2048;
  processor = audioCtx.createScriptProcessor(workSize, 1, 1);
  processor.onaudioprocess = (e) => {
    const d = e.inputBuffer.getChannelData(0);
    for (let i = 0; i < d.length; i++) { ring[idx] = d[i]; idx = (idx + 1) % N_SAMPLES_CTX; }
  };

  src.connect(processor);
  processor.connect(audioCtx.destination);
  setStatus('listening');
  log('Listening… (tap button again if you don’t see updates)');

  // periodic inference
  setInterval(() => runInference(ring, idx, SR_CTX), Math.round(WIN_SEC * 1000));
}

// ==== send alert ====
async function postAlert(probDrone, bearingDeg) {
  var now = Date.now();
  if (now - lastAlertAt < ALERT_COOLDOWN_MS) return; // debounce
  lastAlertAt = now;

  var ts_ns = (BigInt(Date.now()) * 1000000n).toString(); // ms→ns
  var body = {
    nodeId: SENSOR_NODE_ID,
    time_ms: now,
    ts_ns: ts_ns,
    azimuth_deg: bearingDeg !== undefined ? bearingDeg : null, // null if  you don't know the bearing
    confidence: probDrone,          // 0..1
    event: "drone",
    // (Optional) Longitude & Latitude
    // lat: 52.5205, lon: 13.4055
  };
  try {
    log("POST alert → " + WEBHOOK_URL);
    var res = await fetch(WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    var txt = "";
    try { txt = await res.text(); } catch (e) { }
    log("Alert POST → status=" + res.status + " ok=" + res.ok + " body=" + (txt || "<empty>"));
  } catch (e) {
    log("Alert POST error: " + e.message);
  }
}

async function runInference(ring, idx, SR_CTX) {
  try {
    // unwrap
    const bufCtx = new Float32Array(ring.length);
    const tail = ring.slice(idx);
    bufCtx.set(tail, 0);
    bufCtx.set(ring.slice(0, idx), tail.length);

    // resample → exact 0.8 s
    const buf16 = resampleLinear(bufCtx, SR_CTX, SR_TARGET);
    let buf = buf16;
    if (buf.length > N_SAMPLES_TARGET) buf = buf.slice(buf.length - N_SAMPLES_TARGET);
    if (buf.length < N_SAMPLES_TARGET) {
      const x = new Float32Array(N_SAMPLES_TARGET);
      x.set(buf, N_SAMPLES_TARGET - buf.length);
      buf = x;
    }

    // framing
    const frames = [];
    for (let start = 0; start + N_FFT <= buf.length; start += HOP) {
      const fr = buf.slice(start, start + N_FFT);
      for (let i = 0; i < fr.length; i++) fr[i] *= window[i];
      frames.push(fr);
    }

    // per-frame features
    const mfccFrames = [];
    const centroid = [], rolloff = [], flatness = [], bandwidth = [], zcrArr = [], fluxArr = [], brLowArr = [], brMidArr = [];
    let prevSpec = null;

    for (let fr of frames) {
      const S = powerSpectrum(fr);
      const melE = new Float32Array(melFb.length);
      for (let m = 0; m < melFb.length; m++) melE[m] = dot(melFb[m], S);
      for (let m = 0; m < melE.length; m++) melE[m] = Math.log(melE[m] + 1e-9);
      mfccFrames.push(dct2(melE, 20));

      const c = spectralCentroid(freqs, S);
      const b = spectralBandwidth(freqs, S, c);
      const r = spectralRolloff(freqs, S, 0.85);
      const f = spectralFlatness(S);
      const z = zeroCrossRate(fr);
      const fl = spectralFlux(prevSpec, S);
      const brL = bandEnergyRatio(freqs, S, 0, 120);
      const brM = bandEnergyRatio(freqs, S, 150, 450);

      centroid.push(c); bandwidth.push(b); rolloff.push(r); flatness.push(f);
      zcrArr.push(z); fluxArr.push(fl); brLowArr.push(brL); brMidArr.push(brM);
      prevSpec = S;
    }

    // aggregate to 33-D
    const mfccMean = new Float32Array(20);
    for (let k = 0; k < 20; k++) { let s = 0; for (let i = 0; i < mfccFrames.length; i++) s += mfccFrames[i][k]; mfccMean[k] = s / Math.max(1, mfccFrames.length); }
    const energyRms = Math.sqrt(buf.reduce((s, v) => s + v * v, 0) / buf.length);
    function meanStd(a) { const n = a.length || 1; let mu = 0; for (let v of a) mu += v; mu /= n; let v2 = 0; for (let v of a) v2 += (v - mu) * (v - mu); return [mu, Math.sqrt(v2 / n)]; }
    const [centMu, centStd] = meanStd(centroid);
    const [rollMu, rollStd] = meanStd(rolloff);
    const [flatMu, flatStd] = meanStd(flatness);
    const [bandwMu, bandwStd] = meanStd(bandwidth);
    const fluxMu = meanStd(fluxArr)[0];
    const zcrMu = meanStd(zcrArr)[0];
    const brLowMu = meanStd(brLowArr)[0];
    const brMidMu = meanStd(brMidArr)[0];

    const feats = new Float32Array(33);
    for (let i = 0; i < 20; i++) feats[i] = mfccMean[i];
    feats[20] = energyRms;
    feats[21] = centMu; feats[22] = centStd;
    feats[23] = rollMu; feats[24] = rollStd;
    feats[25] = flatMu; feats[26] = flatStd;
    feats[27] = bandwMu; feats[28] = bandwStd;
    feats[29] = fluxMu; feats[30] = zcrMu;
    feats[31] = brLowMu; feats[32] = brMidMu;

    // standardize & infer
    const X = new Float32Array(33);
    for (let i = 0; i < 33; i++) { const std = scalerStd[i] || 1.0; X[i] = (feats[i] - scalerMean[i]) / std; }

    const input = new ort.Tensor('float32', X, [1, 33]);
    const out = await ortSession.run({ input_0: input });
    const outName = Object.keys(out)[0];
    const logits = Array.from(out[outName].data);
    const m = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(v => v / sum);

    updateTable(classes, probs);

    // ==== ADD: decision with hysteresis and logging ====
    var idxDrone = classes ? classes.indexOf(DRONE_CLASS_NAME) : -1; // "drone" index
    if (idxDrone < 0) {
      log('Decision: class "' + DRONE_CLASS_NAME + '" not found — no alert.');
    } else {
      var pDrone = probs[idxDrone]; // 0..1
      var armed = !prevAbove;
      var releaseThresh = ALERT_THRESHOLD * ALERT_RELEASE_RATIO;

      if (pDrone >= ALERT_THRESHOLD && armed) {
        log('Decision: p(drone)=' + pDrone.toFixed(3) + ' >= ' + ALERT_THRESHOLD + ' → POST /webhook/edge');
        prevAbove = true;
        postAlert(pDrone, null); // bearing TBD
      } else {
        if (pDrone < releaseThresh && prevAbove) {
          log('Decision: p(drone) dropped below ' + releaseThresh.toFixed(3) + ' → re-arming');
          prevAbove = false;
        } else {
          // verbose trace (optional): comment out if too chatty
          // log('Decision: hold (p=' + pDrone.toFixed(3) + ', armed=' + armed + ')');
        }
      }
    }
  } catch (e) {
    log('Inference error: ' + e.message);
  }
}

function updateTable(classes, probs) {
  if (!probsTable || !probsTbody) return;
  probsTable.style.display = 'table';
  probsTbody.innerHTML = '';
  const pairs = classes.map((c, i) => ({ c, p: probs[i] })).sort((a, b) => b.p - a.p);
  for (const { c, p } of pairs) {
    const tr = document.createElement('tr');
    const td0 = document.createElement('td'); td0.textContent = c;
    const td1 = document.createElement('td'); td1.textContent = (p * 100).toFixed(1) + '%'; td1.className = 'prob';
    tr.appendChild(td0); tr.appendChild(td1);
    probsTbody.appendChild(tr);
  }
}

// Final sanity log
if (location.protocol !== 'https:') {
  log('WARNING: Page is not served over HTTPS. iOS Safari may block mic or cross-origin requests.');
}
