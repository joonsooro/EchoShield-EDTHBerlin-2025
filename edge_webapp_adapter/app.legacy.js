/***** Drone Classifier – Legacy (no modules) *****/

// --- Global JS error trap & environment dump ---
window.addEventListener('error', function (e) {
  try { log('[JSERROR] ' + (e.message || e.error)); } catch (_) { }
});
(function envDump() {
  try {
    var ua = navigator.userAgent || '';
    var isSafari = ua.includes('Safari') && !ua.includes('Chrome') && !ua.includes('CriOS') && !ua.includes('FxiOS');
    var isInApp = /\bFBAN|FBAV|Instagram|Line\/|GSA\/|DuckDuckGo|Twitter\b/i.test(ua);
    log('[Env] ua=' + ua);
    log('[Env] safari=' + isSafari + ' inApp=' + isInApp + ' https=' + (location.protocol === 'https:') + ' isSecureContext=' + !!window.isSecureContext);
    if (window.top !== window) {
      log('WARNING: Page is running inside an iframe. iOS Safari may block geolocation prompts in iframes.');
    }
    if (location.protocol !== 'https:') {
      log('WARNING: Page is not HTTPS. iOS Safari blocks geolocation on HTTP.');
    }
    if (isInApp) {
      log('WARNING: In-app browsers often block geolocation prompts. Open in Safari app.');
    }
  } catch (e) {
    try { log('[EnvDump error] ' + e.message); } catch (_) { }
  }
})();

// ===== UI hooks =====
var logEl = document.getElementById('log');
var btn = document.getElementById('btnStart');
var statusPill = document.getElementById('status');
var probsTable = document.getElementById('probsTable');
var probsTbody = probsTable ? probsTable.querySelector('tbody') : null;

function log(msg) {
  var time = new Date().toISOString().split('T')[1].split('.')[0];
  if (logEl) {
    logEl.textContent += "[" + time + "] " + msg + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }
}
function setStatus(s) { if (statusPill) statusPill.textContent = s; }

log("App legacy loaded");

// ==== Alert POST config (ADD) ====
var DRONE_CLASS_NAME = "drone";
var ALERT_THRESHOLD = 0.80;          // trigger at 80%
var ALERT_COOLDOWN_MS = 5000;        // debounce: 5s between posts
var WEBHOOK_URL = "/webhook/edge";   // same-origin; ngrok proxies to adapter
var SENSOR_NODE_ID = "NODE_IPHONE_01";

// Hysteresis state to avoid jitter spam
var prevAbove = false;               // true if last state was above threshold
var ALERT_RELEASE_RATIO = 0.8;       // re-arm when below 80% of threshold

var lastAlertAt = 0;

// --- Geo state (optional; populated on capable devices) ---
var lastLat = null;
var lastLon = null;
var lastAcc = null; // meters

var geoWatchId = null;

// --- Geolocation diagnostics ---
function geoDiagLog(context) {
  try {
    var secure = (location.protocol === 'https:') && (window.isSecureContext !== false);
    log("[GeoDiag] ctx=" + context + " https=" + (location.protocol === 'https:') + " isSecureContext=" + !!window.isSecureContext);
    if (navigator.permissions && navigator.permissions.query) {
      navigator.permissions.query({ name: 'geolocation' })
        .then(function (res) {
          log("[GeoDiag] perm.state=" + res.state);
          if (res && typeof res.onchange === 'object') {
            try {
              res.onchange = function () {
                log("[GeoDiag] perm.onchange → " + res.state);
              };
            } catch (_) { }
          }
        })
        .catch(function (e) { log("[GeoDiag] perm.query error: " + e.message); });
    } else {
      log("[GeoDiag] Permissions API not available");
    }
  } catch (e) {
    log("[GeoDiag] error: " + e.message);
  }
}

function startGeoWatch() {
  geoDiagLog("start");
  if (!navigator || !navigator.geolocation || typeof navigator.geolocation.getCurrentPosition !== 'function') {
    log("Geolocation API not present (navigator.geolocation missing).");
    return;
  }
  if (location.protocol !== 'https:') {
    log("WARNING: Geolocation requires HTTPS on iOS. Open the page via an https URL (e.g., ngrok https).");
  }
  try {
    if (geoWatchId !== null) navigator.geolocation.clearWatch(geoWatchId);
  } catch (_) { }

  var opts = { enableHighAccuracy: true, maximumAge: 0, timeout: 8000 };

  // First, request a one-shot position to trigger the permission prompt on iOS.
  try {
    navigator.geolocation.getCurrentPosition(
      function (pos) {
        var c = pos.coords;
        lastLat = (typeof c.latitude === "number") ? c.latitude : null;
        lastLon = (typeof c.longitude === "number") ? c.longitude : null;
        lastAcc = (typeof c.accuracy === "number") ? c.accuracy : null;
        log("Geo one-shot lat=" + lastLat + " lon=" + lastLon + " acc_m=" + lastAcc);
        // After successful one-shot (or after prompt), start the watch for continuous updates.
        geoWatchId = navigator.geolocation.watchPosition(
          function (pos2) {
            var c2 = pos2.coords;
            lastLat = (typeof c2.latitude === "number") ? c2.latitude : null;
            lastLon = (typeof c2.longitude === "number") ? c2.longitude : null;
            lastAcc = (typeof c2.accuracy === "number") ? c2.accuracy : null;
            log("Geo update lat=" + lastLat + " lon=" + lastLon + " acc_m=" + lastAcc);
          },
          function (err2) {
            log("Geo watch error: " + err2.message);
          },
          opts
        );
        log("Geolocation watch started");
      },
      function (err) {
        // Even if one-shot fails (timeout/denied), try starting the watch so that user can grant later.
        log("Geo one-shot error: " + err.message + " (will still try watchPosition)");
        try {
          geoWatchId = navigator.geolocation.watchPosition(
            function (pos2) {
              var c2 = pos2.coords;
              lastLat = (typeof c2.latitude === "number") ? c2.latitude : null;
              lastLon = (typeof c2.longitude === "number") ? c2.longitude : null;
              lastAcc = (typeof c2.accuracy === "number") ? c2.accuracy : null;
              log("Geo update lat=" + lastLat + " lon=" + lastLon + " acc_m=" + lastAcc);
            },
            function (err2) {
              log("Geo watch error: " + err2.message);
            },
            opts
          );
          log("Geolocation watch started (after one-shot fail)");
        } catch (e) {
          log("Geo watch start failed: " + e.message);
        }
      },
      opts
    );
  } catch (e) {
    log("Geolocation getCurrentPosition failed: " + e.message);
  }
}

// --- Direct one-shot request tied to user gesture + fallback watch ---
function requestGeoOnceThenWatch() {
  if (!('geolocation' in navigator)) {
    log('Geolocation API not available in this browser/context.');
    alert('Geolocation not available. Use Safari app with HTTPS.');
    return;
  }
  var opts = { enableHighAccuracy: true, maximumAge: 0, timeout: 8000 };
  log('Geo direct one-shot calling getCurrentPosition()');
  navigator.geolocation.getCurrentPosition(
    function (pos) {
      var c = pos.coords;
      lastLat = (typeof c.latitude === 'number') ? c.latitude : null;
      lastLon = (typeof c.longitude === 'number') ? c.longitude : null;
      lastAcc = (typeof c.accuracy === 'number') ? c.accuracy : null;
      log('Geo direct one-shot lat=' + lastLat + ' lon=' + lastLon + ' acc_m=' + lastAcc);
      log('Geo direct one-shot SUCCESS, starting watchPosition()');
      try { if (geoWatchId !== null) navigator.geolocation.clearWatch(geoWatchId); } catch (_) { }
      geoWatchId = navigator.geolocation.watchPosition(
        function (pos2) {
          var c2 = pos2.coords;
          lastLat = (typeof c2.latitude === 'number') ? c2.latitude : null;
          lastLon = (typeof c2.longitude === 'number') ? c2.longitude : null;
          lastAcc = (typeof c2.accuracy === 'number') ? c2.accuracy : null;
          log('Geo update (direct path) lat=' + lastLat + ' lon=' + lastLon + ' acc_m=' + lastAcc);
        },
        function (err2) {
          log('Geo watch error (direct path): ' + err2.message + ' code=' + err2.code);
          alert('Geo watch error: ' + err2.message);
        },
        opts
      );
    },
    function (err) {
      log('Geo direct one-shot error detail: code=' + err.code + ' message=' + err.message + ' PERM=' + (navigator.permissions ? 'yes' : 'no'));
      alert('Geo error: ' + err.message + ' (Check Safari > Location: Allow, and open via HTTPS in Safari app)');
    },
    opts
  );
}

// On-page diagnostics at load
(function () {
  var diag = {
    protocol: location.protocol,
    origin: location.origin,
    href: location.href,
    hasAudioContext: !!window.AudioContext,
    hasWebkitAudioContext: !!window.webkitAudioContext,
    hasMediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    ua: navigator.userAgent
  };
  log("Support diag: " + JSON.stringify(diag));
})();

// ===== Math/DSP helpers =====
function hannWindow(N) { var w = new Float32Array(N); for (var n = 0; n < N; n++) w[n] = 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1))); return w; }
function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function powerSpectrum(frame) {
  var N = frame.length, re = new Float32Array(N), im = new Float32Array(N);
  for (var k = 0; k < N; k++) {
    var sr = 0, si = 0;
    for (var n = 0; n < N; n++) {
      var ang = -2 * Math.PI * k * n / N;
      sr += frame[n] * Math.cos(ang);
      si += frame[n] * Math.sin(ang);
    }
    re[k] = sr; im[k] = si;
  }
  var ps = new Float32Array(N / 2 + 1);
  for (var j = 0; j <= N / 2; j++) ps[j] = (re[j] * re[j] + im[j] * im[j]) / N;
  return ps;
}
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }
function melFilterBank(sr, nFft, nMels, fmin, fmax) {
  var fminMel = hzToMel(fmin), fmaxMel = hzToMel(fmax);
  var mels = new Float32Array(nMels + 2); for (var i = 0; i < mels.length; i++) mels[i] = fminMel + (i * (fmaxMel - fminMel)) / (nMels + 1);
  var hz = new Float32Array(nMels + 2); for (var j = 0; j < hz.length; j++) hz[j] = melToHz(mels[j]);
  var freqs = new Float32Array(nFft / 2 + 1); for (var k = 0; k < freqs.length; k++) freqs[k] = (sr / 2) * (k / (nFft / 2));
  var fb = new Array(nMels); for (var m = 0; m < nMels; m++) fb[m] = new Float32Array(nFft / 2 + 1);
  for (var m2 = 1; m2 <= nMels; m2++) {
    var f0 = hz[m2 - 1], f1 = hz[m2], f2 = hz[m2 + 1];
    for (var q = 0; q < freqs.length; q++) {
      var f = freqs[q];
      fb[m2 - 1][q] = (f >= f0 && f <= f1) ? (f - f0) / (f1 - f0) : (f > f1 && f <= f2) ? (f2 - f) / (f2 - f1) : 0;
    }
  }
  return { fb: fb, freqs: freqs };
}
function dct2(x, K) {
  var N = x.length, out = new Float32Array(K);
  var s0 = Math.sqrt(1 / N), s = Math.sqrt(2 / N);
  for (var k = 0; k < K; k++) {
    var sum = 0;
    for (var n = 0; n < N; n++) sum += x[n] * Math.cos(Math.PI * (n + 0.5) * k / N);
    out[k] = (k === 0 ? s0 : s) * sum;
  }
  return out;
}
function spectralCentroid(freqs, S) { var num = 0, den = 0; for (var i = 0; i < S.length; i++) { num += freqs[i] * S[i]; den += S[i]; } return den > 0 ? num / den : 0; }
function spectralBandwidth(freqs, S, c) { var num = 0, den = 0; for (var i = 0; i < S.length; i++) { var d = freqs[i] - c; num += d * d * S[i]; den += S[i]; } return den > 0 ? Math.sqrt(num / den) : 0; }
function spectralFlatness(S) { var g = 0, a = 0, eps = 1e-12; for (var i = 0; i < S.length; i++) { var v = S[i] + eps; g += Math.log(v); a += v; } g = Math.exp(g / S.length); a /= S.length; return a > 0 ? g / a : 0; }
function spectralRolloff(freqs, S, roll) { if (roll === void 0) roll = 0.85; var tot = 0; for (var i = 0; i < S.length; i++) tot += S[i]; var th = roll * tot, c = 0; for (var k = 0; k < S.length; k++) { c += S[k]; if (c >= th) return freqs[k]; } return freqs[S.length - 1]; }
function zeroCrossRate(x) { var z = 0; for (var i = 1; i < x.length; i++) { var a = x[i - 1], b = x[i]; if ((a >= 0 && b < 0) || (a < 0 && b >= 0)) z++; } return z / x.length; }
function spectralFlux(prevS, S) { var f = 0; for (var i = 0; i < S.length; i++) { var d = S[i] - (prevS ? prevS[i] : 0); f += Math.max(d, 0); } return f / S.length; }
function bandEnergyRatio(freqs, S, a, b) { var lo = 0, tot = 0; for (var i = 0; i < S.length; i++) { var v = S[i]; tot += v; if (freqs[i] >= a && freqs[i] <= b) lo += v; } return tot > 0 ? lo / tot : 0; }
function resampleLinear(x, srcRate, dstRate) {
  if (srcRate === dstRate) return x.slice();
  var ratio = dstRate / srcRate, nOut = Math.round(x.length * ratio), y = new Float32Array(nOut);
  for (var i = 0; i < nOut; i++) {
    var t = i / ratio, i0 = Math.floor(t), i1 = Math.min(i0 + 1, x.length - 1), frac = t - i0;
    y[i] = x[i0] * (1 - frac) + x[i1] * frac;
  }
  return y;
}

// ===== Config =====
var SR_TARGET = 16000, N_FFT = 1024, HOP = 256, WIN_SEC = 0.8;
var N_SAMPLES_TARGET = Math.round(SR_TARGET * WIN_SEC);
var windowArr = hannWindow(N_FFT);
var mf = melFilterBank(SR_TARGET, N_FFT, 40, 20, 8000);
var melFb = mf.fb, freqs = mf.freqs;

var audioCtx = null, processor = null, mediaStream = null, ortSession = null, classes = null, scalerMean = null, scalerStd = null, inited = false;

function meanStd(a) { var n = a.length || 1, mu = 0; for (var i = 0; i < n; i++) mu += a[i]; mu /= n; var v2 = 0; for (var j = 0; j < n; j++) { var d = a[j] - mu; v2 += d * d; } return [mu, Math.sqrt(v2 / n)]; }

async function init() {
  if (inited) return;
  log('Loading artifacts…');
  var art = await fetch('edge_artifacts/feature_norm_and_config.json').then(function (r) { return r.json(); });
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

if (btn) {
  btn.addEventListener('click', function () {
    log('Enable Microphone clicked');
    setStatus('starting…');
    // If geolocation doesn't update within 6s, emit guidance
    setTimeout(function () {
      if (lastLat === null || lastLon === null) {
        log("Geo WARNING: no location yet. On iOS, ensure:");
        log(" - You opened this page via HTTPS (ngrok https URL)");
        log(" - You tapped the button in Safari (not an in-app browser)");
        log(" - Settings > Privacy > Location Services > Safari Websites = While Using the App");
        log(" - Site-specific permission: address bar 'aA' > Website Settings > Location: Allow");
      }
    }, 6000);
    init().then(function () {
      startGeoWatch();
      return startMic();
    }).catch(function (e) {
      geoDiagLog("init.catch");
      log('Init/Start error: ' + e.message);
      setStatus('error');
    });
  });
} else {
  log('Button not found in DOM');
}

// Ensure the dedicated location button also triggers geolocation under a direct user gesture
document.addEventListener('DOMContentLoaded', function () {
  try {
    var btnLoc = document.getElementById('btnLoc');
    if (btnLoc) {
      btnLoc.addEventListener('click', function () {
        log('Enable Location clicked (gesture path)');
        if (!navigator || !navigator.geolocation) {
          alert('Geolocation API not present. Open in Safari over HTTPS.');
          log('No navigator.geolocation — aborting.');
          return;
        }
        // Force prompt on the same user gesture
        requestGeoOnceThenWatch();
      });
    } else {
      log('btnLoc not found in DOM');
    }
  } catch (e) {
    log('btnLoc handler attach error: ' + e.message);
  }
});

// POST alert to adapter
async function postAlert(probDrone, bearingDeg) {
  var now = Date.now();
  if (now - lastAlertAt < ALERT_COOLDOWN_MS) return; // debounce
  lastAlertAt = now;

  var ts_ns = (BigInt(now) * 1000000n).toString();
  var body = {
    nodeId: SENSOR_NODE_ID,
    time_ms: now,
    ts_ns: ts_ns,
    azimuth_deg: (bearingDeg === undefined ? null : bearingDeg),
    confidence: probDrone,
    event: "drone",
    // optional geo (adapter will handle nulls)
    lat: (lastLat !== null ? lastLat : null),
    lon: (lastLon !== null ? lastLon : null),
    acc_m: (lastAcc !== null ? lastAcc : null)
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

// Mic + inference (iOS-safe)
async function startMic() {
  var AC = window.AudioContext || window.webkitAudioContext;
  log('Support — AudioContext:' + !!window.AudioContext + ' webkitAudioContext:' + !!window.webkitAudioContext + ' HTTPS:' + (location.protocol === 'https:'));
  if (!AC) {
    setStatus('unsupported');
    log('Web Audio not available. Open in Safari app (not in-app), ensure HTTPS, enable Web Audio under Settings → Safari → Advanced → Experimental Features.');
    return;
  }

  // ask mic
  setStatus('requesting mic…');
  var stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    log('Microphone permission granted');
  } catch (err) {
    setStatus('mic blocked');
    log('Microphone permission error: ' + err.message);
    return;
  }

  // create context (no options) and resume if suspended
  var ctx;
  try { ctx = new AC(); }
  catch (e) { setStatus('unsupported'); log('AudioContext creation failed: ' + e.message); return; }
  audioCtx = ctx;
  if (audioCtx.state === 'suspended') {
    log('AudioContext suspended; attempting resume…');
    try { await audioCtx.resume(); log('AudioContext resumed'); } catch (e) { log('Resume failed: ' + e.message); }
  }
  var SR_CTX = audioCtx.sampleRate;
  log('AudioContext rate: ' + SR_CTX + ' Hz');

  var src = audioCtx.createMediaStreamSource(stream);
  mediaStream = stream;

  var N_SAMPLES_CTX = Math.round(SR_CTX * WIN_SEC);
  var ring = new Float32Array(N_SAMPLES_CTX);
  var idx = 0;

  // capture
  var workSize = 2048;
  processor = audioCtx.createScriptProcessor(workSize, 1, 1);
  processor.onaudioprocess = function (e) {
    var d = e.inputBuffer.getChannelData(0);
    for (var i = 0; i < d.length; i++) { ring[idx] = d[i]; idx = (idx + 1) % N_SAMPLES_CTX; }
  };

  src.connect(processor);
  processor.connect(audioCtx.destination);
  setStatus('listening');
  log('Listening…');

  // periodic inference
  setInterval(function () { runInference(ring, idx, SR_CTX); }, Math.round(WIN_SEC * 1000));
}

async function runInference(ring, idx, SR_CTX) {
  try {
    var bufCtx = new Float32Array(ring.length);
    var tail = ring.slice(idx);
    bufCtx.set(tail, 0); bufCtx.set(ring.slice(0, idx), tail.length);

    var buf16 = resampleLinear(bufCtx, SR_CTX, SR_TARGET);

    var buf = buf16;
    if (buf.length > N_SAMPLES_TARGET) buf = buf.slice(buf.length - N_SAMPLES_TARGET);
    if (buf.length < N_SAMPLES_TARGET) { var x = new Float32Array(N_SAMPLES_TARGET); x.set(buf, N_SAMPLES_TARGET - buf.length); buf = x; }

    var frames = [], start = 0;
    for (start = 0; start + N_FFT <= buf.length; start += HOP) {
      var fr = buf.slice(start, start + N_FFT);
      for (var i = 0; i < fr.length; i++) fr[i] *= windowArr[i];
      frames.push(fr);
    }

    var mfccFrames = [], centroid = [], rolloff = [], flatness = [], bandwidth = [], zcrArr = [], fluxArr = [], brLowArr = [], brMidArr = [], prevSpec = null;
    for (var fi = 0; fi < frames.length; fi++) {
      var fr2 = frames[fi];
      var S = powerSpectrum(fr2);
      var melE = new Float32Array(melFb.length);
      for (var m = 0; m < melFb.length; m++) melE[m] = dot(melFb[m], S);
      for (var m2 = 0; m2 < melE.length; m2++) melE[m2] = Math.log(melE[m2] + 1e-9);
      mfccFrames.push(dct2(melE, 20));

      var c = spectralCentroid(freqs, S);
      var b = spectralBandwidth(freqs, S, c);
      var r = spectralRolloff(freqs, S, 0.85);
      var f = spectralFlatness(S);
      var z = zeroCrossRate(fr2);
      var fl = spectralFlux(prevSpec, S);
      var brL = bandEnergyRatio(freqs, S, 0, 120);
      var brM = bandEnergyRatio(freqs, S, 150, 450);

      centroid.push(c); bandwidth.push(b); rolloff.push(r); flatness.push(f);
      zcrArr.push(z); fluxArr.push(fl); brLowArr.push(brL); brMidArr.push(brM);
      prevSpec = S;
    }

    var mfccMean = new Float32Array(20);
    for (var k = 0; k < 20; k++) { var s = 0; for (var ii = 0; ii < mfccFrames.length; ii++) s += mfccFrames[ii][k]; mfccMean[k] = s / Math.max(1, mfccFrames.length); }
    var energyRms = Math.sqrt(buf.reduce(function (s, v) { return s + v * v; }, 0) / buf.length);
    var ms;

    var msCent = meanStd(centroid); var centMu = msCent[0], centStd = msCent[1];
    var msRoll = meanStd(rolloff); var rollMu = msRoll[0], rollStd = msRoll[1];
    var msFlat = meanStd(flatness); var flatMu = msFlat[0], flatStd = msFlat[1];
    var msBand = meanStd(bandwidth); var bandwMu = msBand[0], bandwStd = msBand[1];
    var fluxMu = meanStd(fluxArr)[0];
    var zcrMu = meanStd(zcrArr)[0];
    var brLowMu = meanStd(brLowArr)[0];
    var brMidMu = meanStd(brMidArr)[0];

    var feats = new Float32Array(33);
    for (var t = 0; t < 20; t++) feats[t] = mfccMean[t];
    feats[20] = energyRms;
    feats[21] = centMu; feats[22] = centStd;
    feats[23] = rollMu; feats[24] = rollStd;
    feats[25] = flatMu; feats[26] = flatStd;
    feats[27] = bandwMu; feats[28] = bandwStd;
    feats[29] = fluxMu; feats[30] = zcrMu;
    feats[31] = brLowMu; feats[32] = brMidMu;

    var X = new Float32Array(33);
    for (var p = 0; p < 33; p++) { var std = scalerStd[p] || 1.0; X[p] = (feats[p] - scalerMean[p]) / std; }

    var input = new ort.Tensor('float32', X, [1, 33]);
    var out = await ortSession.run({ input_0: input });
    var outName = Object.keys(out)[0];
    var logits = Array.from(out[outName].data);
    var m = Math.max.apply(null, logits);
    var exps = logits.map(function (v) { return Math.exp(v - m); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);
    var probs = exps.map(function (v) { return v / sum; });

    updateTable(classes, probs);

    // ==== Decision hook: trigger alert with hysteresis ====
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
  var pairs = classes.map(function (c, i) { return { c: c, p: probs[i] }; }).sort(function (a, b) { return b.p - a.p; });
  for (var i = 0; i < pairs.length; i++) {
    var tr = document.createElement('tr');
    var td0 = document.createElement('td'); td0.textContent = pairs[i].c;
    var td1 = document.createElement('td'); td1.textContent = (pairs[i].p * 100).toFixed(1) + '%'; td1.className = 'prob';
    tr.appendChild(td0); tr.appendChild(td1); probsTbody.appendChild(tr);
  }
}

// Final sanity log
if (location.protocol !== 'https:') {
  log('WARNING: Page is not served over HTTPS. iOS Safari may block mic or cross-origin requests.');
}
