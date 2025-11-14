
import logging
import os
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mappers import to_wirepacket
from fastapi import Response

CFG_PATH = os.environ.get("ADAPTER_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

INGEST_URL = os.environ.get("INGEST_URL", CFG.get("INGEST_URL"))
API_KEY = os.environ.get("API_KEY", CFG.get("API_KEY", ""))
LOG_LEVEL = os.environ.get("LOG_LEVEL", CFG.get("LOG_LEVEL", "INFO")).upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("edge_webapp_adapter")

app = FastAPI(title="Edge WebApp Adapter", version="0.1")

@app.get("/whoami", include_in_schema=False)
def whoami_adapter():
    return {"service": "edge_webapp_adapter", "port": 8000}

@app.get("/health")
async def health():
    return {"status": "ok", "ingest_url": INGEST_URL}

@app.get("/geo-test", include_in_schema=False)
def geo_test():
    html = """
<!DOCTYPE html>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geo Test</title>
<button id="go">Ask Location</button>
<pre id="out"></pre>
<script>
  function log(t){ try{ document.getElementById('out').textContent += t + "\\n"; }catch(e){} }
  window.addEventListener('error', function(e){ log('[JSERROR] ' + (e.message || e.error)); });
  (function(){ log('proto=' + location.protocol + ' secure=' + (window.isSecureContext!==false)); log('ua=' + navigator.userAgent); })();
  document.getElementById('go').onclick = function(){
    if (!navigator.geolocation){ alert('no geolocation'); log('no geolocation'); return; }
    log('calling getCurrentPosition()');
    navigator.geolocation.getCurrentPosition(
      function(p){ log('OK lat=' + p.coords.latitude + ' lon=' + p.coords.longitude + ' acc=' + p.coords.accuracy); },
      function(err){ log('ERR code=' + err.code + ' msg=' + err.message); alert('Geo error: ' + err.message); },
      { enableHighAccuracy:true, maximumAge:0, timeout:8000 }
    );
  };
</script>
    """.strip()
    return Response(content=html, media_type="text/html", headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})

@app.post("/webhook/edge")
async def webhook_edge(req: Request):
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    wire = to_wirepacket(payload, CFG)
    # logger.info("Mapped wire sensor_node_id=%s", wire["sensor_node_id"])
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(INGEST_URL, headers=headers, json=wire)
            ok = r.status_code in (200, 202)
            log.info("Forwarded event_id=%s status=%s", wire["event_id"], r.status_code)
            return {"forwarded": ok, "status": r.status_code, "event_id": wire["event_id"], "ingest_url": INGEST_URL}
    except httpx.RequestError as e:
        log.error("HTTP forward error: %s", e)
        raise HTTPException(status_code=502, detail=f"forward error: {e}") from e

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/", StaticFiles(directory=str(Path(__file__).parent), html=True), name="web")

