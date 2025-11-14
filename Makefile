.PHONY: venv install run-adapter run-adapter-dev install-api run-api run-api-dev

venv:
	python -m venv .venv

install: venv
	. .venv/bin/activate && pip install -U pip && pip install -r edge_webapp_adapter/requirements.txt

run-adapter:
	. .venv/bin/activate && uvicorn --app-dir edge_webapp_adapter main:app --host 0.0.0.0 --port 8000

run-adapter-dev:
	. .venv/bin/activate && uvicorn --app-dir edge_webapp_adapter main:app --host 0.0.0.0 --port 8000 --reload

install-api:
	. .venv/bin/activate && pip install -U pip && pip install -r ingest_api/requirements.txt

run-api:
	. .venv/bin/activate && uvicorn ingest_api.app:app --host 0.0.0.0 --port 8080

run-api-dev:
	. .venv/bin/activate && uvicorn ingest_api.app:app --host 0.0.0.0 --port 8080 --reload
