# Pneumonia Detection — shortcuts
#
#   make setup    create venv and install the project stack + pytest
#   make test     run the pytest suite (src/metrics.py + models/ artifacts)
#   make report   print the model-comparison report from models/results.json
#   make app      launch the Streamlit X-ray classifier (needs the trained
#                 weights in models/best_resnet50.pth; not committed)
#   make clean    remove caches and the venv
#
# Usage:
#   make setup && source venv/bin/activate && make test

PY ?= python3
VENV ?= venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: setup test report app clean

setup:
	$(PY) -m venv $(VENV)
	$(ACTIVATE) && pip install -r requirements.txt
	$(ACTIVATE) && pip install pytest

test:
	$(PY) -m pytest tests/ -q

report:
	$(PY) examples/results_report.py

app:
	cd app && streamlit run app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	rm -rf $(VENV)
