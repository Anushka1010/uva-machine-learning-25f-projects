# Quick Start 🚀

## Setup (one-time)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Upgrade pip & install deps
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn  # For API server
```

## Run Tests

```bash
source venv/bin/activate
python -m pytest
```

## Train a Model

```bash
python scripts/run_local.py --train
```

## Start API Server

```bash
python scripts/run_local.py --api
# API: http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
```

## Test API

```bash
python scripts/run_local.py --test-api
```

## Full Pipeline (train + API)

```bash
python scripts/run_local.py
```

---

**Troubleshooting:**

- Use `python -m pytest` not just `pytest` (ensures correct venv)
- Use `python -m uvicorn` not just `uvicorn`
- If venv is broken: `rm -rf venv && python3 -m venv venv`

See [README.md](README.md) for full docs.
