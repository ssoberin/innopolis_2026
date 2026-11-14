# Avito Hackathon — replaceable application

**Only `app/` is updated between versions.** Persistent resources are siblings under `/srv/Hakaton`:

- `/srv/Hakaton/data/` — datasets and splits
- `/srv/Hakaton/models/baseline/tfidf_baseline.joblib` — trained TF-IDF model
- `/srv/Hakaton/models/huggingface/` — Qwen weights
- `/srv/Hakaton/results/` — experiment results
- `/srv/Hakaton/infrastructure/vllm/compose.yaml` — Docker config
- `/srv/Hakaton/.venv/` — Python packages

## One-time environment setup (only if root .venv does not exist)

```bash
cd /srv/Hakaton
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt
```

If there is already a working `/srv/Hakaton/.venv`, reuse it and install only missing packages.

## Run

```bash
cd /srv/Hakaton
source .venv/bin/activate
python -m pytest -q app/tests
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Example classification-only request (no LLM):

```bash
curl -sS -X POST 'http://127.0.0.1:8001/generate?generate_text=false' \
  -H 'Content-Type: application/json' \
  -d '{"description":"Штукатурим стены, укладываем плитку, меняем розетки.","mcId":101}'
```

Full generation (requires vLLM at 127.0.0.1:8000):

```bash
curl -sS -X POST 'http://127.0.0.1:8001/generate?generate_text=true' \
  -H 'Content-Type: application/json' \
  -d '{"description":"Укладываем плитку на стены и пол, устанавливаем раковины и унитазы.","mcId":101}'
```

If Qwen is unavailable or rejects evidence, returns a valid draft with `title` and `text` as `null`.

**Known limitations:** classifier only has trained heads 102–106, so cannot predict target 101 when the source mcId differs from 101. Weak GPT-generated training labels do not guarantee official evaluation F1. Exact evidence matching may reject legitimate generated texts. `GET /health` checks model startup, not Qwen availability. No authentication is provided: keep HTTP ports bound to localhost/private tunnel and do not publicly expose the API without an authenticated gateway.
