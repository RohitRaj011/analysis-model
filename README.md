# Equity analysis (Streamlit)

Browser UI for Altman Z-Score, DuPont ROE, cash conversion cycle (CCC), and DCF. Core calculation scripts are unchanged; they live under `src/analysis_model/`.

## Local run

```bash
cd "/Users/rohit.raj/Downloads/Analysis Model"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example api.env   # set API= to your Financial Modeling Prep key
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Enter a ticker, pick analyses, click **Run analysis**. There is no terminal `input()`.

Optional CLI (still no prompts):

```bash
PYTHONPATH=src python main.py --ticker AAPL --all
```

## Public website (Streamlit Community Cloud)

This app is a long-running Python process, not static HTML. The default public host is **[Streamlit Community Cloud](https://share.streamlit.io)**.

1. Push this repo to GitHub (do **not** commit `api.env`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch, and main file: `app/streamlit_app.py`.
4. Under **Secrets**, add:

   ```toml
   API = "your-fmp-key"
   ```

5. Deploy. Users get a URL like `https://<app-name>.streamlit.app`.

Cloud installs `requirements.txt` from the repo root. `PYTHONPATH` is set in `app/streamlit_app.py` so `analysis_model` imports work without extra config.

The API key stays on the server. Anyone with the public URL can run analyses and consume your FMP quota. `api_tracker` still caps daily calls via `tracker.json`. Auth / a shared password can be added later.

### Backups

| Host | Notes |
| --- | --- |
| **Hugging Face Spaces** | New Space, SDK = Streamlit (or Docker). Secret `API`. URL: `https://huggingface.co/spaces/<user>/<space>`. |
| **Render / Railway / Fly.io** | Deploy `deploy/Dockerfile`, set env `API`. Use if you want a custom domain. Render’s free web service may spin down when idle. |

**Not a fit:** Vercel, Netlify, GitHub Pages (they do not run a Streamlit server).

## Docker (self-host / Render / Fly)

```bash
docker compose -f deploy/docker-compose.yml up --build
# http://localhost:8501
```

Set `API` in the environment or in local `api.env` (compose can load it; it is **not** copied into the image).

## Layout

```
app/streamlit_app.py          website entry
src/analysis_model/           package (data, analysis, graphs, pipeline)
deploy/                       Dockerfile + compose
main.py                       optional CLI
```
