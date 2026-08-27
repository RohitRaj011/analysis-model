# Equity analysis (Streamlit)

Browser UI for Altman Z-Score, Dupoint ROE, cash conversion cycle (CCC), and DCF. Core calculation scripts live under `src/analysis_model/`.

## Local run

```bash
cd "/Users/rohit.raj/Downloads/Analysis Model"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example api.env   # set API= to your Financial Modeling Prep key
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Pick a ticker, choose analyses, optionally change DCF perpetual growth (default 2%; forecast is always 5 years), then click **Run analysis**.

Optional CLI (still no prompts):

```bash
PYTHONPATH=src python main.py --ticker AAPL --all
```

Dev tests:

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
```

## Public website (Streamlit Community Cloud)

This app is a long-running Python process, not static HTML. The default public host is **[Streamlit Community Cloud](https://share.streamlit.io)**.

1. Push this repo to GitHub (do **not** commit `api.env`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch, and main file: `app/streamlit_app.py`.
4. Under **Secrets**, add:

   ```toml
   API = "your-fmp-key"
   APP_PASSWORD = "optional-shared-password"
   ```

   `APP_PASSWORD` is optional. If set, visitors must enter it before running analyses.

5. Deploy. Users get a URL like `https://<app-name>.streamlit.app`.

The app hides Community Cloud Fork / GitHub toolbar links and the “Created by” viewer badge via `.streamlit/config.toml` (`toolbarMode = "minimal"`) and CSS. If those still appear, in the Cloud app **Settings** you can also hide GitHub / viewer identity.

The API key stays on the server. Anyone with the public URL can still consume FMP quota unless you set `APP_PASSWORD`. Limits:

- **Daily tracker:** 30 uncached analysis **runs** per calendar day (`tracker.json`). One run is one ticker fetch (~6 FMP HTTP calls), not one HTTP request. Re-running the same ticker the same day uses the `cache/` JSON and does not increment the tracker.
- **Session cap:** 15 successful runs per browser session.

### Rotate the FMP key

`api.env` was previously a local plaintext file and is gitignored. After cloning or if this folder was ever copied:

1. Create a new key in the [FMP dashboard](https://site.financialmodelingprep.com/developer/docs).
2. Put it only in local `api.env` and in Streamlit Cloud secrets (`API`).
3. Revoke the old key.

Never commit `api.env`.

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
app/tickers.py                sidebar ticker list
src/analysis_model/           package (data, analysis, graphs, pipeline)
tests/                        pytest fixtures for core identities
deploy/                       Dockerfile + compose
main.py                       optional CLI
```
