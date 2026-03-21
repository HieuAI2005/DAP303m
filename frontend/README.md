# MovieRAG Web Frontend

## Start the API

From the project root:

```bash
cd src
python -m movierag.main api --port 8000
```

## Start the Vite frontend

From the `frontend/` directory:

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api/*` requests to `http://127.0.0.1:8000`.

## What the UI covers

- `Overview`: current index and library readiness
- `Ingest`: upload a new video and optional subtitle
- `Library`: inspect stored artifacts per `movie_id`
- `Search & QA`: run retrieval and verify evidence
