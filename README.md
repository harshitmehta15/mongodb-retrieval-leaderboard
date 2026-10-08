# Dublin MUG Retrieval Challenge Leaderboard

Minimal public repo for the Dublin MUG Retrieval Challenge leaderboard.

This repo contains:

- `api/`: Vercel Python serverless API and leaderboard UI.
- `third_party/mongodb-ai-widget/`: vendored challenge widget source, for optional Colab installation from this repo.

Vercel should deploy only the root leaderboard app. The vendored widget is ignored during Vercel deployment via `.vercelignore` and is not a Vercel service.

It does not contain participant credentials, MongoDB connection strings, model API keys, host notes, source notebooks, benchmark answers, or event-internal reports.

## Deploy To Vercel

Import this repo into Vercel and deploy from the repository root.

Required environment variables:

- `LEADERBOARD_MONGODB_URI`
- `LEADERBOARD_DATABASE`
- `LEADERBOARD_COLLECTION`

Optional environment variable:

- `SUBMISSION_CODE`

## Routes

- `/`: leaderboard and score submission UI.
- `/api/config`: challenge metadata.
- `/api/submissions`: `GET` leaderboard and `POST` score submission.
- `/api/health`: health check.

## Notebook Integration

After Vercel deployment, set the participant notebook submission URL to:

```python
SUBMISSION_API_URL = "https://YOUR-VERCEL-APP.vercel.app/api/submissions"
```

If the notebook should install the vendored widget from this repo, use:

```python
EVENT_REPO_URL = "https://github.com/harshitmehta15/mongodb-retrieval-leaderboard"
```

Submission deadline: 08 Oct 2026, 8:50 PM Dublin time.

Real names are shown on the leaderboard.
