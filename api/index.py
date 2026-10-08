from http.server import BaseHTTPRequestHandler
from http import HTTPStatus
from urllib.parse import urlparse

from ._shared import CHALLENGE_CONFIG, json_response, leaderboard_collection, load_leaderboard, save_submission_from_handler, text_response


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dublin MUG Retrieval Challenge</title>
  <style>
    :root { color-scheme: dark; --bg: #011e2b; --panel: #1c2d38; --panel-2: #112733; --line: #3d4f58; --line-hot: #00ed64; --green: #13aa52; --green-dark: #00684a; --text: #f9fafb; --muted: #b7c0c3; --soft: #e8edeb; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Euclid Circular A", "Helvetica Neue", Helvetica, Arial, sans-serif; background: radial-gradient(circle at top left, rgba(0, 237, 100, .14), transparent 34rem), var(--bg); color: var(--text); }
    main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 56px; }
    .shell { border: 1px solid var(--line); border-radius: 14px; background: rgba(28, 45, 56, .92); box-shadow: 0 24px 70px rgba(0, 0, 0, .28); overflow: hidden; }
    .widget-header { display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; padding: 22px 24px 16px; border-bottom: 1px solid var(--line); }
    .eyebrow { color: var(--line-hot); font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }
    h1 { margin: 0; font-family: "MongoDB Value Serif", Georgia, "Times New Roman", serif; font-size: clamp(34px, 5vw, 54px); line-height: 1; font-weight: 400; color: var(--text); }
    p { color: var(--muted); line-height: 1.55; margin: 10px 0 0; }
    .deadline { min-width: 230px; padding: 14px 16px; border: 1px solid #124151; border-radius: 12px; background: var(--panel-2); }
    .deadline span { display: block; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
    .deadline strong { display: block; margin-top: 5px; color: var(--soft); font-size: 15px; }
    .steps-nav { display: flex; gap: 22px; padding: 0 24px; border-bottom: 1px solid var(--line); }
    .step { padding: 13px 0 11px; color: var(--muted); font-size: 13px; font-weight: 600; border-bottom: 3px solid transparent; }
    .step.active { color: var(--soft); border-bottom-color: var(--line-hot); }
    .content-wrapper { padding: 18px; }
    .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
    .summary-card { border: 1px solid #124151; border-radius: 12px; background: var(--panel-2); padding: 14px; }
    .summary-card span { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
    .summary-card strong { display: block; margin-top: 8px; color: var(--text); font-size: 24px; line-height: 1; }
    .leaderboard-card { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); overflow: hidden; }
    .leaderboard-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px 18px; border-bottom: 1px solid var(--line); }
    h2 { margin: 0; font-family: "MongoDB Value Serif", Georgia, "Times New Roman", serif; font-size: 28px; font-weight: 400; }
    button { appearance: none; display: inline-flex; align-items: center; justify-content: center; height: 36px; padding: 0 14px; border-radius: 6px; border: 1px solid var(--line-hot); background: var(--green-dark); color: white; font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }
    button:hover { background: #00593f; box-shadow: 0 0 0 3px rgba(0, 237, 100, .16); }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 13px 16px; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); background: #0b2532; font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
    td { color: var(--soft); font-size: 14px; }
    tbody tr:hover { background: rgba(0, 237, 100, .05); }
    tbody tr:last-child td { border-bottom: 0; }
    .rank { color: var(--line-hot); font-weight: 700; }
    .score { font-variant-numeric: tabular-nums; font-weight: 700; }
    @media (max-width: 820px) { .widget-header { flex-direction: column; } .deadline { width: 100%; } .summary-grid { grid-template-columns: repeat(2, 1fr); } th, td { padding: 11px 10px; } }
    @media (max-width: 560px) { main { padding: 10px; } .summary-grid { grid-template-columns: 1fr; } .steps-nav { overflow-x: auto; } table { font-size: 12px; } }
  </style>
</head>
<body>
  <main>
    <section class="shell">
      <header class="widget-header">
        <div>
          <div class="eyebrow">MongoDB User Group Dublin · Retrieval Challenge</div>
          <h1>Leaderboard</h1>
          <p>Scores are submitted from the participant notebook after running the benchmark.</p>
        </div>
        <div class="deadline">
          <span>Submission Deadline</span>
          <strong>08 Oct 2026, 8:50 PM</strong>
        </div>
      </header>
      <nav class="steps-nav" aria-label="Challenge stages">
        <div class="step">Chunking</div>
        <div class="step">Embedding</div>
        <div class="step">Retrieval</div>
        <div class="step">Reranking</div>
        <div class="step active">Benchmark</div>
      </nav>
      <div class="content-wrapper">
        <div class="summary-grid">
          <div class="summary-card"><span>Questions</span><strong id="qCount">18</strong></div>
          <div class="summary-card"><span>Accuracy Weight</span><strong>0.4</strong></div>
          <div class="summary-card"><span>Latency Weight</span><strong>0.2</strong></div>
          <div class="summary-card"><span>Cost Weight</span><strong>0.4</strong></div>
        </div>
        <section class="leaderboard-card">
          <div class="leaderboard-head">
            <div>
              <h2>Top Runs</h2>
              <p>Best score per participant. Real names are shown.</p>
            </div>
            <button onclick="loadLeaderboard()">Refresh</button>
          </div>
          <table>
            <thead><tr><th>Rank</th><th>Name</th><th>Overall</th><th>Accuracy</th><th>Latency</th><th>Cost</th></tr></thead>
            <tbody id="leaderboard"><tr><td colspan="6">Loading...</td></tr></tbody>
          </table>
        </section>
      </div>
    </section>
  </main>
  <script>
    async function loadConfig() {
      const res = await fetch('/api/config');
      const cfg = await res.json();
      document.getElementById('qCount').textContent = cfg.question_count;
    }
    async function loadLeaderboard() {
      const body = document.getElementById('leaderboard');
      try {
        const res = await fetch('/api/submissions');
        const data = await res.json();
        const rows = data.submissions || [];
        const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[char]));
        body.innerHTML = rows.length ? rows.map(row => `<tr><td class="rank">#${row.rank}</td><td>${escapeHtml(row.name)}</td><td class="score">${Number(row.overall_score).toFixed(1)}</td><td>${Number(row.accuracy_score).toFixed(1)}</td><td>${Number(row.latency_score).toFixed(1)}</td><td>${Number(row.cost_score).toFixed(1)}</td></tr>`).join('') : '<tr><td colspan="6">No submissions yet.</td></tr>';
      } catch (err) {
        body.innerHTML = '<tr><td colspan="6">Could not load leaderboard.</td></tr>';
      }
    }
    loadConfig();
    loadLeaderboard();
  </script>
</body>
</html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/config":
            json_response(self, CHALLENGE_CONFIG)
            return
        if path == "/api/health":
            json_response(self, {"ok": True, "leaderboard_configured": leaderboard_collection() is not None})
            return
        if path == "/api/submissions":
            try:
                json_response(self, {"submissions": load_leaderboard(limit=50)})
            except Exception as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        text_response(self, HTML)

    def do_POST(self):
        try:
            save_submission_from_handler(self)
        except PermissionError as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.FORBIDDEN)
        except ValueError as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
