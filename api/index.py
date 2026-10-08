from http.server import BaseHTTPRequestHandler

from ._shared import text_response


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dublin MUG Retrieval Challenge</title>
  <style>
    :root { color-scheme: light; --green: #13aa52; --ink: #001e2b; --muted: #5c6c75; --line: #d8e1dd; --bg: #f6faf8; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
    main { max-width: 1080px; margin: 0 auto; padding: 48px 20px 72px; }
    .hero { display: grid; grid-template-columns: 1.4fr .9fr; gap: 24px; align-items: stretch; }
    .card { background: white; border: 1px solid var(--line); border-radius: 24px; padding: 24px; box-shadow: 0 18px 50px rgba(0, 30, 43, .06); }
    h1 { font-size: clamp(32px, 5vw, 58px); line-height: 1; margin: 0 0 18px; letter-spacing: -.04em; }
    h2 { font-size: 22px; margin: 0 0 16px; }
    p { color: var(--muted); line-height: 1.55; }
    .pill { display: inline-flex; gap: 8px; align-items: center; padding: 8px 12px; border-radius: 999px; background: #e8f7f0; color: #0b6b3a; font-weight: 700; font-size: 13px; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
    .metric { border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: #fbfefd; }
    .metric strong { display: block; font-size: 26px; }
    label { display: block; font-weight: 700; margin: 14px 0 7px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; font: inherit; background: white; }
    textarea { min-height: 110px; resize: vertical; }
    button { border: 0; border-radius: 999px; padding: 12px 18px; background: var(--green); color: white; font-weight: 800; cursor: pointer; margin-top: 16px; }
    button.secondary { background: var(--ink); }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { text-align: left; padding: 12px; border-bottom: 1px solid var(--line); }
    th { font-size: 13px; color: var(--muted); }
    .status { margin-top: 12px; font-weight: 700; }
    .warn { color: #8a5a00; }
    .ok { color: #0b6b3a; }
    @media (max-width: 820px) { .hero, .two, .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="card">
        <div class="pill">MongoDB User Group Dublin · 8 Oct 2026</div>
        <h1>Retrieval Challenge</h1>
      <p>Tune retrieval over Dublin City Council's Digital Transformation Strategy. Run the benchmark and submit your score from the notebook. This page shows the leaderboard only.</p>
        <p><strong>Submission deadline:</strong> 08 Oct 2026, 8:50 PM Dublin time.</p>
        <div class="grid">
          <div class="metric"><span>Questions</span><strong id="qCount">18</strong></div>
          <div class="metric"><span>Accuracy</span><strong>0.4</strong></div>
          <div class="metric"><span>Latency</span><strong>0.2</strong></div>
          <div class="metric"><span>Cost</span><strong>0.4</strong></div>
        </div>
      </div>
      <div class="card">
        <h2>Fixed Rules</h2>
        <p id="rules">Loading challenge config...</p>
        <p class="warn">Friendly challenge: benchmark answers are client-visible in the notebook. Use held-out evaluation for prizes.</p>
      </div>
    </section>

    <section class="card" style="margin-top:24px">
      <h2>Leaderboard</h2>
      <p>Scores are submitted from the participant notebook. Real names are shown on the leaderboard.</p>
      <button class="secondary" onclick="loadLeaderboard()">Refresh Leaderboard</button>
      <table>
        <thead><tr><th>Rank</th><th>Name</th><th>Overall</th><th>Accuracy</th><th>Latency</th><th>Cost</th></tr></thead>
        <tbody id="leaderboard"><tr><td colspan="6">Loading...</td></tr></tbody>
      </table>
    </section>
  </main>
  <script>
    async function loadConfig() {
      const res = await fetch('/api/config');
      const cfg = await res.json();
      document.getElementById('qCount').textContent = cfg.question_count;
      document.getElementById('rules').textContent = `Top K ${cfg.constraints.min_k}-${cfg.constraints.max_k}; chunk size ${cfg.constraints.min_chunk_size}-${cfg.constraints.max_chunk_size}; models: ${cfg.embedding_models.join(', ')}; rerankers: ${cfg.reranker_models.join(', ')}.`;
    }
    async function loadLeaderboard() {
      const body = document.getElementById('leaderboard');
      try {
        const res = await fetch('/api/submissions');
        const data = await res.json();
        const rows = data.submissions || [];
        const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[char]));
        body.innerHTML = rows.length ? rows.map(row => `<tr><td>${row.rank}</td><td>${escapeHtml(row.name)}</td><td>${Number(row.overall_score).toFixed(1)}</td><td>${Number(row.accuracy_score).toFixed(1)}</td><td>${Number(row.latency_score).toFixed(1)}</td><td>${Number(row.cost_score).toFixed(1)}</td></tr>`).join('') : '<tr><td colspan="6">No submissions yet, or leaderboard is not configured.</td></tr>';
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
        text_response(self, HTML)
