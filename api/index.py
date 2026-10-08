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
        <p>Tune retrieval over Dublin City Council's Digital Transformation Strategy. Run the benchmark in the notebook, then submit your score here without exposing any database or model credentials in the browser.</p>
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
      <h2>Submit Score</h2>
      <p>Paste the JSON export from the notebook or fill in the score fields manually. Real names are shown on the leaderboard.</p>
      <label for="name">Real name</label>
      <input id="name" autocomplete="name" placeholder="Ada Lovelace" />
      <label for="code">Submission code, if provided by host</label>
      <input id="code" placeholder="Optional event code" />
      <label for="jsonExport">Notebook JSON export</label>
      <textarea id="jsonExport" placeholder='Paste { "summary": { "overall_score": ... } } here'></textarea>
      <div class="two">
        <div><label for="overall">Overall</label><input id="overall" type="number" min="0" max="100" step="0.0001" /></div>
        <div><label for="accuracy">Accuracy</label><input id="accuracy" type="number" min="0" max="100" step="0.0001" /></div>
        <div><label for="latency">Latency</label><input id="latency" type="number" min="0" max="100" step="0.0001" /></div>
        <div><label for="cost">Cost</label><input id="cost" type="number" min="0" max="100" step="0.0001" /></div>
      </div>
      <label for="notes">What changed?</label>
      <textarea id="notes" placeholder="Example: enabled hybrid search, chunk size 650, rerank top 3"></textarea>
      <button onclick="submitScore()">Submit Score</button>
      <button class="secondary" onclick="loadLeaderboard()">Refresh Leaderboard</button>
      <div id="status" class="status"></div>
    </section>

    <section class="card" style="margin-top:24px">
      <h2>Leaderboard</h2>
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
    function parseExport() {
      const raw = document.getElementById('jsonExport').value.trim();
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return parsed.summary || parsed;
    }
    async function submitScore() {
      const status = document.getElementById('status');
      status.textContent = 'Submitting...';
      status.className = 'status';
      try {
        const exported = parseExport();
        const payload = {
          name: document.getElementById('name').value,
          consent_public_name: true,
          submission_code: document.getElementById('code').value,
          notes: document.getElementById('notes').value,
          overall_score: exported.overall_score ?? document.getElementById('overall').value,
          accuracy_score: exported.accuracy_score ?? document.getElementById('accuracy').value,
          latency_score: exported.latency_score ?? document.getElementById('latency').value,
          cost_score: exported.cost_score ?? document.getElementById('cost').value
        };
        const res = await fetch('/api/submissions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Submission failed');
        status.textContent = 'Submitted.';
        status.className = 'status ok';
        await loadLeaderboard();
      } catch (err) {
        status.textContent = err.message;
        status.className = 'status warn';
      }
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
