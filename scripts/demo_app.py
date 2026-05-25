from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .improved_retrieval import RetrievalConfig, retrieve_evidence
from .query_understanding import understand_query


HOST = "127.0.0.1"
PORT = 8000


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NBA RAG Evidence Demo</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f7f4; color: #1f2933; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 24px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    form { display: grid; grid-template-columns: minmax(220px, 1fr) 170px 48px; gap: 10px; margin: 18px 0 14px; }
    input, select, button { height: 44px; border: 1px solid #c9c9c2; background: white; color: #1f2933; border-radius: 6px; font-size: 15px; padding: 0 12px; }
    button { cursor: pointer; background: #0f766e; color: white; border-color: #0f766e; font-weight: 700; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 16px; }
    .metric { background: white; border: 1px solid #deded8; border-radius: 8px; padding: 12px; }
    .metric b { display: block; font-size: 12px; color: #667085; margin-bottom: 4px; }
    .layout { display: grid; grid-template-columns: 280px 1fr; gap: 16px; align-items: start; }
    aside, .result { background: white; border: 1px solid #deded8; border-radius: 8px; }
    aside { padding: 14px; position: sticky; top: 12px; }
    .intent { font-size: 14px; line-height: 1.55; }
    .results { display: grid; gap: 10px; }
    .result { padding: 14px; }
    .result-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
    .type { font-size: 12px; font-weight: 800; color: #0f766e; text-transform: uppercase; }
    .score { font-variant-numeric: tabular-nums; color: #475467; font-size: 13px; }
    .meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .pill { border: 1px solid #d8d8d0; border-radius: 999px; padding: 3px 8px; font-size: 12px; color: #344054; background: #fbfbf8; }
    .answer { background: #e9f7f4; border: 1px solid #c8e7df; border-radius: 8px; padding: 14px; margin-bottom: 16px; line-height: 1.5; }
    @media (max-width: 760px) {
      main { padding: 18px; }
      header, .layout { display: block; }
      form { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr 1fr; }
      aside { position: static; margin-bottom: 12px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>NBA RAG Evidence Demo</h1>
    </div>
  </header>
  <form action="/" method="get">
    <input name="q" value="__QUERY__" placeholder="Ask about player stats, teams, recent games, matchups, or season averages">
    <select name="mode">
      __OPTIONS__
    </select>
    <button title="Search">Go</button>
  </form>
  __CONTENT__
</main>
</body>
</html>
"""


MODES = {
    "baseline": RetrievalConfig(metadata=False, dense=True, keyword=False, rerank=False),
    "metadata": RetrievalConfig(metadata=True, dense=True, keyword=False, rerank=False),
    "hybrid": RetrievalConfig(metadata=True, dense=True, keyword=True, rerank=False),
    "reranker": RetrievalConfig(metadata=False, dense=True, keyword=False, rerank=True),
    "full": RetrievalConfig(metadata=True, dense=True, keyword=True, rerank=True),
}


def grounded_answer(query: str, results) -> str:
    if not results:
        return "No matching evidence was retrieved."
    top = results[:3]
    lines = [f"Grounded answer draft for: {html.escape(query)}"]
    for result in top:
        lines.append(html.escape(result.text))
    return "<br>".join(lines)


def render(query: str, mode: str) -> str:
    mode = mode if mode in MODES else "full"
    options = "\n".join(
        f'<option value="{name}" {"selected" if name == mode else ""}>{name}</option>'
        for name in MODES
    )
    if not query:
        content = '<div class="answer">Try: "How did LeBron James play in his last game?" or "What happened in the latest matchup?"</div>'
        return PAGE.replace("__QUERY__", "").replace("__OPTIONS__", options).replace("__CONTENT__", content)

    intent = understand_query(query)
    results = retrieve_evidence(query, k=10, config=MODES[mode], use_db=False)
    top_score = f"{results[0].score:.3f}" if results else "0.000"
    summary = f"""
      <div class="summary">
        <div class="metric"><b>Mode</b>{html.escape(mode)}</div>
        <div class="metric"><b>Intent</b>{html.escape(intent.intent)}</div>
        <div class="metric"><b>Evidence</b>{len(results)} chunks</div>
        <div class="metric"><b>Top Score</b>{top_score}</div>
      </div>
    """
    intent_html = "<br>".join(
        f"<b>{key}</b>: {html.escape(str(value))}"
        for key, value in {
            "players": intent.players,
            "teams": intent.teams,
            "date": intent.date,
            "season": intent.season,
            "chunk types": intent.chunk_types,
        }.items()
    )
    cards = []
    for idx, result in enumerate(results, start=1):
        pills = []
        for label, value in (
            ("player", result.player),
            ("team", result.team),
            ("opponent", result.opponent),
            ("date", result.date),
            ("season", result.season),
            ("game", result.game_id),
            ("source", result.source_type),
        ):
            if value:
                pills.append(f'<span class="pill">{label}: {html.escape(str(value))}</span>')
        cards.append(
            f"""
            <article class="result">
              <div class="result-top">
                <span class="type">#{idx} {html.escape(result.chunk_type)}</span>
                <span class="score">{result.score:.3f}</span>
              </div>
              <div>{html.escape(result.text)}</div>
              <div class="meta">{''.join(pills)}</div>
            </article>
            """
        )
    content = f"""
      {summary}
      <div class="answer">{grounded_answer(query, results)}</div>
      <div class="layout">
        <aside><div class="intent">{intent_html}</div></aside>
        <section class="results">{''.join(cards)}</section>
      </div>
    """
    return PAGE.replace("__QUERY__", html.escape(query)).replace("__OPTIONS__", options).replace("__CONTENT__", content)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            mode = params.get("mode", ["full"])[0]
            results = retrieve_evidence(query, k=10, config=MODES.get(mode, MODES["full"]), use_db=False)
            payload = {
                "intent": understand_query(query).__dict__,
                "results": [result.__dict__ for result in results],
            }
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]
        mode = params.get("mode", ["full"])[0]
        body = render(query, mode).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"NBA RAG evidence demo running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
