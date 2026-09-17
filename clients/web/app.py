"""單一伺服器的 web demo —— 前端 HTML 與判定都由這個 FastAPI 服務。

使用者的決定（方案 A）：一個 server、一個網址。前端是手刻頁面（樣式取自
`demo_ui.CSS`，與 GitHub Pages 版一致），送出時 POST 到 `/judge`，後端跑
`detect()`（四層，語意層是 Gemini）再把 `demo_ui` 渲染好的判定卡回傳，前端直接
塞進畫面。

與 GitHub Pages（Pyodide + 瀏覽器 Gemma）的差別只有一個：LLM 在後端用 Gemini，
key 藏在伺服器不外洩。樣式共用 `demo_ui`，所以長得一樣。

    export $(cat .env | xargs)
    uvicorn clients.web.app:app --port 8100
    cloudflared tunnel --url http://localhost:8100
"""

import logging

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import demo_ui
from clients.line.registry import LLM_ENABLED, REGISTRY, TABLE
from llm_runtime.gemini import GeminiCallFailed
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.pipeline import detect
from scam_guard.types import Message, Request

logger = logging.getLogger("scam_guard.web")

LLM_FAILED_HTML = (
    '<section class="analysis-panel card"><div class="card-title">語意判讀失敗</div>'
    "<p>本次無法完成判定，語意層（Gemini）呼叫失敗，請稍後再試。</p></section>"
)

app = FastAPI(title="scam-guard web demo")

_PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>這是詐騙嗎 · scam-guard</title>
<style>{css}
body{{max-width:820px;margin:0 auto;padding:24px;
background:var(--body-background-fill,#0b0f17);color:var(--body-text-color,#e6e8ec);
font-family:system-ui,-apple-system,"Noto Sans TC",sans-serif;}}
textarea{{width:100%;min-height:120px;padding:12px;border-radius:10px;
border:1px solid #333;background:#141922;color:inherit;font:inherit;box-sizing:border-box;}}
button{{margin-top:12px;padding:10px 20px;border-radius:10px;border:0;
background:#6b8afd;color:#fff;font:inherit;cursor:pointer;}}
#result{{margin-top:20px;}}
</style></head>
<body>
<h1>這是詐騙嗎</h1>
<p>貼上你收到的可疑訊息，系統用四層（網址、規則、分類器、語意）判斷。</p>
<p>語意層由 Gemini 判讀（{llm}）。</p>
<textarea id="msg" placeholder="把收到的訊息貼在這裡…"></textarea>
<button id="go">看看這是不是詐騙</button>
<div id="result"></div>
<script>
const go=document.getElementById('go');
const msg=document.getElementById('msg'),out=document.getElementById('result');
go.onclick=async()=>{{
  const text=msg.value.trim(); if(!text){{out.textContent='請先貼上訊息。';return;}}
  go.disabled=true; out.textContent='判讀中…';
  try{{
    const r=await fetch('/judge',{{method:'POST',
      headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
      body:'text='+encodeURIComponent(text)}});
    out.innerHTML=await r.text();
  }}catch(e){{out.textContent='連線失敗：'+e;}}
  go.disabled=false;
}};
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    llm = "已啟用" if LLM_ENABLED else "未啟用（缺 GEMINI_API_KEY）"
    return _PAGE.format(css=demo_ui.CSS, llm=llm)


@app.post("/judge", response_class=HTMLResponse)
def judge(text: str = Form(...)) -> str:
    """一則訊息 → 判定卡 HTML。語意層失敗回失敗卡，不降級。"""
    request = Request(messages=[Message(text=text)])
    document = build_document(request.messages, DEFAULT_LIMITS)
    try:
        verdict = detect(request, REGISTRY, TABLE, limits=DEFAULT_LIMITS)
    except GeminiCallFailed as error:
        logger.warning("語意層失敗：%s", error)
        return LLM_FAILED_HTML
    return demo_ui.render_verdict_card(verdict, document, TABLE, ())


@app.get("/health")
def health() -> dict[str, object]:
    return {"service": "scam-guard-web", "layers": len(REGISTRY.enabled()), "llm": LLM_ENABLED}
