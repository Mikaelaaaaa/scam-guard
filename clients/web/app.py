"""單一伺服器 web demo —— 重用 `docs/index.html` 的版面，偵測改呼叫後端。

使用者的方案 B：像素級沿用 GitHub Pages 那份 `index.html`（樣式、版面完全一樣），
只把「瀏覽器內 Pyodide 跑四層」換成「呼叫後端 `/judge`」——後端跑同一個
`detect()`（四層，語意層 Gemini），回傳 `demo_ui` 渲染好的面板與細節。

與 Pages 的差別只有一個：偵測在後端、LLM 用 Gemini（key 藏伺服器不外洩）。
`scam_guard/` 一行不改；前端 HTML 直接讀 `docs/index.html` 做字串手術，不維護第二份。

    export $(cat .env | xargs)
    uvicorn clients.web.app:app --port 8100
    cloudflared tunnel --url http://localhost:8100
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse

import demo_ui
from clients.line.registry import REGISTRY, TABLE
from llm_runtime.gemini import GeminiCallFailed
from scam_guard.normalize import DEFAULT_LIMITS, build_document
from scam_guard.pipeline import detect
from scam_guard.types import Message, Request

logger = logging.getLogger("scam_guard.web")

INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "docs" / "index.html"

LLM_FAILED_PANEL = (
    '<section class="analysis-panel card"><div class="card-head">'
    '<div class="card-title">語意判讀失敗</div></div>'
    "<p>本次無法完成判定，語意層（Gemini）呼叫失敗，請稍後再試。</p></section>"
)

app = FastAPI(title="scam-guard web demo")


def _samples_json() -> str:
    """把 demo 範例（`app.SAMPLES`）序列化給前端填 chips 用。

    在函式內 import `app` 而非模組頂端 —— `app.py` 是 Gradio 介面層，import 它會拉進
    gradio，而這個 web demo 不需要 gradio。範例本身只是三個字串欄位。
    """
    import app as gradio_app

    return json.dumps(
        [{"short_label": s.short_label, "text": s.text} for s in gradio_app.SAMPLES],
        ensure_ascii=False,
    )


_FRONT_SCRIPT = """<script>
(function(){{
  const input=document.getElementById('input'), run=document.getElementById('run');
  const card=document.getElementById('card'), details=document.getElementById('inquiry-details');
  const chips=document.getElementById('samples');
  input.disabled=false; run.disabled=false;
  input.placeholder='把收到的可疑訊息貼在這裡…';
  const SAMPLES={samples};
  SAMPLES.forEach(function(s){{
    const b=document.createElement('button');
    b.type='button'; b.className='chip'; b.textContent=s.short_label;
    b.onclick=function(){{ input.value=s.text; }};
    chips.appendChild(b);
  }});
  run.onclick=async function(){{
    const text=input.value.trim(); if(!text)return;
    run.disabled=true; card.innerHTML='<p class="note">判讀中…</p>'; details.innerHTML='';
    try{{
      const r=await fetch('/judge',{{method:'POST',
        headers:{{'Content-Type':'application/x-www-form-urlencoded'}},
        body:'text='+encodeURIComponent(text)}});
      const j=await r.json();
      card.innerHTML=j.panel; details.innerHTML=j.details;
    }}catch(e){{ card.innerHTML='<p class="note">連線失敗：'+e+'</p>'; }}
    run.disabled=false;
  }};
}})();
</script>"""


def _page() -> str:
    """讀 `docs/index.html`，做最小字串手術後回傳：注入 demo_ui.CSS、顯示主畫面、
    砍掉綁 Pyodide 的 script、隱藏模式切換與模型列與對練面板、接上呼叫後端的新 script。
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    html = html.replace(
        '<style id="shared-style"></style>',
        f'<style id="shared-style">{demo_ui.CSS}</style>',
        1,
    )
    html = html.replace('<div id="main-app" hidden>', '<div id="main-app">', 1)
    html = html.replace('<div class="boot" id="boot">', '<div class="boot" id="boot" hidden>', 1)
    html = html.replace(
        '<div class="modes" id="modes">', '<div class="modes" id="modes" hidden>', 1
    )
    html = html.replace(
        '<div class="model" id="model" role="status">', '<div class="model" hidden>', 1
    )
    html = html.replace('<section id="panel-practice">', '<section id="panel-practice" hidden>', 1)
    front = _FRONT_SCRIPT.format(samples=_samples_json())
    return html[: html.index('<script type="module">')] + front + "</body></html>"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page()


@app.post("/judge")
def judge(text: str = Form(...)) -> JSONResponse:
    """一則訊息 → {panel, details}，分別填進 `#card` 與 `#inquiry-details`。

    語意層（Gemini，必備）失敗時面板回失敗卡、細節留空，不降級成只有規則的結果。
    """
    request = Request(messages=[Message(text=text)])
    document = build_document(request.messages, DEFAULT_LIMITS)
    try:
        verdict = detect(request, REGISTRY, TABLE, limits=DEFAULT_LIMITS)
    except GeminiCallFailed as error:
        logger.warning("語意層失敗：%s", error)
        return JSONResponse({"panel": LLM_FAILED_PANEL, "details": ""})
    return JSONResponse(
        {
            "panel": demo_ui.render_analysis_panel(verdict, TABLE, ()),
            "details": demo_ui.render_detection_details(verdict, document, ()),
        }
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {"service": "scam-guard-web", "layers": len(REGISTRY.enabled())}
