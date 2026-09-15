#!/usr/bin/env bash
# 產出 GitHub Pages 的靜態站台到 build/pages/。
#
# CI 與本機跑的是**同一個腳本**（`.github/workflows/pages.yml` 只呼叫它）——
# 兩份各自維護的步驟清單遲早會漂移，而漂移的那一天，本機驗過的東西與線上跑的
# 東西不是同一個，卻沒有任何地方會報告這件事。
#
# 產物全部是建置出來的，不進版控：wheel 由 `pyproject.toml` 產生、PSL 快照由
# `tools/fetch_psl.py` 當場下載。進版控的只有 `docs/` 裡的兩個手寫檔案。
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/build/pages"

cd "$root"
rm -rf "$out"
mkdir -p "$out"

# 站台自己的檔案：頁面、它的 Python 側，以及範例庫。
# `app.py` 不在這裡：它是 Gradio 介面，本站台不經過 Gradio（見 docs/pages_app.py）。
cp docs/index.html docs/pages_app.py demo_samples.json "$out/"

# 偵測核心以 wheel 交付，由瀏覽器內的 micropip 安裝。
python3 -m build --wheel --outdir "$out"

# PSL 快照。這是 `tools/fetch_psl.py` docstring 寫的第 1 條路（建置時取得）：
# 靜態站台沒有啟動階段，快照的新鮮度就等於最後一次部署的新鮮度。
python3 -m tools.fetch_psl --out "$out/psl"

# `index.html` 裡的 wheel 檔名是寫死的（它是一段靜態 HTML，沒有別的辦法）。
# 版本號一改，那一行就指向一個不存在的檔案，而後果是頁面載到一半才 404 ——
# 在這裡擋下來，讓它變成建置失敗。
wheel="$(cd "$out" && echo scam_guard-*.whl)"
if [ ! -f "$out/$wheel" ]; then
  echo "建置失敗：$out 下沒有產生 scam_guard 的 wheel" >&2
  exit 1
fi
if ! grep -q "$wheel" "$out/index.html"; then
  echo "建置失敗：docs/index.html 引用的 wheel 檔名與建出的不符，建出的是 $wheel" >&2
  echo '（版本號改過了嗎？index.html 的 requirements 那一行要一起改。）' >&2
  exit 1
fi

echo "站台已產出於 $out"
