#!/usr/bin/env bash
# 產出 GitHub Pages 的靜態站台到 build/pages/。
#
# CI 與本機跑的是**同一個腳本**（`.github/workflows/pages.yml` 只呼叫它）——
# 兩份各自維護的步驟清單遲早會漂移，而漂移的那一天，本機驗過的東西與線上跑的
# 東西不是同一個，卻沒有任何地方會報告這件事。
#
# 產物全部是建置出來的，不進版控：wheel 由 `pyproject.toml` 產生，PSL、165 涉詐
# 網址黑名單與 Tranco 排名白名單三份快照由 `tools/` 下的取得程式當場下載
# （`.gitignore` 的 `build/`、`data/` 與 `*.jsonl` 三條規則涵蓋它們）。
# 進版控的只有 `docs/` 裡的兩個手寫檔案。
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/build/pages"

cd "$root"
rm -rf "$out"
mkdir -p "$out"

# 站台自己的檔案：頁面、它的 Python 側、兩份介面層共用的標記層、瀏覽器側的
# LLM 那一半，以及範例庫。
# `app.py` 不在這裡：它是 Gradio 介面，本站台不經過 Gradio（見 docs/pages_app.py）。
# `demo_ui.py` 與 `browser_llm.py` 在根目錄而不在 wheel 裡（`packages.find` 只收
# `scam_guard*`），所以它們跟 `pages_app.py` 一樣靠這一行複製過去。
#
# ⚠️ 模型檔不在這裡，也不該在這裡：那 783,858,998 bytes 由使用者的瀏覽器直接向
# huggingface.co 取得。站台的 1 GB 上限與 100 GB／月頻寬都不被它佔用。
cp docs/index.html docs/pages_app.py demo_ui.py browser_llm.py demo_samples.json "$out/"

# 偵測核心以 wheel 交付，由瀏覽器內的 micropip 安裝。
python3 -m build --wheel --outdir "$out"

# PSL 快照。這是 `tools/fetch_psl.py` docstring 寫的第 1 條路（建置時取得）：
# 靜態站台沒有啟動階段，快照的新鮮度就等於最後一次部署的新鮮度。
python3 -m tools.fetch_psl --out "$out/psl"

# 165 涉詐網址黑名單。與上面的 PSL 完全同形，產物是取得程式的輸出**原樣** ——
# 不精簡、不改寫、不轉格式，因為 `BlocklistStore.load()` 只接受這個格式，
# 而任何自製格式都會讓 sha256 比對、manifest 欄位檢查、授權檢查與新鮮度檢查
# 一併失效，並在 `docs/` 側多出一支沒有測試覆蓋的解析器。
#
# **不傳 `--allow-shrink`。** 一份異常縮水的名單上線的後果是召回率系統性下降，
# 而畫面上看起來一切正常；讓建置失敗、上一版部署原地保留才是對的。
python3 -m tools.fetch_blocklist --out "$out/blocklist"

# Tranco 排名白名單。它與黑名單成對：`docs/pages_app.py` 任一份載不起來就兩份
# 都不用。理由是 160055 收錄了 `play.google.com`，沒有白名單時一則直接連向它的
# 正常訊息會取得 `hard=True` 並短路掉整個 LLM 層。
#
# `--psl` 指向上面剛取得的那一份：取得程式要用 PSL 丟掉本身即為 public suffix
# 的項目，而那必須與線上執行時用的是同一份。
python3 -m tools.fetch_tranco --out "$out/allowlist" --psl "$out/psl"

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

# 共用的標記層與瀏覽器側的 LLM 那一半都不在 wheel 裡，漏掉那一行 `cp` 的後果是
# 頁面載到 `pyimport` 那一步才拋 ModuleNotFoundError。比照上面的 wheel 檔名檢查，
# 在這裡擋成建置失敗。
for module in demo_ui.py browser_llm.py pages_app.py; do
  if [ ! -f "$out/$module" ]; then
    echo "建置失敗：$out 下沒有 $module，頁面會在 pyimport 階段找不到模組" >&2
    exit 1
  fi
done

# 兩份新快照各兩個檔案。缺任何一個的後果與少一個模組相同：頁面載到一半才 404，
# 而那時已經部署上去了。比照上面的 wheel 與模組檢查，在這裡擋成建置失敗。
for snapshot in blocklist allowlist; do
  for name in entries.jsonl manifest.json; do
    if [ ! -f "$out/$snapshot/$name" ]; then
      echo "建置失敗：$out/$snapshot/$name 不存在，頁面會在載入快照時 404" >&2
      exit 1
    fi
  done
done

# 傳輸量的上限。GitHub Pages 對靜態檔案套用 gzip（線上實測
# `psl/public_suffix_list.dat` 334,129 B → `content-encoding: gzip`、
# `content-length: 90,674`，`content-type: application/octet-stream`，
# 而且**沒有 brotli**），所以這裡量到的 gzip 後大小就是使用者實際要下載的量。
#
# 2026-09-15 實測值：黑名單 1,268,967 B、白名單 8,212 B（`gzip -9`；GitHub Pages
# 的邊緣壓縮實測落在 level 6 附近，兩者差 2% 以內）。門檻取現值的約兩倍。
# **門檻與實測值一併寫在這裡**，否則下一個撞到門檻的人沒有東西可以比對，
# 只能把門檻調大 —— 而沒有任何機制阻止 176455 下個月變成兩倍大，
# 發現的方式不該是「使用者說頁面變慢了」。
check_gzip_size() {
  actual="$(gzip -9 -c "$1" | wc -c | tr -d ' ')"
  if [ "$actual" -gt "$2" ]; then
    echo "建置失敗：$1 的 gzip 後大小為 $actual B，超過門檻 $2 B" >&2
    echo '（資料集是不是改版了？確認成長合理之後再調整本腳本裡的門檻與實測值。）' >&2
    exit 1
  fi
  echo "$1 gzip 後 $actual B（門檻 $2 B）"
}

check_gzip_size "$out/blocklist/entries.jsonl" 2500000
check_gzip_size "$out/allowlist/entries.jsonl" 200000

# 快照寫出之後**實際載入一次**，參數與 `docs/pages_app.py` 部署時用的完全相同。
# 取得程式與 `load()` 是同一次 CI 執行的兩端，但取得程式不會呼叫 `load()` ——
# 沒有這一步，資料集改版造成的 manifest 欄位缺漏會在**部署之後**才於瀏覽器裡
# 顯現，而那時上一版已經被覆蓋掉了。
#
# 這一步同時驗證 sha256、manifest 欄位、授權與新鮮度四件事，成本是本機的 0.3 秒。
# `require_redistributable` 用預設的 `True`，**MUST NOT 傳 `False`** ——
# 本站台的產出就是給第三方看的判斷依據。
python3 -c '
import sys

from scam_guard.allowlist import RankAllowlist
from scam_guard.blocklist import BlocklistStore
from scam_guard.url import PublicSuffixList

out = sys.argv[1]
psl = PublicSuffixList.load(f"{out}/psl", max_age_days=90)
store = BlocklistStore.load(f"{out}/blocklist", psl, max_age_days={"176455": 60, "165027": 60})
allowlist = RankAllowlist.load(f"{out}/allowlist", max_age_days=30)
print(f"快照載入成功：黑名單 {store.entry_count} 筆、白名單 {allowlist.entry_count} 筆")
' "$out"

# `index.html` 以 ESM 從 CDN 取 transformers.js，版本號寫死在那個 URL 裡。
# 它與 design 記下的版本一改就分家，而後果是一個我們沒有驗證過的函式庫在使用者
# 的瀏覽器裡跑我們的 prompt。比照 wheel 檔名，在這裡擋成建置失敗。
transformers_version="4.2.0"
if ! grep -q "@huggingface/transformers@$transformers_version" "$out/index.html"; then
  echo "建置失敗：docs/index.html 引用的 @huggingface/transformers 版本與建置腳本記下的不符" >&2
  echo "（腳本記的是 $transformers_version；改版本要同時改這兩處與 design。）" >&2
  exit 1
fi
if grep -q "@huggingface/transformers@latest" "$out/index.html"; then
  echo "建置失敗：docs/index.html 用了 @latest —— 外部資產的版本必須是我們記下來的" >&2
  exit 1
fi

# 模型檔進站台是一個會安靜發生的錯（有人為了「離線也能用」把它 cp 進來），
# 而後果是 728 MiB 佔掉站台 1 GB 上限的七成五，外加每月 100 GB 頻寬只夠 127 次下載。
if find "$out" -type f \( -name '*.onnx' -o -name '*.onnx_data' -o -name '*.gguf' \) |
  grep -q .; then
  echo "建置失敗：$out 之下出現模型檔。模型由瀏覽器直接向 huggingface.co 取得，不進站台" >&2
  exit 1
fi

echo "站台已產出於 $out"
