"""模型檔的取得。**版本釘住 commit sha，不用分支名。**

三條路先排除：進版控（806 MB，需要 LFS 配額且每次 clone 都付）；走 `tools/`
預先下載到 `data/`（該目錄被 `.gitignore` 排除，Space 上 clone 出來的 repo
沒有它 —— `add-gradio-chat` 已經踩過這個坑）；內嵌（同樣是 806 MB）。

採用 `hf_hub_download()`，因為這條路徑在 HuggingFace Spaces 上是**原生**的：
容器裡本來就有 `HF_HOME`，走 HF 自己的 CDN，且它以 etag 與 sha 驗證完整性、
重複呼叫時直接命中快取。我們不需要自己寫下載、不需要自己驗 checksum、
不需要自己管快取目錄 —— 這三件事 `tools/fetch_rdap_bootstrap.py` 都得自己做。

⚠️ **免費層休眠之後快取會消失**，檔案系統回到映像檔狀態，所以每次冷啟動要
重新下載 806 MB（HF 內網 CDN **估算** 10–30 MB/s，即 30–90 秒）。
**這不是一個可以技術緩解的問題** —— 持久化儲存是付費功能。
處置：把「示範前先喚醒 Space」寫進操作注意事項。
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ggml-org/gemma-3-1b-it-GGUF"
"""未 gated 的社群量化倉庫（llama.cpp 上游組織的帳號）。

**不用 `google/gemma-3-1b-it-qat-q4_0-gguf`。** 2026-09-14 與 2026-09-15 兩次查
HuggingFace API，該 repo 的 `gated` 欄位皆為 `"manual"`，下載數 831 ——
它需要人工同意 Google 的授權並在下載時帶 token。核心資產若依賴一個需要人工
核准才能取得的東西，「這個 Space 能不能從零建起來」就不再是一個技術問題。
`add-pii-recognizers` 為完全相同的理由否決過一個模型。

| repo | gated | 下載數 | Q4 大小 |
|---|---|---|---|
| `google/gemma-3-1b-it-qat-q4_0-gguf` | **manual** | 831 | 1,003,541,152 B |
| `ggml-org/gemma-3-1b-it-GGUF` | 否 | 193,991 | 806,058,240 B |
| `MaziyarPanahi/gemma-3-1b-it-GGUF` | 否 | 149,427 | 806,058,272 B |

⚠️ **代價要誠實說：QAT（量化感知訓練）的 q4 品質通常優於事後量化的 Q4_K_M，
而我們放棄的正是那個品質。** 差多少沒有數字 —— 沒有中文詐騙語料的評測，
也沒有這兩個檔案的對照實驗。`add-ablation` 可以把它列為一個可選的對照組
（需要有人先同意授權並提供 token），但**預設路徑 MUST 是未 gated 的那個**。
"""

FILENAME = "gemma-3-1b-it-Q4_K_M.gguf"
"""806,058,240 bytes（2026-09-15 實測，與 2026-09-14 相同）。"""

REVISION = "f9c28bcd85737ffc5aef028638d3341d49869c27"
"""**釘住 commit sha，MUST NOT 用 `main`。**

`main` 會變 —— 上傳者重新量化、修正 tokenizer 中繼資料、改 chat template，
都會產生一個新的 commit 而檔名不變。模型檔變了之後 grammar 的行為、
tokenizer 的切法、輸出的品質都可能跟著變，**而沒有任何地方會報告它變了**。
與 `add-blocklist-store` 寫 manifest（`source_url`、`fetched_at`、`sha256`）
是同一個理由：外部資產的版本必須是我們記下來的，不是對方當下給的。

取得日期 2026-09-15，該 sha 下 `FILENAME` 為 806,058,240 bytes。
"""


def ensure_model() -> Path:
    """取得模型檔的本機路徑，必要時下載。

    不自行寫快取、不自行驗 checksum、不提供任何「找不到就用別的檔案」的分支 ——
    一個會安靜換掉模型的 fallback，會讓所有量測結果失去意義。
    下載失敗時 `huggingface_hub` 的例外直接傳播。
    """
    return Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION))
