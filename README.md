---
title: scam-guard
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
---

# scam-guard

台灣中文詐騙訊息偵測。

## Demo

```bash
pip install -e ".[dev,demo]"
python app.py
```

介面有兩個模式：

- **詐騙對練** —— 你扮演詐騙方打字，受害方由系統回應，而受害方說的每一句都是
  判定結果的對話化呈現。它只會說判定裡有的東西。
- **這是詐騙嗎** —— 貼上你真的收到的訊息，輸出是判定、依據與建議動作。
  多則轉傳以**空行**分隔。

`app.py` 是介面層的唯一檔案。位置不是自由選擇：`pyproject.toml` 的 `banned-api`
對 `gradio` 的訊息已經指名它，`per-file-ignores` 也已經豁免它，而 HuggingFace
Spaces 固定執行 repo 根目錄的 `app.py`。

## 這個系統會不會把你的訊息送出去

**預設不會。** 訊息的正規化、切句、規則比對、165 涉詐網址黑名單比對、
TLD 風險、品牌相似度全部在本機完成，用的是預先下載好的本機快照。

**唯一的例外是網域年齡檢查（`domain_age`），而它預設不啟用。**
啟用它需要組裝層顯式注入一個查詢器（`net.rdap.RdapLookup`）；
不注入時系統照常運作，只是少一個訊號。

啟用之後會發生的事，逐項列舉：

- 訊息中每個連結的**可註冊網域**（例如 `https://login-esunbank.evil.com/a?token=abc`
  只取 `evil.com`）會被送到**該網域的註冊局**，查詢它的註冊日期。
- **不送**完整網址、路徑、query 參數、主機的子網域、訊息內容，
  也不送任何可識別你的資訊。
- 已知短網址服務的網域（`reurl.cc`、`bit.ly` 等）**不查詢**。
- 黑名單已命中時**不查詢** —— 最可疑的那批網域反而不會外流。
- 查過的網域會存進本機的 SQLite 快取（預設 `data/rdap/cache.sqlite3`），
  7 天內不重複查。清除用 `python -m tools.prune_rdap_cache`。

註冊局（以及網路路徑上的觀察者）因此會知道「某個 IP 在某個時間查了這個網域」。
它學不到是誰收到訊息、訊息內容是什麼、系統最後判成什麼。這比展開短網址
少非常多，但它不是零 —— 所以這個檢查預設關閉，開啟它是一個要顯式做的決定。

## 開發

```bash
python3 -m pytest -q        # 全部測試，不需要網路
python3 -m ruff check .     # lint，含架構界線（scam_guard/ 不得 import 網路函式庫）
```

`tests/test_gradio_demo.py` 以 `pytest.importorskip("gradio")` 開頭，本機沒裝
`[demo]` extra 時整個檔案會被跳過。CI 因此安裝 `.[dev,demo]` —— skipped 不會讓
CI 變紅，安靜跳過的測試等於沒有測試。

部署前需要的本機快照：

```bash
python -m tools.fetch_psl                # Public Suffix List
python -m tools.fetch_blocklist          # 165 涉詐網址黑名單
python -m tools.fetch_rdap_bootstrap     # IANA RDAP 端點對照（僅啟用 domain_age 時需要）
```

## 授權

本專案為 MIT（見 `LICENSE`）。

**例外：`demo_samples.json` 的內容為 CC BY-SA 4.0。** 該檔的訊息取自
[Cofacts 真的假的](https://cofacts.tw) 的開放資料，姓名標示以每一筆的
`source_uri` 滿足。散布該檔或其衍生內容時須以相同條款釋出，不適用本專案的 MIT。
