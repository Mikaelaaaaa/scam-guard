# 強訊號信心級：實作檢查報告

## 中華郵政案例的前提漂移

測試文字：

```text
Chunghwa Post：包裹因關稅未繳而暫扣。請支付 159.71 元，付款請於此 ： https://e.vg/post-gov
```

以目前版控中的 `ngram_model.json` 重跑，分類器分數為 **0.234076**，低於門檻
**0.376648**；實際沒有任何訊號命中，信心為 **0.05**，`scam_probability=None`。
因此 proposal 所載「0.501、只有分類器命中」已不是目前模型的可重現結果。
`tests/fixtures/ngram_phishing_case.json` 已更新為目前模型的實測值。

分類器的 measured 權重為 **4.467675**，高於 `decision_score=1.5`。

## 強制命中情境的逐欄位記帳

為驗證 design D6，不把已漂移的模型輸出偽裝成命中；測試直接注入同形的
`CheckResult`，隔離檢查信心層契約。

| 情境 | base | cap_unseen_pattern | confidence | score | confidence gate | decision gate |
|---|---:|---:|---:|---:|---|---|
| 只有 `ngram_classifier`，類型空白 | 0.70 | 0.35（生效） | 0.35 | 4.467675 | 關閉，`scam_probability=None` | 分數達 1.5，但因信心閘門關閉而不判定 |
| `ngram_classifier` + 填出 `FAKE_PARCEL` 的 `llm_scam` | 0.70 | 不生效 | 0.70 | 5.067675 | 開啟，`scam_probability` 非 `None` | 分數達 1.5，判為詐騙 |

本 change **只開信心閘門**。判定卡是否為「很可能是詐騙」，仍取決於
`score.value >= decision_score`；`decision_score` 與所有訊號權重均未修改。

## Holdout 校準前後量測狀態

本 worktree 沒有不進版控的 `data/testset` holdout 原文，原始工作區亦無此資料，
所以無法在同一份 holdout 重跑校準前後。為避免把舊報告（其 registry 未掛載
分類器）冒充成這次量測，本報告不引用一組無法重現的前後數字。

待取得相同 holdout 後，必須用既有 `tools.eval.report.SubsetReport` / `Rate`
重跑並同時補上：hard-negative 誤判率、其 95% Wilson 上界、scam 召回率、棄權率。
在那之前 OpenSpec task 7.1 維持未完成。
