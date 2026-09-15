# 權重校準報告

把 `weights.toml` 有資料的 `placeholder` 條目改成 `measured`。量測只用 tune
（572 則 scam、824 則 ham）；兩個條件機率、Wilson 區間與可估性三態全部取自
`tools/eval/signals.py`，本 change 只是它的執行者。

## 四個出口

每個權重條目（訊號 × hard/soft）恰好走一個出口。`ngram_classifier` 的兩個
條目不在此列 —— 它們由 `add-ngram-classifier` 在同一份 tune 上量出，本 change 不重量。

| 出口 | 條目數 |
|---|---|
| measured（兩側皆命中且區間寬度 ≤ 1.9） | 2 |
| estimable_but_wide（兩側皆命中但寬度 > 1.9） | 10 |
| lower_bound_only（ham 側 0 命中） | 3 |
| not_estimable（scam 側 0 命中） | 31 |
| **合計** | **46** |

46 個非分類器條目裡只有 2 個量得出 measured，是一個關於這份語料的結論，不是本 change 未完成：
Cofacts 的詐騙訊息多是轉傳型敘述文字，而規則層瞄準的是祈使型言語行為，
在 1,396 則 tune 上有 31 個條目 scam 側零命中。

### 逐條目明細（含兩側命中數）

| 出口 | 訊號 | 條目 | scam 命中 | ham 命中 | 區間寬度 | 值／權重下界 |
|---|---|---|---|---|---|---|
| estimable_but_wide | evasion_invisible | weight_soft | 1/572 | 4/824 | 5.344 | — |
| estimable_but_wide | evasion_split_word | weight_soft | 1/572 | 2/824 | 6.044 | — |
| estimable_but_wide | guaranteed_return | weight_soft | 2/572 | 1/824 | 6.043 | — |
| estimable_but_wide | high_pay_no_skill | weight_soft | 6/572 | 3/824 | 3.701 | — |
| estimable_but_wide | identity_docs | weight_soft | 3/572 | 4/824 | 4.031 | — |
| estimable_but_wide | remote_control_tool | weight_soft | 7/572 | 38/824 | 2.057 | — |
| estimable_but_wide | safe_account | weight_soft | 3/572 | 3/824 | 4.300 | — |
| estimable_but_wide | url_blocklist | weight_hard | 29/572 | 3/824 | 2.856 | — |
| estimable_but_wide | url_blocklist | weight_soft | 12/572 | 5/824 | 2.798 | — |
| estimable_but_wide | url_brand | weight_soft | 2/572 | 2/824 | 5.162 | — |
| lower_bound_only | atm_operation | weight_soft | 3/572 | 0/824 | — | w ≥ 0.1224 |
| lower_bound_only | deliver_bank_instrument | weight_hard | 1/572 | 0/824 | — | w ≥ -0.9762 |
| lower_bound_only | url_tld_risk | weight_soft | 7/572 | 0/824 | — | w ≥ 0.9697 |
| measured | quotation | weight_soft | 26/572 | 180/824 | 1.003 | value = -1.569821 |
| measured | url_shortener | weight_soft | 7/572 | 86/824 | 1.838 | value = -2.143387 |
| not_estimable | atm_operation | weight_hard | 0/572 | 1/824 | — | — |
| not_estimable | charity_personal_account | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | deliver_bank_instrument | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | domain_age | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | escort_deposit | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | evasion_homophone | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | evasion_width_mix | weight_soft | 0/572 | 1/824 | — | — |
| not_estimable | evasion_zhuyin | weight_soft | 0/572 | 5/824 | — | — |
| not_estimable | game_code | weight_soft | 0/572 | 4/824 | — | — |
| not_estimable | identity_reset | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | llm_scam | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | llm_suspicious | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | loan_no_check | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | parcel_notice | weight_soft | 0/572 | 1/824 | — | — |
| not_estimable | prepay_to_receive | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | prepay_to_receive | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | relationship_building | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | remote_control_tool | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | romance_pretext | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | safe_account | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | secrecy_demand | weight_hard | 0/572 | 1/824 | — | — |
| not_estimable | secrecy_demand | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | seller_verification | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | seller_verification | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | solicit_bank_credentials | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | solicit_bank_credentials | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | solicit_card_secret | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | solicit_card_secret | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | solicit_otp | weight_hard | 0/572 | 0/824 | — | — |
| not_estimable | solicit_otp | weight_soft | 0/572 | 0/824 | — | — |
| not_estimable | url_host_shape | weight_soft | 0/572 | 2/824 | — | — |

### 改為 measured 的條目

- **quotation / weight_soft**：`value = -1.569821` = ln(0.045455 / 0.218447)
  - `p_hit_given_scam` = 4.55%（26/572，95% CI 3.12%–6.58%）
  - `p_hit_given_ham` = 21.84%（180/824，95% CI 19.16%–24.79%）
  - 區間寬度 1.003 ≤ 1.9（1.9 = 2.5 − 0.6，Tier-A 與 Tier-B 佔位值的間距）
- **url_shortener / weight_soft**：`value = -2.143387` = ln(0.012238 / 0.104369)
  - `p_hit_given_scam` = 1.22%（7/572，95% CI 0.59%–2.50%）
  - `p_hit_given_ham` = 10.44%（86/824，95% CI 8.53%–12.71%）
  - 區間寬度 1.838 ≤ 1.9（1.9 = 2.5 − 0.6，Tier-A 與 Tier-B 佔位值的間距）

兩條都是**負權重**：引述使一則訊息更可能是轉述而非施行，短網址在此語料上
更常出現在合法宣導與廣告裡。校準把 `url_shortener` 從佔位的 `0.0`（宣告資訊
不足）改成量出來的 `-2.143387`，這是本 change 最大的單一權重變化。

### lower_bound_only：ham 側零命中，只報下界（不進表）

- **deliver_bank_instrument / weight_hard**：scam 側 1/572、ham 側 0 命中，權重下界 `w ≥ -0.9762`。條目維持 `placeholder`，下界只進報告
- **atm_operation / weight_soft**：scam 側 3/572、ham 側 0 命中，權重下界 `w ≥ 0.1224`。條目維持 `placeholder`，下界只進報告
- **url_tld_risk / weight_soft**：scam 側 7/572、ham 側 0 命中，權重下界 `w ≥ 0.9697`。條目維持 `placeholder`，下界只進報告

下界 MUST NOT 寫進 `value`：`p_ham = 0` 不落在開區間 `(0, 1)`，寫成 `measured`
載入即拋例外；寫成 `placeholder` 則值必須是四個佔位常數之一。表的驗證正確地
擋住這個看起來很合理的錯誤。

## decision_score

- 結果：維持原值（約束不可滿足），`value = 1.5`
- rationale：約束為「單一 measured 弱訊號不跨過門檻、單一 measured 硬證據跨過門檻」。measured 弱訊號權重的下界為 ngram_classifier 的 4.467675，沒有任何 weight_hard 條目量得出 measured（全部 hard_capable 訊號在 tune 上的硬命中數為 0），區間不可滿足，故 decision_score 維持 1.5、不挑一個數字。這代表在此語料上規則層幾乎不產生硬證據，門檻無法由量測重新錨定

**這是本 change 的主要發現之一。** decision_score 的舊 rationale 引用「單一 Tier-A
命中（2.5）跨過、單一 Tier-B 命中（0.6）不跨過」，而 2.5 / 0.6 這兩個錨點被本
change 刪除。重算需要 measured 的 weight_soft 當下界、measured 的 weight_hard 當
上界，但 tune 上沒有任何 weight_hard 量得出 measured（全部 hard_capable 訊號的硬
命中數為 0），而唯一夠強的 measured 弱訊號是門檻在同一份 tune 上選過的
`ngram_classifier`（4.467675）。區間因此不可滿足，decision_score 維持 1.5。

## 校準前後對照（holdout，1,467 則）

兩份指標由同一個報告產生器（`report.subset_report`）產出，差別只有輸入的表。
**誤判率一律附 95% Wilson 上界，且與棄權率綁定**（`SubsetReport` 缺棄權率即建構失敗）。
`self_sms_ham`（真實銀行／物流／政府通知）未蒐集，所以下表的誤判率量不到最危險的那一類。

| 子集 | 指標 | 校準前 | 校準後 |
|---|---|---|---|
| cofacts_scam | 召回率 | 5.73%（36/628，95% CI 4.17%–7.83%） | 5.57%（35/628，95% CI 4.03%–7.65%） |
| cofacts_scam | 棄權率 | 93.31%（586/628，95% CI 91.08%–95.01%） | 93.31%（586/628，95% CI 91.08%–95.01%） |
| cofacts_ham_ad | 誤判率（Wilson 上界 1.37% → 1.37%） | 0.24%（1/411，95% CI 0.04%–1.37%） | 0.24%（1/411，95% CI 0.04%–1.37%） |
| cofacts_ham_ad | 棄權率 | 97.08%（399/411，95% CI 94.97%–98.32%） | 97.08%（399/411，95% CI 94.97%–98.32%） |
| cofacts_ham_suspected | 誤判率（Wilson 上界 2.38% → 2.38%） | 0.93%（4/428，95% CI 0.36%–2.38%） | 0.93%（4/428，95% CI 0.36%–2.38%） |
| cofacts_ham_suspected | 棄權率 | 94.16%（403/428，95% CI 91.52%–96.01%） | 94.16%（403/428，95% CI 91.52%–96.01%） |

## tune 與 holdout 並列（校準後）

本 change 正是在 tune 上估權重的那一個，兩個切分的差距就是過擬合的量。

| 子集 | 指標 | tune | holdout |
|---|---|---|---|
| cofacts_scam | 召回率 | 5.24%（30/572，95% CI 3.70%–7.39%） | 5.57%（35/628，95% CI 4.03%–7.65%） |
| cofacts_ham_ad | 誤判率 | 0.26%（1/389，95% CI 0.05%–1.44%） | 0.24%（1/411，95% CI 0.04%–1.37%） |
| cofacts_ham_suspected | 誤判率 | 0.69%（3/435，95% CI 0.23%–2.01%） | 0.93%（4/428，95% CI 0.36%–2.38%） |

## ngram_classifier：不同級的來源

`ngram_classifier` 的訊號條目與 `ngram_threshold` 已在 `add-ngram-classifier` 中
量出，本 change 讀不重算。**它與其餘 34 個訊號不同級**：它的門檻是在同一份 tune 上
選出的，所以它的 `p_hit_given_scam`（84.62%）是一個選過的最大值，其餘訊號不是。
`weights.toml` 沒有欄位承載「這一條是選過的」，表面上兩者同形，記為已知弱點。
本 change 的評估 registry 未掛載分類器，其 holdout 命中率見 add-ngram-classifier 報告。

## 那則釣魚訊息

```
Chunghwa Post：包裹因關稅未繳而暫扣。請支付 159.71 元，付款請於此 ： https://e.vg/post-gov
```

| 指標 | 校準前 | 校準後 |
|---|---|---|
| scam_probability | None | None |
| confidence | 0.05 | 0.05 |
| scam_type | None | None |
| 命中訊號 | （無） | （無） |
| 依據行數 | 0 | 0 |

校準後這則訊息**仍然拒答**（`scam_probability = None`）—— 純規則模式下它命中不到
任何規則，分數為 0、信心 0.05 落在 `base_no_hit` 這一級。這不是失敗，是資料告訴
我們的事實：本系統的規則層對這一則英中夾雜、以短網址收尾的釣魚訊息沒有任何訊號。
接上分類器與 LLM 之後的判定要等那兩層在評估管線裡掛載才量得到，本 change 不為了
讓這一則過關而回頭調任何門檻。

