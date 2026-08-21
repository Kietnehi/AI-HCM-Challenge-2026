# 04 — TASK CHECKLIST

> Backlog thực thi. Mỗi task có **Definition of Done (DoD)** rõ ràng. Agent cập nhật checkbox khi xong.

## Ký hiệu

- 🔴 **Blocker** — task sau không làm được nếu chưa xong
- 👤 **Người làm** — cần phán đoán của người, agent không tự quyết
- ⏱️ Thời gian dự kiến

---

## SPRINT 0 — Verify giả định (làm TRƯỚC khi viết code) ⏱️ 1–2h

Bốn task này rẻ nhưng nếu bỏ qua thì sẽ đập đi làm lại cả notebook.

- [x] ~~**T0.1 — Verify `xiaomi/mimo-v2.5` trên OpenRouter**~~ ✅ **XONG** (qua `GET /api/v1/models` public, 2026-08-21)
  **Kết quả:** `xiaomi/mimo-v2.5` **có tồn tại**, `input_modalities = ["text","audio","image","video"]` → nhận vision. Giữ nguyên `LLM_MODEL`/`VLM_MODEL` ở `00_MASTER_PLAN.md §5`. **R1 đóng lại.**
  ⚠️ **Bẫy:** `xiaomi/mimo-v2.5-pro` là **text-only** (`["text"]`) — nghe "xịn hơn" nhưng đặt vào `VLM_MODEL` sẽ chết Stage 4. Đừng đổi.
  Fallback đã xác nhận tồn tại và nhận ảnh: `qwen/qwen3-vl-{8b,30b-a3b,32b,235b-a22b}-{instruct,thinking}` (7 bản), `google/gemini-2.5-flash`, `google/gemini-2.5-flash-lite`.
  *Còn lại:* Cell 2d của NB02 vẫn gửi 1 ảnh thật để chắc model không im lặng ignore ảnh — cần API key nên phải chạy trên Kaggle.

- [x] ~~**T0.2a — Verify path 3 dataset**~~ ✅ **XONG** (qua Kaggle API 2026-08-20)
  Kết quả ghi ở [`05_KAGGLE_PATHS.md`](./05_KAGGLE_PATHS.md): `feature-aic-2026` có thêm lớp `Feature_Dataset/`; mount dạng owner-qualified; `dethithunghiem` phẳng 24 file; ảnh keyframe theo `Keyframes_<batch>/keyframes/<vid>/<nnn>.jpg`.

- [ ] 🔴 **T0.2b — Verify `build_kf_index()` trả về đủ 873 video**
  Attach `fatle542/aic-dataset` (115.75 GB) vào 1 notebook trống, chạy `build_kf_index()` (`05_KAGGLE_PATHS.md §4`).
  **DoD:** `len(KF_DIR) == 873` + mở được 20 ảnh random bằng PIL. Riêng L26 (498 video) cần kiểm kỹ vì tên batch dir chưa biết (`Keyframes_L26` hay `Keyframes_L26_a…e`). Thiếu video nào thì in ra danh sách.

- [x] ~~**T0.3 — Quyết định embedding provider**~~ ✅ **CHỐT: `EMBED_PROVIDER="openrouter"`** (2026-08-21)
  **Phát hiện:** OpenRouter **có** endpoint `/api/v1/embeddings` (probe trả 401, không phải 404) → QĐ-1 đã lỗi thời, xem `00_MASTER_PLAN.md §4 QĐ-1` đã sửa.
  **DoD:** chỉ cần **1 key** `OPENROUTER_API_KEY` cho cả embedding + LLM + VLM. Không cần key OpenAI. NB01 để Accelerator **`None`** (CPU) — tiết kiệm quota GPU cho NB02.
  Vẫn giữ 2 mode dự phòng: `"openai"` (nếu có key riêng) và `"local"` + `bge-m3` (nếu smoke test `cos(en,vi) < 0.35`).

- [ ] **T0.4 — Quyết định reranker** (QĐ-3)
  **DoD:** chốt `RERANK_MODE`. Nếu `"local"` → xác nhận `Qwen/Qwen3-Reranker-4B` load được fp16 trên 1× T4 (chạy thử 1 cell, xem `nvidia-smi`).

---

## SPRINT 1 — NB01 Index Builder ⏱️ 1 ngày code + 4h chạy

Theo spec `02_NB01_INDEX_BUILDER.md`.

- [ ] **T1.1 — Cell 1–2: SETUP + PATH_VERIFY**
  **DoD:** 8 glob đếm đúng số kỳ vọng (873/873/873/873/866/873/873/700 — lưu ý 866 summary gồm 1 file sentinel `_failed.json`); `ocr_index.jsonl` = 128,664 dòng; `detected_classes.txt` parse ra đúng **584 class** sau khi bỏ 2 dòng `#`, 1 dòng trống và `.strip()` bỏ `\r`.

- [ ] 🔴 **T1.2 — Cell 3: `keyframes.parquet`**
  **DoD:** 177,321 dòng · `kf_id` unique · **invariant `len(group) == npy.shape[0]` pass cho cả 873 video** (nếu có video lệch: ghi list ra và dừng, đừng tự sửa).

- [ ] **T1.3 — Cell 4: `videos.parquet`**
  **DoD:** 873 dòng · **không có `video_id == "_failed"`** · `has_ocr.sum() == 700` · `has_summary.sum() == **865**` (8 video thiếu: `L26_V072`–`L26_V079`) · fallback `summary → description` hoạt động.

- [ ] 🔴 **T1.4 — Cell 5: `text_units.parquet`**
  **DoD:** **~319K dòng** · `value_counts("channel")` khớp ước lượng ±25% (caption ~163,000 · **ocr đúng 128,664** · asr ~21,500 · summary ~5,200 · meta 873) · `unit_id` unique · caption dup có `emb_row == -2` · ASR `frame_idx` không null · **ASR đã dedupe: không 2 unit nào cùng `(video_id, text_en)`** · spot-check 10 unit mỗi channel bằng mắt.

- [ ] **T1.5 — Cell 6: objects**
  **DoD:** `objects_matrix.shape == (177321, 584)` · `object_classes.txt` xuất ra đúng 584 dòng sạch (không `#`, không `\r`) · `float()` cast cho `detection_scores` (JSON lưu dạng chuỗi) · spot-check: kf có `Person` trong `classes` thì matrix cũng `True` tại cột `Person`.

- [ ] 🔴 **T1.6 — Cell 7: `vision.faiss`**
  **DoD:** `ntotal == 177321` · dtype fp32 · đã `normalize_L2` · **test: encode `"lion dance on poles"` bằng `clip-ViT-B-32`, top-10 phải là ảnh múa lân** (mở ảnh ra xem).

- [ ] **T1.7 — Cell 8: BM25 × 6**
  **DoD:** 6 index load được · `len(ids) == n_docs` mỗi cái · **test: `tok_vi("FANA")` trên `bm25_ocr_vi` trả về frame có chữ FANA** · ghi `tokenizer_vi` đã dùng vào manifest.

- [ ] 🔴 **T1.8 — Cell 9: `embed_texts()` + smoke test**
  **DoD:** shape đúng, finite · **`cos(en, vi) > 0.5`** (nếu < 0.35 → đổi sang `local`/`bge-m3` và ghi lại lý do) · retry 429 hoạt động (test bằng cách gửi burst).

- [ ] **T1.9 — Cell 10: PHASE_B embed toàn corpus** ⏱️ 1.5–3h
  **DoD:** checkpoint đầy đủ cho 5 channel · **chạy lại cell lần 2 phải skip 100% và xong trong < 1 phút** (chứng minh resumable) · log tổng token + USD.
  Thứ tự: `meta` → `summary` → `asr` → `ocr` → `caption`.

- [ ] 🔴 **T1.10 — Cell 11: `text_*.faiss` + patch dup**
  **DoD:** 5 index · `ntotal` khớp số unit embed · `ids` khớp từng phần tử với `text_units` · sau patch: **không còn `emb_row == -2`** nào.

- [ ] **T1.11 — Cell 12–13: MANIFEST + SELF_TEST**
  **DoD:** manifest đầy đủ (có `videos_missing_ocr` thật) · **8/8 assertion pass** · **3/3 truy vấn thử cho kết quả hợp lý khi xem bằng mắt**.

- [ ] 🔴 **T1.12 — Publish dataset `aic26-index`**
  **DoD:** dataset private tồn tại, < 20GB, attach thử vào 1 notebook mới và load được cả 3 parquet + 6 FAISS.

---

## SPRINT 2 — NB02 Pipeline ⏱️ 1–2 ngày code

Theo spec `03_NB02_PIPELINE_SUBMIT.md`.

- [ ] 🔴 **T2.1 — Cell 2: PREFLIGHT**
  **DoD:** manifest guard pass · `KEYFRAME_ROOT` tự dò được + 20/20 ảnh random mở được · `VLM_MODEL` xác nhận nhận vision.

- [ ] **T2.2 — Cell 3: LOAD_INDEX**
  **DoD:** load hết < 3 phút · RAM < 12GB · dict tra cứu `KF2FIDX` hoạt động.

- [ ] 🔴 **T2.3 — Cell 4–5: STAGE0 Query Understanding**
  Chạy trên **cả 24 query** ở `QUERY_ROOT`.
  **DoD:** `parsed_queries.json` đủ 24 entry · `type` khớp hậu tố file 24/24 · `n_events` (đếm **số dòng**) khớp `len(events)` cho mọi TRAKE — riêng `query-p1-18-trake` phải ra **4** dù các dòng là `E1/E2/E2/E4` · `object_classes` ⊆ 584 class · `visual_desc_en` ≤ 60 từ.
  👤 **Review tay:** `query-p1-15-qa` phải có `"FANA"` trong `ocr_hints`; `query-p1-19-qa` phải có `"Nguyễn Trung Trực"` trong `named_entities`. Không có → sửa prompt.

- [ ] 🔴 **T2.4 — Cell 6: RETRIEVERS (12 channel)**
  **DoD:** mỗi channel chạy độc lập được và trả về `(ids, scores)` đúng dài `TOPK_PER_CHANNEL` · ASR spread ra keyframe đúng khoảng thời gian · NFC-normalize query trước khi tokenize BM25.

- [ ] **T2.5 — Cell 7: FUSION**
  **DoD:** `candidates_<qid>.parquet` cho 24 query · cột `rank_*` đúng (`-1` khi channel không hit) · `fused_score` giảm dần · **kiểm: video thiếu OCR vẫn xuất hiện được trong top-100** (chứng minh R5 đã xử lý).

- [ ] **T2.6 — Cell 8: STAGE3 text rerank**
  **DoD:** evidence card đủ 6 mục · 1000 → 100 · không OOM · thời gian < 2 phút/query.

- [ ] **T2.7 — Cell 9: STAGE4 VLM rerank**
  **DoD:** ảnh resize ≤ 768 + base64 đúng · JSON parse robust (model có thể trả markdown → strip fence) · `vlm_reason` được lưu · dedupe kf cách nhau < 2s · <= 20 call/query.

- [ ] **T2.8 — Cell 10: KIS rows**
  **DoD:** 100 dòng/query · diversity cap hoạt động (đếm: không video nào > 5 frame trong 30 dòng đầu) · temporal padding có mặt.

- [ ] 🔴 **T2.9 — Cell 11: QA rows**
  **DoD:** answer ≤ 100 ký tự 100% · vote clustering hoạt động · **answer mặc định là verbatim tiếng Việt** (thể lệ mâu thuẫn giữa "ngữ nghĩa" và "chuỗi chính xác" — xem QĐ-5) · có dòng hedge tiếng Anh ở cuối file.
  👤 **Review tay bắt buộc:** `query-p1-19-qa` (2 câu thơ) phải là **tiếng Việt nguyên văn**, không dịch. `query-p1-15-qa` (tên xã) phải là tên xã tiếng Việt. Đây là R6 — rủi ro cao nhất của dạng QA.

- [ ] 🔴 **T2.10 — Cell 12: TRAKE + DP alignment**
  **DoD:** DP trả về `i_1 < ... < i_K` **tăng dần nghiêm ngặt** (unit test trên matrix giả) · `len(frames) == n_events` 100% · beam sinh ≥ 25 hàng khác nhau · cùng 1 `video_id` mỗi hàng.
  Test riêng cả **3** dạng cấu trúc: `query-p1-4-trake` (4 event, không mô tả cảnh chung) · `query-p1-16-trake` (1 dòng mô tả cảnh + 4 event) · `query-p1-18-trake` (**BTC đánh máy sai: `E1/E2/E2/E4`** — phải ra đúng 4 event theo thứ tự dòng, không dedupe theo con số).

- [ ] 🔴 **T2.11 — Cell 13–14: WRITER + VALIDATOR**
  **DoD:** **12/12 check pass** trên toàn bộ 24 file CSV. Đặc biệt check #7 (frame_idx tồn tại thật) và #10 (TRAKE ordering).

- [ ] **T2.12 — Cell 15: ZIP**
  **DoD:** `zipfile.namelist()` — mọi entry bắt đầu bằng `submission/`.

---

## SPRINT 3 — Tune & nộp ⏱️ liên tục

- [ ] 👤 **T3.1 — Label tay ground-truth partial**
  Chọn ≥ 10 query KIS, tìm `video_id` đúng bằng cách xem kết quả top-100. Ghi ra `Planning/gt_partial.csv` (`query_id, video_id, frame_idx_approx`).
  **DoD:** ≥ 10 query có video_id đúng. **Không có file này thì không thể tune có căn cứ** — mọi thay đổi weight chỉ là đoán.

- [ ] **T3.2 — Đo Recall@100 từng channel riêng lẻ**
  **DoD:** bảng `channel × recall@100` trên `gt_partial`. Channel nào recall < 5% → hạ weight về ~0 hoặc bỏ (tiết kiệm thời gian chạy).

- [ ] **T3.3 — Grid-search `CHANNEL_WEIGHTS`**
  Chỉ tune 6 tham số quan trọng: `vision`, `caption`, `ocr`, `bm25_ocr_vi`, `asr`, `summary_prior`. Grid thô `{0, 0.3, 0.6, 1.0}`.
  **DoD:** ghi weight tốt nhất + Recall@100 trước/sau vào `Planning/tuning_log.md`.

- [ ] **T3.4 — Nộp lần 1 (baseline an toàn)**
  **DoD:** validator 12/12 pass · zip đúng cấu trúc · ghi lại điểm Public LB.

- [ ] **T3.5 — Nộp lần 2 (sau tune + sửa tay `ocr_hints`)**
  **DoD:** điểm > lần 1. Nếu không hơn → **giữ lại lần 3**, đừng nộp bừa.

- [ ] 👤 **T3.6 — Nộp lần 3 (final, được tính điểm)**
  **DoD:** review tay toàn bộ answer QA + TRAKE ordering · chắc chắn tốt hơn lần 2 (không phải nhiễu Public LB — nhớ Public chỉ tính 50% đáp án).

---

## SPRINT 4 — Nâng cấp (chỉ làm SAU khi đã có baseline nộp được)

Thứ tự theo tỉ lệ lợi ích / công sức:

- [ ] **T4.1 — SigLIP2 re-embed** (QĐ-2). Lợi ích **cao nhất** cho KIS: `siglip2-giant > so400m > siglip2-base > clip-ViT-B-32`. Dùng notebook `aic2026-siglip-embedding` (`Link.txt`). Phải re-embed 177,321 ảnh **và** đổi text tower. Chạy notebook riêng, publish `aic26-index-siglip`, đổi `VISUAL_MODEL`/`VISUAL_DIM`. Chi phí: ~1 session GPU.
- [ ] **T4.2 — OCR cho 173 video còn thiếu** (L28–L30). Xóa được R5 hoàn toàn. Code sẵn có: `Code-Extract-Input/OCR/aic-ocr-26-easyocr-vietocr-kaggle.ipynb`.
- [ ] **T4.3 — Query expansion.** Sinh 3 biến thể `q_en` bằng LLM, retrieve cả 3, RRF hợp nhất. Rẻ, thường tăng recall vài %.
- [ ] **T4.4 — Temporal smoothing cho KIS.** Score của kf `i` cộng thêm `0.3 × max(score[i-1], score[i+1])`. Sự kiện thường trải trên nhiều keyframe liền nhau.
- [ ] **T4.5 — Qwen3-Reranker-8B AWQ 4-bit** thay 4B (nếu đo được là hơn thật).
- [ ] **T4.6 — Dataset keyframe nhẹ.** Resize 177K keyframe xuống 384px WEBP q80 (~4.4 GB thay vì 115.75 GB), publish `aic26-keyframes-small`. NB02 mount nhanh hơn nhiều mà VLM cũng chỉ cần ≤768px.

---

## Bảng theo dõi

| Sprint | Task | Trạng thái | Ghi chú |
|:--|:--|:--|:--|
| 0 | T0.1 | ✅ | `xiaomi/mimo-v2.5` có thật + nhận image. R1 đóng. Bản `-pro` là text-only, đừng dùng |
| 0 | T0.2a | ✅ | path 3 dataset đã verify qua Kaggle API |
| 0 | T0.3 | ✅ | Chốt `EMBED_PROVIDER="openrouter"` — OpenRouter CÓ /embeddings, chỉ cần 1 key |
| 0 | T0.2b, T0.4 | ⬜ | cần chạy trên Kaggle (mount 115GB / thử load reranker trên T4) |
| 1 | T1.1–T1.12 | ⬜ | |
| 2 | T2.1–T2.12 | ⬜ | |
| 3 | T3.1–T3.6 | ⬜ | |
| 4 | T4.1–T4.5 | ⬜ | chỉ sau khi có baseline |
