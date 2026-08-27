# Hướng dẫn chạy pipeline LOCAL trên Kaggle GPU (2×T4)

Áp dụng cho 3 notebook trong thư mục này. Đọc kèm `README.md` (tóm tắt thiết kế).

---

## 0. Chuẩn bị một lần

### 0.1. Dataset cần attach

| Dataset | Dùng ở NB | Bắt buộc |
|---|---|---|
| `fatle542/aic-dataset` (Videos_*, Keyframes_*) | 01 (audit), 02 (**frame refinement**), 03 (mở rộng cửa sổ frame) | 02: rất nên có — thiếu thì frame không được refine |
| `kitnehi1211/feature-aic-2026` | 01 | ✅ |
| `kitnehi1211/dethithunghiem` (bộ `query-*.txt`) | 02 | ✅ |
| output NB01 (`artifacts/`) | 02 | ✅ |
| output NB02 (`artifacts02_mimo/review_package/`) | 03 | ✅ |

Attach bằng **Add Input → Datasets** (hoặc **Notebook Output** cho artifact của NB01/NB02).

### 0.2. Cài đặt Session (menu bên phải)

| Notebook | Accelerator | Internet | Persistence |
|---|---|---|---|
| 01 | **GPU T4 ×2** | On (tải `BAAI/bge-m3`) | Files only |
| 02 | **GPU T4 ×2** | On (tải SigLIP2/reranker và gọi MiMo API) | Files only |
| 03 | **None (CPU)** | Off | Files only |

> Chỉ NB01 và NB02 cần GPU. NB03 chạy CPU cho nhanh và khỏi tốn quota GPU.
> Quota GPU Kaggle ~30 h/tuần — đừng để NB03 chiếm.

### 0.3. Nếu tài khoản không được bật Internet

Tạo trước một dataset chứa cache model (tải ở máy khác rồi upload), cấu trúc:

```
model-cache/
├── bge-m3/                     (config.json, model.safetensors, tokenizer*)
├── siglip2-giant-opt-patch16-384/
├── bge-reranker-v2-m3/
```

Rồi đặt trong CFG của NB01/NB02: `CFG["LOCAL_MODEL_DIR"] = "/kaggle/input/model-cache"`.

---

## 1. NB01 — `01_build_indices_local.ipynb`

### 1.1. Smoke run trước (~10 phút, bắt buộc)

Cell 2 (CFG), sửa 1 dòng:

```python
LIMIT_VIDEOS   = 20,
```

Run All. Kiểm tra output:

- Cell 3: `FEATURE_ROOT` / `DATASET_ROOT` phải trỏ đúng `/kaggle/input/...`, không có modality nào `None` (trừ `clip` — không dùng).
- Cell 5: coverage — `siglip / mapkf / caption / summary` nên ~100%; `ocr` có thể thấp; `video` = 0% nếu chưa attach `aic-dataset` (chấp nhận được ở NB01).
- Cell 6: `P0 mismatch ... NONE OK` → SigLIP2 đúng 1536-d và số hàng khớp `map-keyframes`.
- Cell 10: `smoke BGE` — `cos(vi,en)` phải **lớn hơn** `cos(vi,unrelated)`; nếu assert fail là FP16 lỗi → đặt `BGE_FP16 = False`.
- Cell 14: bảng integrity **ALL PASS**.

### 1.2. Full run

Đặt lại `LIMIT_VIDEOS = None` → **Save Version → Save & Run All (Commit)**.
Đừng chạy full trong chế độ interactive: session interactive bị ngắt khi đóng tab, còn Commit chạy nền tới 12 h.

**Runtime tham chiếu** (873 video, ~250k keyframe, T4):

| Stage | Thời gian |
|---|---|
| records (cell 7–8) | 10–20 phút |
| FAISS SigLIP2 (cell 9) | 5–10 phút |
| BGE-M3 caption ~250k doc (cell 11) | 45–80 phút |
| BGE-M3 transcript + summary | 10–15 phút |
| BM25 ×4 + object index (cell 12–13) | 5–15 phút |
| **Tổng** | **1.5–2.5 h** |

Nếu chạm `WALL_BUDGET_H = 10.5`, notebook raise `TimeoutError` có kiểm soát → **Save Version**, rồi tạo version mới attach chính output đó: shard embedding và stage đã xong sẽ được **skip theo fingerprint**, chỉ chạy phần còn thiếu.

### 1.3. Sau khi xong

Ghi lại slug output (dạng `<user>/<notebook-slug>`) — NB02 cần nó.
Artifact nằm ở `/kaggle/working/artifacts` → khi attach vào NB02 nó thành
`/kaggle/input/<slug>/artifacts`.

---

## 2. NB02 — `02_retrieve_refine_candidates_local.ipynb`

### 2.1. Cấu hình

Cell 2:

```python
ART_INPUT    = "/kaggle/input/<slug-nb01>/artifacts",
ART_OUT      = "/kaggle/working/artifacts02_mimo",
LIMIT_QUERIES = 3,        # smoke trước; sau đó None
```

Bạn có thể dùng nguyên output cũ của NB01; không cần chạy lại NB01 nếu `manifest.json` có `integrity_all_pass: true`. Chỉ attach output NB01 cũ làm input và để NB02 ghi sang `artifacts02_mimo`.

Trong Kaggle, vào **Add-ons → Secrets**, tạo secret tên `OPENROUTER_API_KEY` chứa API key OpenRouter. Bật Internet cho notebook. NB02 dùng MiMo chỉ cho text query planning; visual verification và answer để human review xử lý:

```python
MIMO_MODEL      = "xiaomi/mimo-v2.5",
USE_MIMO_PARSE  = True,
MIMO_DRY_RUN    = False,
REQUIRE_MIMO_PARSE = True,
USE_QWEN_PARSE  = False,
USE_QWEN_VERIFY = False,
USE_QWEN_ANSWER = False,
```

Chạy cell 6–7 và kiểm tra log:

- `MiMo parser ready` và `MiMo API key: True` → cấu hình API hợp lệ.
- `parser=mimo` và `q_en` không rỗng → MiMo parse/dịch đang hoạt động.
- `parser=rule` hoặc `q_en` rỗng → API/schema chưa hợp lệ; với `REQUIRE_MIMO_PARSE=True`, notebook sẽ dừng trước retrieval.
- `USE_QWEN_VERIFY=False` nghĩa là không dùng Qwen để nhìn ảnh/xác minh candidate.
- `USE_QWEN_ANSWER=False` nghĩa là answer Q&A sẽ được điền trong bước human review.

MiMo là API text-only nên không gây CUDA OOM. Notebook cache từng response trong `artifacts02_mimo/mimo_cache`; chạy lại trong cùng output sẽ không gọi lại những request đã thành công.

### 2.2. Kiểm tra ở smoke run

- Cell 3: `videos in siglip rowmap` khớp NB01; `contiguous` gần bằng tổng số video (TRAKE dùng đường nhanh này).
- Cell 4: `loaded N queries` với phân bố `{kis, qa, trake}` đúng; `video files` > 0 nếu muốn refinement.
- Cell 5: `smoke SigLIP2 text->visual` trả về keyframe có nghĩa (score ~0.05–0.3 là bình thường với SigLIP).
- Cell 7: `parser distribution` — full run cần thấy `{'mimo': N}` và `empty q_en: 0`.
- Cell 13 (orchestrator): mỗi query in `rows=… runtime=…`.

### 2.3. Full run

`LIMIT_QUERIES = None` → **Save & Run All (Commit)**.

**Runtime tham chiếu / query**: MiMo parse phụ thuộc độ trễ API; retrieval + fusion 5–20 s · rerank 5–15 s · refinement 12 candidate ×(±3 s decode) 60–180 s. Bản mặc định không chạy Qwen visual verify/Q&A.
→ nhanh hơn bản full-VLM; TRAKE vẫn nặng hơn do alignment/refinement.

Mặc định mới dùng `REFINE_TOPR=20` và render `SHEET_TOP=12`. Nếu cần smoke nhanh, có thể tạm giảm `REFINE_TOPR=6`, nhưng full run nên trả lại 20.

Timeout an toàn: mỗi query đã xong được ghi `ckpt/q_<qid>.json`; chạy version mới với artifact cũ attach thì các query đó bị **SKIP**.

Trong `review_package`, mở `frame_catalog.csv` để tra chính xác ảnh của từng query/rank/event. Ảnh decode thật được lưu theo dạng `frames/L30_V068/004179.jpg`; cột `source_keyframe_image` ghi keyframe gốc dạng `L30_V068/163.jpg`. Hai số này khác nhau vì `4179` là frame video sau refinement, còn `163` là thứ tự file keyframe dùng để truy xuất ban đầu.

### 2.4. VRAM thực tế (1 model tại một thời điểm)

| Model | dtype | VRAM |
|---|---|---|
| BGE-M3 (NB01) | FP16 | ~1.5 GB |
| SigLIP2 giant-opt | FP16 | ~4–5 GB (kèm activation ảnh) |
| BGE-reranker-v2-m3 | FP16 | ~1.5 GB (nạp lên GPU 1) |

MiMo chạy qua API và không dùng VRAM. Pipeline `unload` sau mỗi GPU stage. Nếu vẫn OOM: `REFINE_BATCH = 8`, `RERANK_TOPN = 40`.

---

## 3. NB03 — `03_human_review_submit_local.ipynb`

### 3.1. Cấu hình

```python
PKG_INPUT = "/kaggle/input/<slug-nb02>/artifacts02_mimo/review_package",
ZIP_NAME  = "team_<tenteam>_round1.zip",
```

Attach thêm `aic-dataset` nếu muốn bấm **⤢ mở rộng** để xem cửa sổ ±3/6/12 s quanh frame.

### 3.2. Quy trình review

1. Chạy cell 1 → 6 (không chạy cell 7 nếu chỉ muốn regenerate).
2. Cell 3 in **thứ tự review đề xuất**: TRAKE → Q&A cần đếm/OCR → confidence thấp → KIS.
3. Cell 7: `review()` mở query ưu tiên nhất, hoặc `review("query-p1-16-trake")` mở đúng query.
   - KIS: sửa `frame:` nếu cần → **✔ chọn** (thêm vào danh sách xác nhận) hoặc **⬆ rank1** (đưa lên hạng 1) → **✖ loại** với candidate sai.
   - Q&A: sửa cả `frame` và `answer` (≤100 ký tự) trước khi chọn.
   - TRAKE: sửa từng `E1..EN`, bấm **kiểm tra thứ tự** để xác nhận đúng số event + tăng dần → **✔ chọn sequence**.
   - Bấm **✅ query DONE** rồi chạy lại cell 7 để sang query kế tiếp.
   - Mọi thao tác ghi ngay vào `review_decisions.json` → đóng notebook không mất.
4. Cell 8: bảng tổng hợp, kiểm tra `top1_source = human` ở các query đã review.
5. Cell 9: validator. Chỉ khi **không còn FAIL (P0)** mới sang bước tiếp.
6. Cell 10: tạo ZIP và tự verify lại nội dung file zip.

### 3.3. Regenerate submission (rất nhanh, không cần model)

Sửa quyết định review → chạy lại **cell 8 → 9 → 10**. Index và model output không bị build lại.

Muốn giữ decision qua nhiều session: download `review_decisions.json`, hoặc upload nó thành dataset và copy vào `OUT_DIR` ở đầu session sau.

### 3.4. Nộp bài

Download `/kaggle/working/submit/team_<tenteam>_round1.zip` → nộp trên hệ thống BTC.
Nhớ: tối đa **3 lần nộp/gói**, lần cuối được tính điểm; nộp sai format vẫn tính 1 lần.

---

## 4. Sự cố thường gặp

| Hiện tượng | Nguyên nhân / xử lý |
|---|---|
| `Không tìm thấy artifact NB01` | chưa attach output NB01, hoặc set `CFG["ART_INPUT"]` sai. Kiểm tra `!ls /kaggle/input` |
| Coverage `video = 0%` ở NB02 | chưa attach `aic-dataset` → refinement bị bỏ qua, frame_idx giữ theo keyframe |
| `Thiếu Kaggle Secret OPENROUTER_API_KEY` | Tạo secret đúng tên, bật quyền truy cập secret cho notebook rồi chạy lại từ đầu. |
| `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'float'` | transformers 5.x đổi return type của `get_text_features`. Đã fix trong notebook bằng `as_tensor()` — đảm bảo bạn dùng bản notebook mới nhất |
| Cell 3/4 của NB02 treo 15–25 phút | do `glob(**, recursive=True)` đi bộ toàn bộ `/kaggle/input`. Đã thay bằng `fast_glob()` quét từng tầng → còn vài giây. Dùng bản notebook mới nhất |
| `OpenRouter/MiMo lỗi` | Kiểm tra Internet, API key, credit/rate limit. Response thành công đã được cache nên có thể chạy lại an toàn. |
| `torch_dtype is deprecated` | chỉ là warning; notebook đã dùng `load_hf()` thử `dtype=` trước rồi mới `torch_dtype=` |
| `CUDA out of memory` | pipeline tự giảm batch; nếu vẫn lỗi: `REFINE_BATCH=8`, `BGE_BATCH=8`, `RERANK_TOPN=40` |
| BM25 pickle load lỗi | NB02 đã định nghĩa lại `BM25MultiField` + `bm25_load` ở cell 3 — phải chạy cell 3 trước mọi cell retrieval |
| ipywidgets không hiện gì ở NB03 | chạy trong Kaggle Notebook editor (interactive), không phải xem log của Commit. Widget không render trong output của bản Commit |
| Query bị `error` trong `candidates_manifest.json` | xoá `ckpt/q_<qid>.json` tương ứng rồi chạy lại cell 13 của NB02 |
| Notebook dừng vì `TimeoutError ... budget` | đúng thiết kế: Save Version, tạo version mới attach output cũ → resume theo fingerprint/checkpoint |

---

## 5. Checklist trước khi nộp

- [ ] NB01 integrity report **ALL PASS**, `manifest.json` có `integrity_all_pass: true`
- [ ] NB02 `candidates_manifest.json`: mọi query có `rows > 0`, không có `error`
- [ ] Mỗi query TRAKE đã được người xem lại từng event (điểm chia theo tỉ lệ frame khớp)
- [ ] Mọi query Q&A có answer không rỗng, ≤100 ký tự
- [ ] NB03 `validation_report.csv`: 0 FAIL
- [ ] Mở thử ZIP: có thư mục `submission/`, mỗi CSV ≤100 dòng, `video_id` không đuôi `.mp4`
