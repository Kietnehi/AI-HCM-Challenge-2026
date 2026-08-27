# Hướng dẫn chạy Pipeline API trên Kaggle GPU — AIC 2026

Áp dụng cho 3 notebook trong thư mục này. Đọc các mục 0 → 7 trước khi bấm Run.

---

## 0. Chuẩn bị một lần

### 0.1. Lấy API key OpenRouter

1. Đăng nhập https://openrouter.ai → **Keys** → **Create Key**.
2. Nạp credit tối thiểu khoảng `$3` (ngân sách thiết kế là `$2`, nên chừa dư để không bị chặn giữa lúc chạy).
3. Sao chép key dạng `sk-or-v1-...`. **Không dán key trực tiếp vào notebook.**

### 0.2. Khai báo key làm Kaggle Secret

Trong notebook Kaggle, chọn **Add-ons → Secrets → Add a new secret**:

| Trường | Giá trị |
|---|---|
| Label | `OPENROUTER_API_KEY` |
| Value | key vừa tạo |

Bật **Attached** cho notebook đang mở. Notebook đọc key theo thứ tự: biến môi trường
`OPENROUTER_API_KEY` → Kaggle Secret cùng tên. Nếu không có key, pipeline vẫn chạy được ở chế độ `DRY_RUN`.

### 0.3. Attach dataset

Chọn **Add Data** và attach cả 3 dataset:

| Dataset | Mount path dự kiến | Dùng cho |
|---|---|---|
| `fatle542/AIC-Dataset` | `/kaggle/input/AIC-Dataset` | video gốc và keyframe |
| `kitnehi1211/feature-AIC-2026` | `/kaggle/input/feature-AIC-2026` | SigLIP2, caption, OCR, transcript, summary, object |
| `kitnehi1211/dethithunghiem` | `/kaggle/input/dethithunghiem` | bộ query `query-*.txt` hoặc `Query-*.TXT` |

Notebook tự dò các thư mục này. Nếu dò sai, override trong CELL 1:

```python
CFG["feature_root"] = "/kaggle/input/feature-AIC-2026"
CFG["dataset_root"] = "/kaggle/input/AIC-Dataset"
CFG["query_dir"] = "/kaggle/input/dethithunghiem"   # NB02
```

### 0.4. Bật GPU và Internet

Trong **Settings**:

- **Accelerator**: `GPU T4 x2`.
- **Internet**: `On` (cần tải SigLIP2 và gọi OpenRouter).
- **Persistence**: `Files only` để giữ `/kaggle/working` giữa các lần chạy interactive.

NB01 hầu như không cần GPU; NB02 cần GPU cho SigLIP2; NB03 không cần GPU.
Mỗi session Kaggle tối đa 12 giờ. Pipeline có checkpoint theo stage/query nên có thể chạy tiếp sau timeout.

---

## 1. Cài thư viện

Thêm một cell ở đầu NB01 và NB02:

```python
!pip install -q faiss-cpu
!pip install -q -U transformers
```

Kaggle image thường đã có `torch`, `pandas`, `pyarrow`, `scipy`, `opencv-python`, `Pillow`, `ipywidgets`.

---

## 2. Notebook 1 — Build Indices

Tạo notebook mới và upload `01-build-indices-api.ipynb` (**File → Import Notebook**).

### 2.1. Chạy thử miễn phí

Giữ `CFG["DRY_RUN"] = True` và chọn **Run All**. Cần kiểm tra:

| Cell | Kết quả mong đợi |
|---|---|
| 3 | các modality được tìm thấy; `videos found` và `keyframe dirs` lớn hơn 0 |
| 5 | kích thước SigLIP2, L2-norm và coverage hợp lệ |
| 8 | smoke self-search trả về row 0 |
| 12 | bảng **COST ESTIMATE** |
| 14 | `P0 còn lại: 0` và `NB01 READY` |

Nên chạy smoke test trước:

```python
CFG["limit_videos"] = 20
```

Sau khi thấy `SMOKE: OK`, đặt lại `limit_videos = 0`, xóa checkpoint rồi chạy toàn bộ dataset:

```python
!rm -rf /kaggle/working/artifacts
```

### 2.2. Bật API cho text embedding

Chỉ sau khi xem COST ESTIMATE và chấp nhận chi phí, đặt:

```python
CFG["DRY_RUN"] = False
```

Sau đó chạy lại CELL 12 → CELL 13 → CELL 14. Có thể giảm chi phí bằng cách tắt một số modality:

```python
CFG["embed_targets"] = {"caption": True, "transcript_en": True, "summary": False}
```

Lưu phiên bản bằng **Save Version → Save & Run All (Commit)** để `/kaggle/working/artifacts` trở thành output cho NB02.

---

## 3. Chuyển artifact từ NB01 sang NB02

### Cách A — Attach Notebook Output (khuyến nghị)

Trong NB02 chọn **Add Data → Notebook Output**, chọn phiên bản đã commit của NB01, rồi đặt trong CELL 1:

```python
CFG["art_dir"] = "/kaggle/input/<slug-notebook-01>/artifacts"
```

Mount tại `/kaggle/input` là read-only. Notebook sẽ ghi ledger và error log vào `review_package/` khi cần.

### Cách B — Dùng chung một session

Giữ `CFG["art_dir"] = "/kaggle/working/artifacts"` cho cả 3 notebook và bật **Persistence: Files only**.

---

## 4. Notebook 2 — Retrieve / Refine

### 4.1. Chạy thử một query

Trong CELL 1:

```python
CFG["only_queries"] = ["Query-p1-1-KIS"]
CFG["DRY_RUN"] = True
```

Kiểm tra visual index, số query, SigLIP2 trên CUDA, kích thước các nhánh và `SMOKE: OK`.

### 4.2. Chạy toàn bộ

```python
CFG["only_queries"] = []
CFG["DRY_RUN"] = False
CFG["allow_vlm_verify"] = False
CFG["refine_enabled"] = True
```

Mặc định VLM không được gọi. Nếu reviewer cần MiMo hỗ trợ một query khó, bật `allow_vlm_verify` và dùng
`request_vlm(qid)` trong NB03, sau đó chạy lại đúng query đó với `force_requery = True`.

Các tham số chính:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `vlm_topm_requested` | 10 | số candidate gửi cho VLM khi được yêu cầu |
| `rerank_topn` | 60 | số candidate được rerank |
| `refine_topk` | 12 | số candidate được decode/refine |
| `refine_max_frames_per_cand` | 90 | giới hạn frame decode mỗi candidate |
| `nb02_budget_usd` | 1.20 | trần chi phí riêng NB02 |

CELL 17 (**EXPORT SUBMISSION**) xuất gói nộp ngay trong NB02:
`/kaggle/working/submission_build/submission/<query_id>.csv` + `/kaggle/working/submission.zip`.
Cell in ra bảng kiểm format (số cột theo KIS/QA/TRAKE, ≤100 dòng, frame_idx là số nguyên, answer ≤100 ký tự)
và liệt kê query không xuất được file. Tải zip ở tab **Output** rồi nộp thẳng lên hệ thống BTC.

Q&A: mặc định `allow_vlm_verify=False` nên `answer` rỗng → tạo `/kaggle/working/qa_answers.json`
(`{"query-p1-15-qa": "12"}`) rồi chạy lại **riêng CELL 17**, không phải chạy lại pipeline.

Lưu output bằng **Save & Run All (Commit)** để NB03 dùng `review_package`.

---

## 5. Notebook 3 — Human Review và Submit

Attach output của NB02, sau đó đặt:

```python
CFG["art_dir"] = "/kaggle/input/<slug-notebook-01>/artifacts"
CFG["pkg_dir"] = "/kaggle/input/<slug-notebook-02>/review_package"
# decision_file mặc định ở /kaggle/working/review_decisions.json
```

Chạy interactive, **không Commit** trong lúc review.

NB03 cung cấp widget và text mode để xem contact sheet, sửa frame/answer, promote/reject candidate,
đọc bằng chứng OCR/ASR/caption và đánh dấu query hoàn tất. Ví dụ:

```python
review_text("Query-p1-16-TRAKE")
set_event_frame("Query-p1-16-TRAKE", 1, 2, 3450)
promote_seq("Query-p1-16-TRAKE", 3, 1)
set_answer("Query-p1-15-qa", 1, "Vạn Ninh")
print_evidence("L21_V001", pts_time=124.5)
mark_done("Query-p1-15-qa")
```

Sau khi review, chạy CELL 8 → 9 → 10 để sinh CSV, validate và tạo ZIP. Nếu còn lỗi P0, sửa review rồi chạy lại.
Sau mỗi lần sửa chỉ cần gọi `Regenerate()`; thao tác này không gọi API và không build lại index.

---

## 6. Sự cố thường gặp

| Hiện tượng | Xử lý |
|---|---|
| Không tìm thấy SigLIP2 | kiểm tra Internet và `CFG["feature_root"]` |
| Row alignment sai | xóa `/kaggle/working/artifacts` và chạy lại NB01 |
| SigLIP2 không load được | bật Internet hoặc trỏ `siglip_local_dir` tới model offline |
| CUDA OOM | giảm `CFG["siglip_batch"]` và `refine_max_frames_per_cand` |
| Chạm trần chi phí | xem `cost_ledger.json`, giảm VLM/rerank hoặc tăng cap nếu chấp nhận |
| HTTP 401 | kiểm tra API key và Kaggle Secret |
| Không đọc được video | attach `AIC-Dataset`; refinement sẽ tắt và dùng frame của keyframe |
| Q&A có answer rỗng | tạo `/kaggle/working/qa_answers.json` rồi chạy lại CELL 17 của NB02 |
| Output không có file nộp | chạy **CELL 17** của NB02 → `/kaggle/working/submission.zip` |
| Lỗi read-only filesystem | để `decision_file` và `sub_dir` trong `/kaggle/working` |

---

## 7. Thứ tự tối thiểu để có bài nộp

```text
NB01: DRY_RUN=True, limit_videos=20  → kiểm tra SMOKE OK
NB01: limit_videos=0, xóa artifacts  → build đầy đủ
NB01: DRY_RUN=False, chạy CELL 12-14 → Save & Run All (Commit)
NB02: only_queries=[1 query], DRY_RUN=True  → kiểm tra degraded/SMOKE
NB02: only_queries=[], DRY_RUN=False        → Save & Run All (Commit)
NB03: interactive, review P0→P3             → CELL 8-9-10 → submission.zip
```
