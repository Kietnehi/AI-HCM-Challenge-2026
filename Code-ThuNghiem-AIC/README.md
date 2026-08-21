# Code-ThuNghiem-AIC — Implementation của `Planning/`

Hai notebook Kaggle triển khai đúng spec trong [`../Planning/`](../Planning/).

## Nội dung

| File | Mô tả |
|:--|:--|
| `aic26-01-index-builder.ipynb` | **NB01** — corpus → embedding → FAISS + BM25 → publish dataset `aic26-index` |
| `aic26-02-pipeline-submit.ipynb` | **NB02** — query → retrieval → fusion → rerank → CSV/zip |
| `src/nb01_index_builder.py` | Source NB01 dạng percent-format (**sửa ở đây**, dễ diff hơn `.ipynb`) |
| `src/nb02_pipeline_submit.py` | Source NB02 |
| `build_notebooks.py` | `src/*.py` → `.ipynb` |
| `tests/test_logic.py` | 74 check các hàm logic thuần, chạy trên dữ liệu local thật |
| `tests/smoke_nb01.py` | Exec **đúng source** Phase A của NB01 trên `../Feature_Dataset` |
| `tests/smoke_nb02.py` | Exec sinh-row + WRITER + VALIDATOR của NB02 trên index tổng hợp |

## Quy trình sửa code

```bash
# 1. sửa src/*.py   (KHÔNG sửa .ipynb trực tiếp — sẽ bị ghi đè)
# 2. sinh lại notebook
python build_notebooks.py
# 3. chạy test
python tests/test_logic.py && python tests/smoke_nb01.py && python tests/smoke_nb02.py
```

Tất cả test chạy **offline**: không cần Kaggle, API key, hay GPU.
`smoke_nb01.py` cần `../Feature_Dataset` (tự skip nếu không có);
`smoke_nb02.py` dựng index tổng hợp 12 video nên chạy được ở mọi máy.

## Chạy trên Kaggle

### NB01 — `aic26-01-index-builder`
- **Attach 1 dataset:** `kitnehi1211/feature-aic-2026` (KHÔNG attach `fatle542/aic-dataset`)
- Accelerator `None` nếu `EMBED_PROVIDER="openai"`, `GPU T4 x2` nếu `"local"` · **Internet ON**
- Điền `OPENAI_API_KEY` ở cell CONFIG · chạy ~3–4h
- Publish `/kaggle/working/index/` → Kaggle Dataset **`aic26-index`** (private)

### NB02 — `aic26-02-pipeline-submit`
- **Attach 4 dataset:** `aic26-index` · `feature-aic-2026` · `dethithunghiem` · `fatle542/aic-dataset`
- `GPU T4 x2` · **Internet ON** · điền `OPENROUTER_API_KEY` + `OPENAI_API_KEY`
- **Đổi `TEAM_NAME`** khỏi giá trị mặc định `team_XXX`
- Chạy ~1–3h → `/kaggle/working/<TEAM_NAME>_round1.zip`

## Hai điểm dừng bắt buộc của con người 👤

1. **Cell 5 NB02 (`STAGE0_REVIEW`)** — đọc bảng `ocr_hints`. Sai/thiếu `ocr_hints` là nguyên
   nhân miss lớn nhất với news video. Sửa `/kaggle/working/parsed_queries.json` rồi đặt
   `RELOAD_PARSED = True` và chạy lại Cell 4c.
2. **Cell 12d NB02** — đọc answer QA. Câu thơ / tên xã / tiêu đề món ăn **phải là tiếng Việt
   nguyên văn**, không dịch (QĐ-5, R6).

## Số liệu đo được trên `Feature_Dataset` thật

Khác với ước lượng ở `00_MASTER_PLAN.md §6` ở hai chỗ — **cả hai đều không phải bug**,
notebook đã dùng số đã hiệu chỉnh:

| Channel | Plan ước | Đo thật | Ghi chú |
|:--|--:|--:|:--|
| caption | ~163,000 | **177,321** unit / **145,505** cần embed | Plan đã trừ dup, nhưng theo data contracts §3.2 dòng dup **vẫn tạo unit**. Tỉ lệ dup thật **17.9%** (không phải 8.2%) |
| ocr | 128,664 | **128,300** | bỏ 364 chuỗi < 2 ký tự |
| asr | ~21,500 | **23,200** | sau 2 lớp dedupe |
| summary | ~5,200 | **1,355** | `chunk_summaries` thực tế chỉ ~0.6 chunk/video |
| meta | 873 | **873** | |
| **Tổng** | ~319,000 | **331,049** | ~299,233 unit cần embed |

Segment ASR dài trung bình **16.2s** (plan ghi 24.2s), nhưng tỉ lệ dư do cửa sổ trùng vẫn
**1.60×** — đúng như plan dự đoán, nên bước dedupe vẫn bắt buộc.

## Hai bug đã bị test bắt trước khi lên Kaggle

1. **ASR dedupe 1 lớp là không đủ** — hai tập segment *khác nhau* vẫn sinh text *giống hệt*
   (khi segment thêm vào có text rỗng). Đã thêm lớp dedupe thứ hai theo `text_embed`.
2. **KIS diversity cap deadlock** — `cap = 5 if len(rows) < 30 else 10` kẹt cứng khi mọi video
   đã chạm 5 mà `rows` vẫn < 30; vòng bù-quota (không cap) sau đó chèn dòng vào *trong* 30 dòng
   đầu và phá vỡ chính ràng buộc diversity. Đã đổi sang duyệt theo tier `(30,5) → (60,10) → (100,None)`.
