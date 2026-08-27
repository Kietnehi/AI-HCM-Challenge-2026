# AIC 2026 — Pipeline LOCAL (Kaggle 2×T4), 3 notebook

> Hướng dẫn chạy chi tiết trên Kaggle GPU: [KAGGLE_RUN_GUIDE.md](KAGGLE_RUN_GUIDE.md)

Triển khai theo `Kiet-Prompt/Prompt_Plan_Local.md`. Toàn bộ code nằm trong notebook (không có file `.py`).

| # | Notebook | Vai trò | Artifact ra |
|---|----------|---------|-------------|
| 1 | `01_build_indices_local.ipynb` | discovery + audit schema/coverage, canonical records, build FAISS (SigLIP2 1536-d, BGE-M3 1024-d ×3), BM25 multi-field ×4, object inverted index, integrity check | `/kaggle/working/artifacts/` (+ `manifest.json`) |
| 2 | `02_retrieve_refine_candidates_local.ipynb` | MiMo API parse/dịch/mở rộng query bằng text, retrieval song ngữ đa nhánh, weighted RRF + video prior, temporal NMS + diversity, BGE-reranker, joint-score frame refinement trên video gốc, human review tự xử lý visual/Q&A, TRAKE alignment theo độ phủ event | `artifacts02_mimo/review_package/` (`candidates.parquet`, `frame_catalog.csv`, `frames/`, `queries_parsed.json`, `sheets/`) |
| 3 | `03_human_review_submit_local.ipynb` | review UI (ipywidgets) cho KIS/Q&A/TRAKE, decision store resume được, validator P0/P1, `submission.zip` | `/kaggle/working/submit/` |

## Chạy trên Kaggle

1. **NB01** — attach dataset `aic-dataset`, `feature-aic-2026`, `dethithunghiem`. Bật GPU (BGE-M3 encode).
   Notebook tự dò root trong `/kaggle/input`; override bằng `CFG["FEATURE_ROOT"]`… nếu cần.
   Smoke trước: đặt `CFG["LIMIT_VIDEOS"] = 20`, chạy hết, xem `integrity_report.csv` → rồi bỏ limit và Save & Version.
2. **NB02** — attach output NB01 + `aic-dataset` (cần video gốc cho refinement) + bộ đề.
   Đặt `CFG["ART_INPUT"] = "/kaggle/input/<slug-nb01>/artifacts"`. Bật GPU + Internet (tải model).
   Thêm Kaggle Secret `OPENROUTER_API_KEY`; NB02 mặc định dùng `xiaomi/mimo-v2.5` để parse/dịch/mở rộng query bằng text. `USE_QWEN_VERIFY=False` và `USE_QWEN_ANSWER=False`, nên keyframe/video và Q&A vẫn do human review xử lý.
   Nếu MiMo API lỗi hoặc `q_en` rỗng, notebook sẽ dừng ở cell parse (`REQUIRE_MIMO_PARSE=True`) để tránh tạo package chất lượng thấp.
3. **NB03** — attach output NB02 (+ `aic-dataset` nếu muốn mở rộng cửa sổ frame). **Không cần GPU.**
   `CFG["PKG_INPUT"] = "/kaggle/input/<slug-nb02>/artifacts02_mimo/review_package"`.
   Chạy cell 1→7, review từng query (`review()` lấy query ưu tiên nhất), rồi chạy cell 8→10 để xuất ZIP.
   Regenerate submission: chỉ cần chạy lại cell 8→9→10.

## Ghi chú thiết kế quan trọng

- **Không trộn embedding space**: SigLIP2 (1536), BGE-M3 (1024), CLIP (512) mỗi loại một FAISS index riêng; index persist ở CPU.
- **GPU lifecycle**: mỗi stage GPU load một model rồi `unload`; MiMo chạy qua API nên không chiếm VRAM. Batch tự giảm khi OOM.
- **Checkpoint/resume**: NB01 skip theo fingerprint thư mục + shard embedding; NB02 checkpoint từng query (`ckpt/q_<qid>.json`); NB03 lưu `review_decisions.json`.
- **Frame thật**: keyframe chỉ khoanh vùng; frame_idx cuối lấy từ decode video gốc (coarse-to-fine ±3 s), TRAKE refine từng event ±1.5 s và ép thứ tự đơn điệu.
- **Tên ảnh review chính xác**: mỗi panel được xuất thành `frames/<video_id>/<frame_idx 6 số>.jpg`; `frame_catalog.csv` ánh xạ query/rank/event sang ảnh này và keyframe nguồn `<video_id>/<keyframe_n 3 số>.jpg`.
- **Missing modality không loại candidate**: mọi nhánh đóng góp qua RRF có trọng số; object detection chỉ soft-boost.
- **Submission**: 1 CSV/query, UTF-8, không header, ≤100 dòng, video_id không `.mp4`, Q&A ≤100 ký tự, TRAKE đúng N frame tăng dần, ZIP luôn chứa thư mục `submission/`. Validator chặn tạo ZIP khi còn lỗi P0.

## Tham số nên tune trước (đo trên validation set gán nhãn tay)

`CFG["W_*"]` (trọng số RRF theo profile query), `VIDEO_PRIOR_W`, `RERANK_W`, `NMS_TIME_SEC`,
`MAX_PER_VIDEO_TOP20`, `REFINE_WIN_SEC`, `TRAKE_PER_EVENT`/`TRAKE_BEAM`.
Thêm từng thay đổi một theo experiment ladder ở mục 16 của prompt kế hoạch.
