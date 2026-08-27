# Pipeline API — AIC 2026 (Kaggle, 3 Notebook)

Sinh từ `Kiet-Prompt/Prompt_Plan_API.md`. Toàn bộ logic nằm trong notebook, **không có tệp `.py`**.

> Hướng dẫn chạy từng bước trên Kaggle GPU: **[HUONG_DAN_CHAY_KAGGLE.md](HUONG_DAN_CHAY_KAGGLE.md)**

| Notebook | Vai trò | Đầu ra chính |
|---|---|---|
| `01-build-indices-api.ipynb` | discovery + audit + canonical records + FAISS/BM25/Object Index + text embedding | `/kaggle/working/artifacts/` + `artifact_manifest.json` |
| `02-retrieve-refine-candidates-api.ipynb` | parse query, retrieval đa nhánh, fusion, rerank, frame refinement, KIS/Q&A/TRAKE, **xuất submission** | `/kaggle/working/review_package/` + `review_manifest.json` + **`/kaggle/working/submission.zip`** |
| `03_human_review_submit_api.ipynb` | human review, validator, ZIP | `submission.zip` + `validation_report.csv` |

Chạy đúng thứ tự 01 → 02 → 03. NB03 **không gọi API, không build lại index** — sửa review rồi gọi `Regenerate()` để tạo ZIP mới.

> **NB03 hiện không còn trong repo.** Để output của NB02 luôn có gói nộp hợp lệ, NB02 đã có **CELL 17 — EXPORT SUBMISSION**:
> ghi `submission/<query_id>.csv` theo đúng `TheLeCuocThi-DeThi/sotuyenAIC.md`, tự validate format rồi nén `/kaggle/working/submission.zip`
> (bên trong zip **có** thư mục `submission/`). Q&A cần answer: tạo `/kaggle/working/qa_answers.json` dạng
> `{"query-p1-15-qa": "12"}` rồi chạy lại riêng CELL 17.

## Dataset trên Kaggle

Attach 3 dataset: `fatle542/AIC-Dataset`, `kitnehi1211/feature-AIC-2026`, `kitnehi1211/dethithunghiem`.
Notebook **tự dò** dataset root trong `/kaggle/input` (không hard-code path); override bằng
`CFG["feature_root"]`, `CFG["dataset_root"]` hoặc `CFG["query_dir"]` khi cần.

## API key

Chỉ đọc từ biến môi trường `OPENROUTER_API_KEY` hoặc Kaggle Secret cùng tên. Trong code, biến key để rỗng `""`.
Key không bao giờ được log hay ghi vào artifact.

## Chi phí — mặc định an toàn

Cả Notebook 01/02 mặc định `CFG["DRY_RUN"] = True`: chỉ đếm đọc/token và in bảng ước tính, **không gọi API**.
Muốn chạy thật thì đặt `DRY_RUN = False`.

- Hard cap dùng chung: `MAX_TOTAL_COST_USD = 2.00` trong `artifacts/cost_ledger.json`; cảnh báo ở 80%, dừng ở 90%.
- MiMo mặc định chỉ lập kế hoạch query, dịch `q_en` và tạo `retrieval_queries`; human review là bước kiểm tra visual và sửa answer.
- `allow_vlm_verify = False`, `vlm_topm_by_type = {"kis": 0, "qa": 0, "trake": 0}`.
- Trần riêng: NB01 embedding `$0.45`, NB02 `$1.20`.
- Cache content-addressed cho embedding/rerank/LLM nên chạy lại không tốn thêm tiền.

## Model

- Visual: `google/siglip2-giant-opt-patch16-384`.
- Text embedding: `openai/text-embedding-3-small` qua OpenRouter — **FAISS index riêng**.
- Text reranker: `voyageai/rerank-2.5-lite`.
- Query analysis/structured parse/translation: `xiaomi/mimo-v2.5`.
- Lexical: BM25-Okapi tự cài trên `scipy.sparse` (local, không cần API).

## Fallback khi thiếu thành phần

| Thiếu | Hành vi |
|---|---|
| Không load được SigLIP2 | tắt nhánh visual, chạy bằng text branch, ghi `degraded.visual_branch=False` |
| Chưa có text-embedding index | chỉ dùng BM25 + visual |
| Không có object index | tắt object soft boost; object không bao giờ là hard filter |
| Không có video gốc/cv2 | tắt frame refinement, dùng `frame_idx` của keyframe |
| MiMo lỗi/DRY_RUN | dùng parser rule-based; Q&A để answer trống để người review điền |
| Video thiếu OCR/ASR/caption/metadata | candidate không bị loại, chỉ mất nhánh tương ứng |

## Checkpoint/resume

- NB01: `_done_<stage>.json` cho từng stage; build lại bằng `CFG["force_stages"]`.
- NB02: checkpoint theo query tại `review_package/candidates/<query_id>.json`.
- NB03: `review_decisions.json` được ghi atomic sau mỗi thao tác.

## Tham số quan trọng của NB02

- `topk.*`: top-k của từng retriever.
- `weights_*`: các profile trọng số, tự chọn theo structured query.
- `rrf_k = 60`, `video_prior_alpha = 0.25`, `rerank_weight = 0.9`.
- `nms_time_window_s = 2.0`, `max_per_video_top5 = 2`.
- `refine_window_s = 3.0`, `refine_topk = 12`.
- `trake_videos = 6`, `trake_beam = 8`, `trake_seq_per_query = 40`.

## Validator (NB03)

CSV/query phải đúng tên query, UTF-8, dấu phẩy, không header, tối đa 100 dòng; `video_id` không có `.mp4`,
`frame_id` là số nguyên, Q&A đúng 3 cột và answer không quá 100 ký tự. TRAKE phải có đúng N frame theo thứ tự tăng dần.
ZIP luôn chứa thư mục `Submission/`.
