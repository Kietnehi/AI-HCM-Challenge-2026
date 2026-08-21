# 🗂️ Planning Package — AIC 2026 Vòng Sơ Tuyển

> **Đây là entry point.** Agent/dev đọc file này trước, rồi đọc theo thứ tự bên dưới.

## Đọc theo thứ tự

| # | File | Nội dung | Bắt buộc đọc khi... |
|:-:|:--|:--|:--|
| 0 | [`00_MASTER_PLAN.md`](./00_MASTER_PLAN.md) | Bối cảnh, kiến trúc tổng, **6 quyết định kỹ thuật quan trọng** (có 3 điểm chặn cần biết trước khi code), config block, risk register | **Luôn luôn** |
| 1 | [`01_DATA_CONTRACTS.md`](./01_DATA_CONTRACTS.md) | Schema chính xác của mọi artifact trung gian (parquet / faiss / json) | Khi viết bất kỳ notebook nào |
| 2 | [`02_NB01_INDEX_BUILDER.md`](./02_NB01_INDEX_BUILDER.md) | Spec cell-by-cell Notebook 1: corpus → embedding → FAISS + BM25 | Khi làm phần index |
| 3 | [`03_NB02_PIPELINE_SUBMIT.md`](./03_NB02_PIPELINE_SUBMIT.md) | Spec cell-by-cell Notebook 2: query → retrieval → fusion → rerank → CSV/zip | Khi làm phần pipeline |
| 4 | [`04_TASK_CHECKLIST.md`](./04_TASK_CHECKLIST.md) | Backlog dạng checkbox, thứ tự thực thi, definition-of-done từng task | Khi bắt đầu / báo cáo tiến độ |
| 5 | [`05_KAGGLE_PATHS.md`](./05_KAGGLE_PATHS.md) | **Path Kaggle thật (đã verify qua API)** + resolver + đặc điểm dữ liệu dễ gây bug | **Trước khi viết dòng code đầu tiên** |

## Quy tắc dành cho AI Agent

1. **KHÔNG tự đổi model ID.** Mọi model ID nằm trong `CONFIG` cell (xem `00_MASTER_PLAN.md §5`). Nếu thấy model không khả dụng → báo lại, đừng thay thầm.
2. **KHÔNG hard-code API key.** Luôn ghi `OPENROUTER_API_KEY = ""` và `OPENAI_API_KEY = ""` để user tự điền.
2b. **KHÔNG tự bịa path.** Copy `PATHS` block + `resolve()` từ `05_KAGGLE_PATHS.md §2-3`. Path đã verify thật, đừng đoán lại.
3. **Mọi bước gọi API phải resumable.** Kaggle timeout 12h khi "Save & Version". Checkpoint mỗi N batch ra `/kaggle/working`, khởi động lại thì skip phần đã xong.
4. **`kf_id` là khóa chính duy nhất** của toàn hệ thống: `f"{video_id}#{n:03d}"`. Mọi channel phải resolve về `kf_id`.
5. **Giá trị nộp bài là `frame_idx`**, không phải `n`, không phải `pts_time`. Luôn lookup qua `map-keyframes`.
6. **Trước khi kết luận "xong"**, chạy validator ở `03_NB02_PIPELINE_SUBMIT.md §8`.

## Nguồn dữ liệu tham chiếu (đọc khi cần chi tiết)

- Đặc tả features: `../Feature_Dataset/README.md`
- Thể lệ & format nộp bài: `../TheLeCuocThi/sotuyenAIC.md`
- Bộ đề thử nghiệm (**24 query thật** — 18 KIS, 3 QA, 3 TRAKE): local `../THUNGHIEM-bo-de-thi/` · Kaggle `kitnehi1211/dethithunghiem`
- Yêu cầu gốc từ user: `../Kiet-Prompt/Prompt_Plan.txt`, `Model_Embedding.txt`
- Kaggle datasets & repo tham khảo: `../Link.txt` — **path mount thật đã verify ở `05_KAGGLE_PATHS.md`**
