# Script vòng Sơ tuyển AIC 2026 — bộ đề Thử nghiệm P1

`aic2026_sotuyen_p1_pipeline.ipynb` — notebook Kaggle (2×GPU 16GB) sinh ra `submission.zip`
cho 24 truy vấn trong `THUNGHIEM-bo-de-thi` (18 KIS, 3 Q&A, 3 TRAKE).

## Chuẩn bị trên Kaggle

1. Upload `Feature_Dataset` thành Kaggle Dataset (clip-features-32, map-keyframes,
   Image_captioning, OCR_EasyOCR_VietOCR, ocr_index.jsonl, Summary_video,
   Transcript_Translated, objects-aic25-b1, media-info).
2. Upload `THUNGHIEM-bo-de-thi` (24 file `query-p1-*.txt`).
3. *(nên có)* Add dataset `Videos_L*` → bật `USE_VIDEO_FINE_ALIGN = True` trong CFG.
   TRAKE yêu cầu frame chính xác trong khoảng < 10 frame; keyframe có sẵn quá thưa
   (cách nhau hàng chục–hàng trăm frame) nên **gần như không thể trúng nếu không quét ở mức frame**.
4. *(nên có)* Add dataset `Keyframes_L*` → bật `USE_VLM_RERANK = True` để VLM xem ảnh thật.
5. Add-ons → Secrets → thêm `ANTHROPIC_API_KEY`.
6. Settings: Accelerator = GPU T4 ×2, Internet = On.
7. Run All. Kết quả: `/kaggle/working/submission.zip` (bên trong có thư mục `submission/`).

Notebook tự nhận diện đường dẫn dataset theo tên thư mục — không cần sửa path.

## Thời gian chạy

| Bước | Lần đầu | Lần sau (có cache trong `/kaggle/working/cache`) |
|---|---|---|
| Index keyframe + nạp CLIP features | ~3 phút | vài giây |
| Index object (177k JSON) | 5–10 phút | ~30 giây |
| Encode dense 177k keyframe doc (e5-large, 2 GPU) | 40–70 phút | vài giây |
| BM25 index | ~2 phút | vài giây |
| 24 truy vấn (LLM + rerank) | 10–20 phút | nhanh hơn nhờ `llm_cache.json` |

Nên **Save Version** sau lần chạy đầu để giữ cache.
Nếu muốn chạy nhanh để thử: đặt `USE_DENSE = False`, `USE_OBJECTS = False`.

## Kiến trúc

- **Query understanding** (LLM): truy vấn tiếng Việt → `topic_en/vi`, `clip_prompts[]` (mô tả thị giác
  kiểu caption để khớp CLIP), `keywords_vi/en`, `ocr_terms[]`, `objects[]`, và `events[]` cho TRAKE.
  LLM cũng giải luôn tri thức ngoài (ví dụ "phim Spielberg 1975" → *great white shark*,
  "đại học ở Lausanne" → *EPFL*).
- **Truy hồi 2 tầng**: video-level (BM25 + dense trên summary/transcript/media-info) làm *prior*,
  keyframe-level gộp CLIP ViT-B/32 + dense e5 (caption + OCR) + BM25 (caption+OCR+ASR) +
  bonus OCR exact-match + bonus object.
- **Rerank**: LLM trên text evidence, tuỳ chọn thêm VLM trên ảnh keyframe.
- **Sinh dòng nộp**: hedging — vì `Final = mean(R@1, R@5, R@20, R@50, R@100)` và cửa sổ đáp án rất hẹp,
  các hạng sau trải thêm frame trong cùng shot / offset quanh frame dự đoán.
- **TRAKE**: DP căn chỉnh đơn điệu (E1 < E2 < ... < EN) chọn video, rồi fine-align mức frame bằng
  CLIP ảnh-frame trên video gốc, cuối cùng quét offset + jitter để phủ 100 dòng.

## Kiểm tra trước khi nộp

Cell 15 tự validate: đúng 100 dòng, `video_id` khớp `L\d\d_V\d\d\d` (không có `.mp4`),
frame là số nguyên, số frame TRAKE khớp số event, answer Q&A ≤ 100 ký tự,
zip có thư mục `submission/`, CSV thuần UTF-8 không header.

Mỗi gói được nộp tối đa 3 lần và **chỉ lần cuối tính điểm** — nên nộp thử 1 lần để xem
Public LB (50% đáp án), điều chỉnh CFG rồi nộp lần cuối.
