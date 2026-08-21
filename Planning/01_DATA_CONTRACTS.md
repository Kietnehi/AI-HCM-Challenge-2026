# 01 — DATA CONTRACTS

> **Hợp đồng dữ liệu giữa NB01 và NB02.** Mọi artifact phải khớp schema dưới đây. Nếu cần đổi schema thì sửa file này TRƯỚC, rồi mới sửa code.
>
> File này nói **cấu trúc dữ liệu**, không nói path. Path thật (đã verify qua Kaggle API) ở [`05_KAGGLE_PATHS.md`](./05_KAGGLE_PATHS.md) — dùng biến `P_CLIP`, `P_MAPKF`, `P_CAPTION`, … từ PATHS block.

## 0. Khóa chính toàn hệ thống

```python
kf_id = f"{video_id}#{n:03d}"     # ví dụ: "L21_V001#001"
```

| Khái niệm | Ý nghĩa | Dùng ở đâu |
|:--|:--|:--|
| `video_id` | `L21_V001` — không có `.mp4` | Nộp bài (cột 1) |
| `n` | Số thứ tự keyframe, **1-based**. `n=1` → file `001.jpg` | Nội bộ, dựng path ảnh |
| `frame_idx` | Chỉ số frame thực trong video | **NỘP BÀI (cột 2+)** |
| `pts_time` | Mốc thời gian (giây) | Align ASR/event theo thời gian |

> **Sai lầm chết người:** nộp `n` hoặc `pts_time` thay cho `frame_idx`. Luôn lookup qua `map-keyframes`.

---

## 1. `keyframes.parquet` — 177,321 dòng

Một dòng / keyframe. Đây là bảng gốc để resolve mọi kết quả về format nộp bài.

| Cột | Dtype | Null? | Nguồn | Ghi chú |
|:--|:--|:-:|:--|:--|
| `kf_id` | `string` | ✗ | computed | **PK**, `f"{video_id}#{n:03d}"` |
| `video_id` | `category` | ✗ | tên file | |
| `n` | `int32` | ✗ | `map-keyframes.csv:n` | cast từ float (`1.0` → `1`) |
| `frame_idx` | `int32` | ✗ | `map-keyframes.csv:frame_idx` | **giá trị nộp bài** |
| `pts_time` | `float32` | ✗ | `map-keyframes.csv:pts_time` | giây |
| `fps` | `float32` | ✗ | `map-keyframes.csv:fps` | |
| `kf_path` | `string` | ✗ | computed | relative: `f"{video_id}/{n:03d}.jpg"` — **verify QĐ-4** |
| `vis_row` | `int32` | ✗ | computed | row index trong `vision.faiss` |

**Build:** đọc 873 file `map-keyframes-aic25-b1/map-keyframes/<video_id>.csv`, concat, sort theo `(video_id, n)`, rồi gán `vis_row = range(len(df))`.

> **Invariant bắt buộc kiểm:** với mỗi `video_id`, số dòng trong `keyframes.parquet` phải **== `np.load(clip-features/<video_id>.npy).shape[0]`**. Nếu lệch → visual FAISS sẽ lệch hàng và toàn bộ kết quả visual sai. Log ra danh sách video lệch và **dừng lại**, đừng tự sửa.

---

## 2. `videos.parquet` — 873 dòng

| Cột | Dtype | Null? | Nguồn |
|:--|:--|:-:|:--|
| `video_id` | `string` | ✗ | **PK** |
| `n_keyframes` | `int32` | ✗ | count từ `keyframes.parquet` |
| `duration_sec` | `float32` | ✓ | `media-info.length` hoặc `max(pts_time)` |
| `fps` | `float32` | ✗ | mode của `keyframes.fps` |
| `title` | `string` | ✓ | `media-info.title` |
| `description` | `string` | ✓ | `media-info.description` |
| `keywords` | `list<string>` | ✓ | `media-info.keywords` |
| `author` | `string` | ✓ | `media-info.author` |
| `publish_date` | `string` | ✓ | `media-info.publish_date` |
| `watch_url` | `string` | ✓ | `media-info.watch_url` |
| `summary_en` | `string` | ✓ | `Summary_video/<id>.json:summary` |
| `topics` | `list<string>` | ✓ | `Summary_video:topics` |
| `entities` | `list<string>` | ✓ | `Summary_video:entities` |
| `transcript_vi` | `string` | ✓ | `Transcript_Translated:text` (full) |
| `transcript_en` | `string` | ✓ | `Transcript_Translated:text_en` (full) |
| `has_ocr` | `bool` | ✗ | tồn tại trong `ocr_index.jsonl` |
| `has_summary` | `bool` | ✗ | |
| `has_transcript` | `bool` | ✗ | `len(transcript_en) > 0` |

**Fallback bắt buộc:** `summary_en` null → dùng `description`. Nếu cả hai null → chuỗi rỗng, và `has_summary=False`.

---

## 3. `text_units.parquet` — ~319,000 dòng

Corpus text hợp nhất. **Mọi** kênh text (embedding + BM25) đọc từ đây.

| Cột | Dtype | Null? | Ghi chú |
|:--|:--|:-:|:--|
| `unit_id` | `string` | ✗ | **PK**, format ở §3.1 |
| `channel` | `category` | ✗ | `caption` \| `ocr` \| `asr` \| `summary` \| `meta` |
| `video_id` | `category` | ✗ | |
| `kf_id` | `string` | ✓ | null cho `asr`/`summary`/`meta` |
| `frame_idx` | `int32` | ✓ | **anchor để nộp bài**; với `asr` = frame_idx của keyframe gần `t_start` nhất |
| `t_start` | `float32` | ✓ | giây |
| `t_end` | `float32` | ✓ | giây |
| `text_en` | `string` | ✓ | dùng cho embedding + BM25-EN |
| `text_vi` | `string` | ✓ | dùng cho BM25-VI |
| `text_embed` | `string` | ✗ | **text thực tế gửi đi embed** (xem §3.2) |
| `lang_native` | `category` | ✗ | `vi` \| `en` \| `mixed` |
| `emb_row` | `int32` | ✗ | row trong FAISS của channel đó; `-1` = không embed |

### 3.1 Format `unit_id`

| Channel | `unit_id` |
|:--|:--|
| caption | `cap:{video_id}#{n:03d}` |
| ocr | `ocr:{video_id}#{n:03d}` |
| asr | `asr:{video_id}#{win_idx:04d}` |
| summary | `sum:{video_id}` (video-level) hoặc `sum:{video_id}#c{chunk_idx:02d}` |
| meta | `meta:{video_id}` |

### 3.2 Luật build từng channel

#### `caption` — ~163,000 unit
- Nguồn: `Image_captioning/<video_id>.json` → `keyframes[]`
- **Bỏ qua** phần tử có `duplicate_of != null` khi embed (`emb_row = -1`), nhưng **vẫn tạo dòng** trong `text_units` và set `text_en` = caption của keyframe canonical. ⚠️ `duplicate_of` là **tên file** (`"009.jpg"`), không phải `kf_id` — phải parse ra `n` rồi mới dựng `f"{video_id}#{n:03d}"`. (Đã verify trên `L21_V001`: 28/28 caption dup **giống hệt** canonical → chiến lược không embed lại là đúng, không mất thông tin.) Lý do: keyframe trùng vẫn là kết quả nộp bài hợp lệ, chỉ không cần tốn tiền embed lại.
- Sau khi build FAISS: với dòng dup, gán `emb_row = emb_row của canonical` để retrieval vẫn trả về nó.
- `text_en` = `caption`; `text_vi` = null; `lang_native = "en"`
- `text_embed` = `caption`
- `frame_idx`, `t_start = t_end = pts_time` lấy từ `keyframes.parquet` (join qua `kf_id`)

#### `ocr` — **128,664 unit** (1 unit / keyframe)
- Nguồn: `ocr_index.jsonl` — mỗi dòng là `{video_id, frame_idx, pts_time, keyframe, text}`.
- **KHÔNG gộp gì cả.** Đã verify: 128,664 dòng ↔ **128,664 cặp `(video_id, keyframe)` unique**, max 1 dòng/keyframe. File đã được gộp sẵn ở khâu trích xuất. Mọi bước "nối text bằng `" | "`" là no-op — đừng viết.
- **Không cần join `keyframes.parquet`** để lấy `frame_idx`/`pts_time`: mỗi dòng đã có sẵn. Vẫn nên assert giá trị khớp với `keyframes.parquet` để bắt lệch dữ liệu.
- `text_vi` = `text`; `text_en` = null; `lang_native = "vi"`
- `text_embed` = `text_vi` — **embed trực tiếp tiếng Việt**, KHÔNG dịch. Lý do: `text-embedding-3` xử lý tiếng Việt đủ tốt, và dịch 128K đoạn OCR ngắn (thường là fragment vô nghĩa) qua VLM thì đắt và nhiễu.
- Lọc rác: bỏ unit có `len(text_vi.strip()) < 2`. Lưu ý phần lớn OCR news là nhiễu logo/đồng hồ (`"Thập H 06:30:11 giây"`) — đừng kỳ vọng mọi unit đều có nghĩa, nhưng **đừng lọc mạnh tay**: chuỗi giá trị nhất (tên xã, tên CLB, tiêu đề công thức) nằm lẫn trong đó.
- Chỉ 700/873 video có OCR → `has_ocr=False` cho phần còn lại.

#### `asr` — ~21,500 unit (sau dedupe)
- Nguồn: `Transcript_Translated/Videos_L*_?/video/<video_id>.json` → `segments[]` (mỗi segment có `start`, `end`, `text`, `text_en`).
- **Windowing:** gộp segment liên tiếp thành cửa sổ ~**25 giây, stride 10 giây** (overlap để không cắt mất ngữ cảnh). Nếu video ngắn hơn 25s thì 1 window.
- **BẮT BUỘC dedupe sau khi window.** Segment ASR ở bộ này dài trung bình **24.2 giây** — gần bằng đúng kích thước cửa sổ. Với stride 10s, cùng một segment rơi vào 2–3 cửa sổ liên tiếp và sinh ra unit **giống hệt nhau**. Đo trên 40 video ngẫu nhiên: 1,600 window nhưng chỉ **985 tập segment khác nhau → dư 1.62×**.
  Hệ quả nếu không dedupe: phí ~38% ngân sách embed kênh ASR, và tệ hơn — cùng một đoạn text xuất hiện ở **nhiều rank** trong cùng channel, tự khuếch đại điểm RRF của chính nó.
  Cách sửa: khoá dedupe = `tuple(sorted(seg.id for seg in window))`, giữ window đầu tiên. Ước lượng sau dedupe: ~35,000 → **~21,500 unit**.
- `text_vi` = nối `segment.text`; `text_en` = nối `segment.text_en`
- `text_embed` = `text_en` (fallback `text_vi` nếu `text_en` rỗng)
- `lang_native = "vi"` (gốc là tiếng Việt)
- `frame_idx` = frame_idx của keyframe có `pts_time` gần `(t_start+t_end)/2` nhất

#### `summary` — ~5,200 unit
- Nguồn: `Summary_video/<video_id>.json`
- Tạo **1 unit video-level** (`summary` + `" | ".join(topics)` + `" | ".join(entities)`) và **1 unit / `chunk_summaries[i]`**.
- Nếu `chunk_summaries[i]` có mốc thời gian → set `t_start/t_end/frame_idx`; không có thì để null.
- `text_en` = nội dung; `lang_native = "en"`; `text_embed = text_en`
- ⚠️ **KHÔNG dùng `evidence`.** Đã verify: `evidence` **không phải** list sự kiện có `timestamp` + `description`, mà là một **dict metadata provenance**: `{visual_count, speech_count, has_caption, has_transcript, duration_hint, caption_source, transcript_source, ...}`. Không có trường thời gian nào. (Bản plan trước mô tả sai trường này.)
- **Tín hiệu thời gian thật nằm trong `chunk_summaries`**, dạng chuỗi có tiền tố mốc: `"- [00:00-02:00] A Vietnamese news report from ..."`. Đây là **nguồn text-theo-thời-gian duy nhất** ở cấp video, nên đáng đầu tư parse cho chắc:
  ```python
  m = re.match(r"\s*-?\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\]", chunk)
  # -> t_start, t_end (giây); frame_idx = keyframe gần (t_start+t_end)/2 nhất
  # khong match -> t_start = t_end = frame_idx = null, van giu unit
  ```
- **Skip file sentinel:** `Summary_video/` chứa `_failed.json` không phải video. Bỏ qua mọi stem bắt đầu bằng `_`.

#### `meta` — 873 unit
- `text_embed` = `f"{title}. {description}. Keywords: {', '.join(keywords)}"`
- `text_vi` = `text_en` = cùng nội dung; `lang_native = "mixed"`
- `kf_id`, `frame_idx`, `t_start`, `t_end` = null (video-level)

---

## 4. `objects.parquet` — bảng phẳng cho hard/soft filter

Đọc 177,321 file JSON nhỏ trong `objects-aic25-b1/objects/` thì **rất chậm**. Phải flatten 1 lần ở NB01.

| Cột | Dtype | Ghi chú |
|:--|:--|:--|
| `kf_id` | `string` | |
| `classes` | `list<string>` | `detection_class_entities` đã lọc `float(score) >= 0.30`, unique. ⚠️ `detection_scores` trong JSON là **chuỗi** (`"0.79673874"`) — so sánh trực tiếp với float sẽ `TypeError`, phải cast |
| `scores` | `list<float32>` | max score / class, cùng thứ tự với `classes` |

**Phụ trợ — ⚠️ `detected_classes.txt` KHÔNG phải danh sách thuần, đừng copy nguyên file.**

Đã verify: file có **587 dòng, line ending CRLF**, gồm 2 dòng comment `#`, 1 dòng trống, rồi mới 584 tên class:

```
# Total Unique Detected Classes: 584
# Format: Class_Entity | Class_Name_ID | Class_Label | ...
<dòng trống>
Person
Clothing
...
```

Phải parse rồi mới ghi ra `object_classes.txt` sạch 584 dòng:

```python
def load_object_classes(path):
    lines = [l.strip() for l in open(path, encoding="utf-8")]   # .strip() bo luon \r
    cls = [l for l in lines if l and not l.startswith("#")]
    assert len(cls) == 584, len(cls)
    return cls
```

Hai cách sai đã thấy trong bản plan cũ, **cả hai đều hỏng âm thầm**:
- `enumerate(f.read().splitlines())` → giữ 2 comment + 1 dòng trống ⇒ **lệch index 3 cột**, object bonus chấm nhầm class mà không báo lỗi.
- `f.read().split("\n")` → mọi tên dính `\r` (`"Person\r"`) ⇒ **không class nào match**, kênh object im lặng trả 0.

NB02 Stage 0 **chỉ được** sinh `object_classes` nằm trong danh sách đã parse này.

**Tối ưu:** thay vì `list<string>`, có thể lưu thêm `objects_matrix.npz` — sparse bool matrix `[177321, 584]` (~13MB nén). Filter bằng phép nhân ma trận thay vì duyệt Python → nhanh hơn ~100×. Khuyến nghị làm cả hai.

---

## 5. FAISS indexes

| File | Type | Dim | N vectors | Metric |
|:--|:--|--:|--:|:--|
| `vision.faiss` | `IndexFlatIP` | 512 | 177,321 | cosine (L2-normalize trước) |
| `text_caption.faiss` | `IndexFlatIP` | 1536 | ~163,000 | cosine |
| `text_ocr.faiss` | `IndexFlatIP` | 1536 | **128,664** | cosine |
| `text_asr.faiss` | `IndexFlatIP` | 1536 | ~21,500 | cosine |
| `text_summary.faiss` | `IndexFlatIP` | 1536 | ~5,200 | cosine |
| `text_meta.faiss` | `IndexFlatIP` | 1536 | 873 | cosine |

**Quy tắc:**
- Dùng `IndexFlatIP` (exact), **không** dùng IVF/HNSW. Ở scale ≤ 200K vector thì flat search chỉ mất ~10-50ms và cho recall 100%. Đừng đánh đổi accuracy để lấy tốc độ mình không cần.
- **Bắt buộc L2-normalize** mọi vector trước khi add (`faiss.normalize_L2`) — nếu không thì inner-product ≠ cosine và điểm sai lệch.
- CLIP `.npy` là `float16` → **cast sang `float32`** trước khi add, FAISS không nhận fp16.
- Mapping row → id: `vision.faiss[i]` ↔ `keyframes[vis_row == i]`; `text_X.faiss[i]` ↔ `text_units[(channel==X) & (emb_row == i)]`. Lưu thêm 2 file `.npy` mapping ngược để tra cứu O(1):
  - `vision_rowmap.npy` → array of `kf_id` (dtype object hoặc `<U16`)
  - `text_{channel}_rowmap.npy` → array of `unit_id`

---

## 6. BM25 indexes

Dùng thư viện **`bm25s`** (`pip install bm25s`) thay vì `rank_bm25`. Lý do: `bm25s` dùng sparse scipy matrix, truy vấn ~ms trên 160K doc; `rank_bm25` là pure-Python và mất hàng giây → không chấp nhận được khi phải chạy nhiều channel × nhiều query.

| File | Corpus | Tokenizer |
|:--|:--|:--|
| `bm25_caption_en/` | `text_units[channel=caption].text_en` | EN: lowercase, strip punct |
| `bm25_ocr_vi/` | `text_units[channel=ocr].text_vi` | VI (xem dưới) |
| `bm25_asr_vi/` | `text_units[channel=asr].text_vi` | VI |
| `bm25_asr_en/` | `text_units[channel=asr].text_en` | EN |
| `bm25_summary_en/` | `text_units[channel=summary].text_en` | EN |
| `bm25_meta/` | `text_units[channel=meta].text_embed` | VI (metadata chủ yếu tiếng Việt) |

**Tokenizer tiếng Việt** — thứ tự ưu tiên:
1. `pyvi.ViTokenizer.tokenize()` (word segmentation, tốt nhất cho tiếng Việt)
2. Fallback nếu `pyvi` lỗi/không cài được: lowercase + `unicodedata.normalize('NFC')` + split trên `\W+`

> **Quan trọng:** phải **normalize Unicode NFC** cho cả corpus và query. Tiếng Việt có 2 dạng dựng sẵn/tổ hợp (`ế` = 1 codepoint hoặc `e` + dấu) — không normalize thì BM25 miss hoàn toàn. Đây là bug rất hay gặp và rất khó phát hiện.

Mỗi index lưu kèm `ids.npy` = array `unit_id` theo đúng thứ tự corpus.

---

## 7. `parsed_queries.json` — output NB02 Stage 0

Artifact **quan trọng nhất để review tay**. Sinh ra xong thì đọc lại bằng mắt trước khi chạy retrieval.

```json
[
  {
    "query_id": "query-p1-16-trake",
    "query_file": "query-p1-16-trake.txt",
    "type": "trake",
    "q_vi": "<nguyên văn file .txt>",
    "q_en": "A lion dance performance with a yellow, black and white lion...",
    "visual_desc_en": "yellow black white lion dance on poles, judges, dragon",
    "keywords_vi": ["múa lân", "cột", "ban giám khảo", "con rồng"],
    "keywords_en": ["lion dance", "pole", "judges", "dragon"],
    "ocr_hints": [],
    "named_entities": [],
    "object_classes": ["Person", "Dragon"],
    "question_en": null,
    "n_events": 4,
    "events": [
      {"idx": 1, "desc_vi": "Lân quay vòng trên cột số 4...", "desc_en": "Lion spins on pole 4 using front legs", "visual_desc_en": "lion spinning on top of pole"},
      {"idx": 2, "desc_vi": "...", "desc_en": "...", "visual_desc_en": "..."},
      {"idx": 3, "desc_vi": "...", "desc_en": "...", "visual_desc_en": "..."},
      {"idx": 4, "desc_vi": "...", "desc_en": "...", "visual_desc_en": "..."}
    ]
  }
]
```

**Quy tắc validate ngay sau Stage 0 (fail fast):**

| Check | Điều kiện |
|:--|:--|
| `type` khớp hậu tố tên file | `kis` / `qa` / `trake` |
| `type == "trake"` | `n_events == len(events)` và `n_events >= 2` |
| `type == "qa"` | `question_en` không null |
| Mọi type | `visual_desc_en` ≤ 60 từ (giới hạn 77 token của CLIP) |
| `object_classes` | tập con của 584 class trong `object_classes.txt` |
| `n_events` | Đếm **số DÒNG** khớp regex trong file gốc, **KHÔNG tin LLM đếm**, và **KHÔNG dedupe/sort theo con số sau chữ E** |

> **Lưu ý về `n_events`:** ba query TRAKE có ba cấu trúc khác nhau —
> - `query-p1-4-trake.txt`: 4 dòng `E1:`–`E4:`, không có dòng mô tả cảnh chung.
> - `query-p1-16-trake.txt`: 1 dòng mô tả cảnh + 4 dòng event.
> - `query-p1-18-trake.txt`: ⚠️ **BTC đánh máy sai** — các dòng là `E1:`, `E2:`, **`E2:`**, `E4:`. Số `E3` không tồn tại và `E2` xuất hiện hai lần.
>
> **Hệ quả:** đếm theo **số dòng** ra 4 → đúng. Nhưng bất kỳ chỗ nào `set()`, dedupe theo con số, hoặc `sort(key=idx)` sẽ ra **3 event** → sai số cột → **0 điểm câu đó**. Quy tắc bắt buộc: `n_events = số dòng khớp`, và **thứ tự event = thứ tự dòng trong file**, con số sau `E` chỉ để tham khảo chứ không dùng để sắp xếp hay đánh khoá.
>
> Lỗi này gần như chắc chắn còn lặp lại ở gói đề thật — đừng "sửa hộ" BTC, cứ bám thứ tự dòng.

---

## 8. `candidates_<query_id>.parquet` — output NB02 Stage 2/3 (debug artifact)

Lưu lại để debug và tune weight mà không phải chạy lại retrieval.

| Cột | Dtype | Ghi chú |
|:--|:--|:--|
| `kf_id` | `string` | |
| `video_id` | `string` | |
| `frame_idx` | `int32` | |
| `fused_score` | `float32` | điểm RRF sau fusion |
| `rank_vision`, `rank_caption`, `rank_ocr`, `rank_asr` | `int32` | rank trong channel đó, `-1` nếu không xuất hiện |
| `video_prior` | `float32` | |
| `object_bonus` | `float32` | |
| `rerank_text_score` | `float32` | null nếu chưa qua Stage 3 |
| `rerank_vlm_score` | `float32` | null nếu chưa qua Stage 4 |
| `vlm_reason` | `string` | lý do VLM cho điểm — **rất hữu ích để debug** |

---

## 9. Cấu trúc dataset output của NB01

Publish `/kaggle/working/index/` thành Kaggle Dataset tên **`aic26-index`**:

```
aic26-index/
├── keyframes.parquet
├── videos.parquet
├── text_units.parquet
├── objects.parquet
├── objects_matrix.npz
├── object_classes.txt
├── faiss/
│   ├── vision.faiss
│   ├── vision_rowmap.npy
│   ├── text_caption.faiss
│   ├── text_caption_rowmap.npy
│   ├── text_ocr.faiss
│   ├── text_ocr_rowmap.npy
│   ├── text_asr.faiss
│   ├── text_asr_rowmap.npy
│   ├── text_summary.faiss
│   ├── text_summary_rowmap.npy
│   ├── text_meta.faiss
│   └── text_meta_rowmap.npy
├── bm25/
│   ├── bm25_caption_en/       (bm25s save dir + ids.npy)
│   ├── bm25_ocr_vi/
│   ├── bm25_asr_vi/
│   ├── bm25_asr_en/
│   ├── bm25_summary_en/
│   └── bm25_meta/
└── BUILD_MANIFEST.json
```

### `BUILD_MANIFEST.json` — bắt buộc có

Không có manifest thì NB02 không biết index được build bằng model nào và sẽ dùng sai text encoder (QĐ-2). Đây là cơ chế chống sai lệch quan trọng nhất.

```json
{
  "built_at": "2026-08-21T10:00:00Z",
  "embed_provider": "openai",
  "embed_model": "text-embedding-3-small",
  "embed_dim": 1536,
  "visual_model": "clip-ViT-B-32",
  "visual_dim": 512,
  "n_videos": 873,
  "n_keyframes": 177321,
  "n_text_units": {"caption": 0, "ocr": 0, "asr": 0, "summary": 0, "meta": 0},
  "n_embedded": {"caption": 0, "ocr": 0, "asr": 0, "summary": 0, "meta": 0},
  "asr_window_sec": 25,
  "asr_stride_sec": 10,
  "asr_dedup_by_segment_set": true,
  "ocr_min_len": 2,
  "object_score_threshold": 0.30,
  "n_object_classes": 584,
  "videos_missing_ocr": [],
  "videos_missing_summary": [],
  "videos_keyframe_count_mismatch": []
}
```

**NB02 phải assert** `manifest.embed_model == EMBED_MODEL` và `manifest.visual_model == VISUAL_MODEL` ở cell đầu. Lệch thì **dừng ngay**, đừng chạy tiếp.
