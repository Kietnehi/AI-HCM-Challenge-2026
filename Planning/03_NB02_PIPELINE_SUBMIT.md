# 03 — NB02: END-TO-END PIPELINE & SUBMISSION

**Tên notebook Kaggle:** `aic26-02-pipeline-submit`
**Input datasets (4 cái):**
- `kitnehi1211/aic26-index` — output NB01
- `kitnehi1211/feature-aic-2026` — cần `videos.parquet` fallback + object classes
- `kitnehi1211/dethithunghiem` — **gói đề** (24 file `.txt`, phẳng ở gốc)
- `fatle542/aic-dataset` — **ảnh keyframe**, 115.75 GB (QĐ-4)

**Path:** copy `PATHS` block + `resolve()` + `build_kf_index()` từ [`05_KAGGLE_PATHS.md §2–4`](./05_KAGGLE_PATHS.md)
**Output:** `/kaggle/working/team_XXX_roundN.zip`
**Accelerator:** `GPU T4 x2` (cho `RERANK_MODE="local"`) · `None` nếu dùng `"openrouter_llm"`
**Internet:** **BẮT BUỘC ON**
**Thời gian dự kiến:** 1–3h / gói 24 query

---

## Cấu trúc cell

| Cell | Tên | Nội dung |
|:-:|:--|:--|
| 1 | `SETUP` | pip, import, CONFIG |
| 2 | `PREFLIGHT` | verify manifest + path ảnh + model OpenRouter khả dụng |
| 3 | `LOAD_INDEX` | load parquet + FAISS + BM25 vào RAM |
| 4 | `STAGE0` | Query understanding → `parsed_queries.json` |
| 5 | `STAGE0_REVIEW` | **in ra để người đọc bằng mắt** |
| 6 | `RETRIEVERS` | hàm search từng channel |
| 7 | `STAGE1_2` | retrieval + RRF fusion → `candidates_*.parquet` |
| 8 | `STAGE3` | text rerank (Qwen3-Reranker) |
| 9 | `STAGE4` | VLM rerank (MiMo-V2.5 + ảnh) |
| 10 | `KIS` | sinh rows cho KIS |
| 11 | `QA` | sinh rows cho QA |
| 12 | `TRAKE` | DP alignment → sinh rows cho TRAKE |
| 13 | `WRITER` | ghi CSV |
| 14 | `VALIDATOR` | 12 check bắt buộc |
| 15 | `ZIP` | đóng gói `submission/` → `.zip` |

---

## 1. Cell 2 — `PREFLIGHT` (đừng bỏ qua)

Ba thứ chặn cả pipeline nếu sai, kiểm hết ở đây với chi phí ~30 giây:

### 1.1 Manifest guard (QĐ-2)
```python
mf = json.load(open(f"{INDEX_ROOT}/BUILD_MANIFEST.json"))
assert mf["embed_model"]  == EMBED_MODEL,  f"index build bằng {mf['embed_model']}"
assert mf["visual_model"] == VISUAL_MODEL, f"index build bằng {mf['visual_model']}"
EMBED_DIM  = mf["embed_dim"]      # lấy từ manifest, không tin CONFIG
VISUAL_DIM = mf["visual_dim"]
TOK_VI     = mf.get("tokenizer_vi", "pyvi")
```

### 1.2 Keyframe path discovery (QĐ-4, R2)

Pattern **đã verify**: `Keyframes_L21/keyframes/L21_V001/001.jpg`. Dùng `build_kf_index()` ở `05_KAGGLE_PATHS.md §4`:

```python
KF_DIR = build_kf_index(KEYFRAME_DS)          # video_id -> dir, cache ra WORK
assert len(KF_DIR) == 873, f"chi thay {len(KF_DIR)}/873 video co anh"

import random
from PIL import Image
for kfid in random.sample(list(kf.kf_id), 20):
    vid, n = kfid.split("#")
    Image.open(kf_image_path(vid, int(n))).verify()   # phai mo duoc that
```

Nếu `len(KF_DIR) < 873`: in `sorted(set(kf.video_id) - set(KF_DIR))`. Những video đó **không VLM-rerank được**.

Nếu không tìm được ảnh nào: **vẫn chạy được** Stage 0–3 và KIS/TRAKE (kém hơn), nhưng QA gần như chắc chắn sai. Set `SKIP_VLM = True` và **báo rõ trong output**, đừng chạy tiếp im lặng.

> Thư mục batch cho L26 (498 video) **chưa verify** — có thể là `Keyframes_L26_a…e`. `build_kf_index()` glob ở mức `*/keyframes/*` nên tự xử lý mọi cách đặt tên. **Đừng** glob xuống tới `.jpg`: 177K file trên dataset 115GB sẽ treo rất lâu.

### 1.3 Model availability (R1)
```python
models = requests.get(f"{OPENROUTER_BASE_URL}/models",
                      headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}).json()
ids = {m["id"] for m in models["data"]}
if VLM_MODEL not in ids:
    print(f"[WARN] {VLM_MODEL} không có. Ứng viên vision thay thế:")
    print([i for i in sorted(ids) if any(k in i for k in ("vl","vision","gemini","mimo"))])
    raise SystemExit("Chọn lại VLM_MODEL rồi chạy lại")
```
Cũng gửi 1 request thử **có ảnh** để chắc model thật sự nhận vision input (nhiều model text-only vẫn trả 200 rồi ignore ảnh).

---

## 2. Cell 3 — `LOAD_INDEX`

Load hết vào RAM (Kaggle có 30GB, index ~2.5GB nên thoải mái):
```python
kf     = pd.read_parquet(f"{INDEX_ROOT}/keyframes.parquet")
videos = pd.read_parquet(f"{INDEX_ROOT}/videos.parquet")
units  = pd.read_parquet(f"{INDEX_ROOT}/text_units.parquet")
KF2ROW = dict(zip(kf.kf_id, kf.index))          # tra cứu O(1)
KF2FIDX= dict(zip(kf.kf_id, kf.frame_idx))
FAISS  = {c: faiss.read_index(...) for c in ["vision","caption","ocr","asr","summary","meta"]}
ROWMAP = {c: np.load(...) for c in ...}
BM25   = {name: bm25s.BM25.load(...) for name in [...]}
OBJ_M  = scipy.sparse.load_npz(...)             # [177321, 584]
OBJ_CLS= {c:i for i,c in enumerate(load_object_classes(...))}  # 584 muc; xem 01_DATA_CONTRACTS §4
                                                # KHONG enumerate(splitlines()) tho: file goc con
                                                # 2 dong comment + 1 dong trong -> lech index 3 cot
clip_txt = SentenceTransformer(VISUAL_MODEL)    # text tower cho kênh visual
```

---

## 3. Cell 4 — `STAGE0`: Query Understanding

Đọc đề bằng `load_queries(QUERY_ROOT)` (`05_KAGGLE_PATHS.md §5`) — hàm này đã tự lấy `type` từ hậu tố tên file và đếm `n_events` bằng regex.

> Gói đề thử có **24 file**, `query-p1-3` **không tồn tại**. Luôn `glob`, đừng `range(1, 26)`.

Sau đó một call MiMo-V2.5 / query. Output đúng schema `01_DATA_CONTRACTS.md §7`.

### System prompt (dùng nguyên văn, đây là phần quyết định chất lượng cả pipeline)

```
You are a query analyst for a Vietnamese TV-news video retrieval system
(873 videos, 177k keyframes; sources: CLIP visual features, English image captions,
Vietnamese on-screen OCR, Vietnamese speech transcripts + English translations,
English video summaries, and Vietnamese YouTube metadata).

Given a Vietnamese query, output ONE JSON object. No markdown, no prose.

{
  "type": "kis" | "qa" | "trake",
  "q_en": "faithful English translation of the whole query",
  "visual_desc_en": "ONLY the purely visual, camera-observable content. <= 40 words.
                     No questions, no reasoning, no 'the video shows'. Just the scene.",
  "keywords_vi": ["..."],   // 3-8 distinctive Vietnamese terms, as they would literally appear
  "keywords_en": ["..."],   // 3-8 English equivalents
  "ocr_hints":   ["..."],   // EXACT strings likely rendered on screen: brand names,
                            // club names, banners, chyrons, jersey numbers, dates, place signs.
                            // Empty list if none. Keep original casing/diacritics.
  "named_entities": ["..."],// people, organisations, provinces, landmarks
  "object_classes": ["..."],// ONLY from the allowed list provided in the user message
  "question_en": "..." | null,   // qa only: the actual question being asked
  "events": [                    // trake only, one per E1..EN, in order
    {"idx": 1, "desc_vi": "...", "desc_en": "...", "visual_desc_en": "<= 25 words"}
  ]
}

Rules:
- "ocr_hints" is the highest-value field. Vietnamese TV news burns titles, names and
  locations into the frame. If the query names a club, program, province or number,
  put the literal Vietnamese string there.
- Do NOT invent entities that are not in the query.
- For "trake", produce exactly one event per LINE matching "E<number>:", **in the order the
  lines appear**. The numbers may be wrong: they can repeat or skip (e.g. E1, E2, E2, E4).
  Never merge, drop, renumber or reorder them - one output event per matching line, line order.
- Keep proper nouns unchanged in q_en (do not anglicise Vietnamese names/places).
```

User message: `f"ALLOWED_OBJECT_CLASSES:\n{', '.join(sorted(OBJ_CLS))}\n\nQUERY:\n{q_vi}"`

Gọi với `temperature=0`, `response_format={"type":"json_object"}` nếu model hỗ trợ.

### Post-processing bắt buộc (đừng tin LLM)

```python
# 1. type từ TÊN FILE, không từ LLM
qtype = query_file.rsplit("-",1)[-1].replace(".txt","")     # kis|qa|trake

# 2. n_events đếm bằng regex, KHÔNG tin LLM (xem data contracts §7)
n_events = len(re.findall(r"^\s*E\s*(\d+)\s*[:.]", q_vi, flags=re.M))   # dem DONG, khong dedupe
# CANH BAO: query-p1-18-trake.txt cua BTC danh may sai -> cac dong la E1, E2, E2, E4.
#   Dem theo dong  -> 4 (dung).  set()/dedupe theo con so -> 3 (SAI -> 0 diem).
#   Thu tu event LUON la thu tu dong, khong bao gio sort theo con so sau chu E.
if qtype == "trake":
    assert n_events >= 2, f"{query_file}: không parse được event"
    if len(parsed["events"]) != n_events:
        # gọi lại LLM 1 lần với ép buộc n_events, vẫn sai thì fallback:
        # tự split q_vi theo regex và dịch từng dòng riêng
        ...

# 3. object_classes lọc về tập hợp lệ
parsed["object_classes"] = [c for c in parsed["object_classes"] if c in OBJ_CLS]

# 4. visual_desc_en truncate <= 60 từ (giới hạn 77 token của CLIP)
# 5. NFC-normalize mọi field tiếng Việt (khớp tokenizer BM25 -> data contracts §6)
```

## 4. Cell 5 — `STAGE0_REVIEW`

In bảng cho cả 24 query: `query_id | type | n_events | ocr_hints | object_classes | visual_desc_en`.

> **Đây là điểm dừng của con người.** `ocr_hints` sai/thiếu là nguyên nhân miss lớn nhất với news video. Cho phép sửa tay bằng cách ghi `parsed_queries.json` ra `/kaggle/working`, sửa, rồi load lại (`RELOAD_PARSED = True`).

---

## 5. Cell 6–7 — Retrieval & Fusion

### 5.1 Các channel và input tương ứng

| Channel | Index | Query input | Trả về |
|:--|:--|:--|:--|
| `vision` | `vision.faiss` (512-d) | `clip_txt.encode(visual_desc_en)` | kf-level |
| `caption` | `text_caption.faiss` | `embed(q_en)` | kf-level |
| `ocr` | `text_ocr.faiss` | `embed(q_vi)` ← **VI**, vì corpus OCR là VI | kf-level |
| `asr` | `text_asr.faiss` | `embed(q_en)` | unit → spread ra kf |
| `summary` | `text_summary.faiss` | `embed(q_en)` | → **video prior** |
| `meta` | `text_meta.faiss` | `embed(q_en)` | → **video prior** |
| `bm25_ocr_vi` | BM25 | `tok_vi(q_vi + " " + " ".join(ocr_hints)*3)` | kf-level |
| `bm25_caption_en` | BM25 | `tok_en(q_en)` | kf-level |
| `bm25_asr_vi` | BM25 | `tok_vi(q_vi)` | unit → kf |
| `bm25_asr_en` | BM25 | `tok_en(q_en)` | unit → kf |
| `bm25_summary_en` | BM25 | `tok_en(q_en)` | → video prior |
| `bm25_meta` | BM25 | `tok_vi(q_vi)` | → video prior |

> **`ocr_hints` nhân 3 lần** trong query BM25 là thủ thuật đơn giản để tăng trọng số term chính xác — với news video nó thường là tín hiệu mạnh nhất. Ví dụ `query-p1-15-qa` có "FANA": exact match trên OCR gần như định vị được ngay video.

### 5.2 ASR/summary → keyframe (spread)

Unit `asr` phủ một khoảng thời gian, không phải 1 frame:
```python
# với unit asr có (video_id, t_start, t_end) và rank r:
# mọi keyframe của video đó có t_start - 2 <= pts_time <= t_end + 2
#   nhận rank r (chia sẻ cùng rank, không giảm dần)
```
Padding ±2s để bù lệch giữa lời nói và hình ảnh minh họa.

### 5.3 Video prior

`summary` / `meta` là video-level nên **không sinh candidate**, chỉ boost:
```python
video_prior = {}   # video_id -> [0,1]
for ch, w in [("summary", W["summary_prior"]), ("meta", W["meta_prior"]),
              ("bm25_summary_en", ...), ("bm25_meta", ...)]:
    for rank, vid in enumerate(top_videos_from(ch)):
        video_prior[vid] += w / (RRF_K + rank + 1)
# min-max normalize về [0,1]
```

### 5.4 Object bonus

```python
if parsed["object_classes"]:
    cols = [OBJ_CLS[c] for c in parsed["object_classes"]]
    hit  = np.asarray(OBJ_M[:, cols].sum(axis=1)).ravel()   # vectorized, không loop
    frac = hit / len(cols)                                   # tỉ lệ class khớp
    object_bonus = W["object_bonus"] * frac                   # SOFT bonus
```

> **Dùng SOFT bonus, KHÔNG hard filter.** Object detector 584-class có recall hạn chế; hard-filter `Dragon` sẽ loại sạch video múa lân nếu detector miss con rồng. Chỉ hard-filter khi query cực rõ **và** class đó phổ biến (`Person`, `Car`) — mà khi đó filter cũng gần vô dụng. Kết luận: luôn soft.

### 5.5 Fusion (RRF)

```python
score[kf_id] = sum(W[ch] / (RRF_K + rank_ch[kf_id] + 1) for ch in channels_hit) \
             + video_prior[video_of(kf_id)] \
             + object_bonus[vis_row_of(kf_id)]
```

**Vì sao RRF (rank-based) chứ không phải weighted-sum của score thô:** score cosine của các channel khác nhau không cùng thang, và **173 video thiếu OCR** (R5) sẽ bị trừng phạt bất công nếu cộng score thô (OCR score = 0). RRF chỉ dùng thứ hạng trong từng channel nên channel vắng mặt đơn giản là không góp điểm, không kéo xuống.

Lấy `TOPK_FUSED = 1000` kf → ghi `candidates_<query_id>.parquet`.

---

## 6. Cell 8–12 — Rerank & per-type output

### 6.1 Stage 3 — Text rerank (1000 → 100)

**Evidence card** cho mỗi kf (đây là "document" đưa vào reranker):
```
[VIDEO] {title}
[SUMMARY] {summary_en[:300]}
[FRAME t={pts_time:.1f}s] {caption}
[ON-SCREEN TEXT] {ocr_text}
[SPEECH] {asr window text_en bao quanh pts_time}
[OBJECTS] {", ".join(classes[:12])}
```

- `RERANK_MODE="local"`: `Qwen/Qwen3-Reranker-4B`, cặp `(q_en + " || " + q_vi, evidence_card)`, batch 16, fp16, truncate 1024 token.
- `RERANK_MODE="openrouter_llm"`: listwise, 20 candidate/call, prompt trả về `[{"id":..,"score":0-10}]`.

Kết hợp: `final = 0.4 * norm(fused_score) + 0.6 * norm(rerank_text_score)` — giữ lại tín hiệu fusion để reranker không "quên" bằng chứng đa kênh.

### 6.2 Stage 4 — VLM rerank (100 → 20)

Gửi MiMo-V2.5: **ảnh keyframe** (resize long side ≤ 768, JPEG q=85, base64) + evidence card + query.

```
You are judging whether ONE video keyframe satisfies a retrieval query.

QUERY (Vietnamese): {q_vi}
QUERY (English):    {q_en}

Return JSON only:
{"score": 0-10, "reason": "<= 25 words", "answer": "<see below>"}

score: 10 = this exact frame is the answer; 7-9 = right scene/moment, maybe off by
a second; 4-6 = right video, wrong moment; 1-3 = related topic only; 0 = unrelated.
answer: only for Q&A queries -> the answer to "{question_en}", else null.
        DEFAULT TO VIETNAMESE, VERBATIM. If the answer is text visible in the frame
        (a place name, club name, banner, recipe title, a line of poetry) or spoken by
        someone, copy it EXACTLY as it appears - same wording, same diacritics, no
        translation, no paraphrase, no added words. Only answer in English when the
        question is purely descriptive (a count, a colour) AND no Vietnamese string on
        screen expresses it. Max 100 characters.
```

**Batching:** 1 ảnh / call (nhiều ảnh cùng call làm model lẫn frame). 100 kf x 24 query = 2,400 call. **Quá đắt** → chỉ VLM-rerank top-**20** sau Stage 3, và bỏ qua kf trùng: nếu 2 kf cùng video cách nhau < 2s thì chỉ gửi 1 cái đại diện.

### 6.3 KIS — Cell 10

```python
rows = []            # tối đa 100
# sort theo (rerank_vlm_score, rerank_text_score, fused_score) giảm dần
# DIVERSITY: mỗi video tối đa 5 frame trong 30 dòng đầu -> tránh dồn hết vào 1 video sai
# 30 dòng sau: nới lên 10 frame/video
# TEMPORAL PADDING: với mỗi kf top-10, thêm kf lân cận (n-1, n+1) làm hàng dự phòng
#   vì "khoảnh khắc đầu tiên" rất dễ lệch 1 keyframe
rows.append((video_id, frame_idx))
```

> Được 100 dòng thì **dùng hết 100**. Không có penalty cho dòng sai, chỉ có phần thưởng cho dòng đúng.

### 6.4 QA — Cell 11

1. Lấy top-20 sau Stage 4, mỗi cái có `answer` từ VLM.
2. **Gom answer theo cụm ngữ nghĩa** (normalize lowercase + strip dấu để so), đếm vote. Answer nhiều vote nhất và đến từ frame điểm cao nhất = answer chính.
3. Ràng buộc output (QĐ-5):

| Loại câu hỏi | Ngôn ngữ answer | Ví dụ từ bộ đề |
|:--|:--|:--|
| **Mặc định** — chữ đọc được trên khung hình hoặc lời nói | **Verbatim tiếng Việt** | cả 3/3 query QA của bộ đề đều rơi vào đây |
| Tên riêng (xã/tỉnh/người/tổ chức) | **Verbatim tiếng Việt** | `query-p1-15-qa` → tên xã ở Khánh Hòa |
| Trích dẫn / câu thơ | **Verbatim tiếng Việt** | `query-p1-19-qa` → 2 câu thơ về Nguyễn Trung Trực |
| Tiêu đề / nhãn in trên vật thể | **Verbatim tiếng Việt** | `query-p1-22-qa` → tiêu đề công thức món ăn |
| Số lượng / màu sắc thuần mô tả, **không** có chuỗi VI tương ứng | English, ngắn | `"Five people"`, `"Red"` — **0/3** query quan sát được thuộc loại này |

> **Vì sao đảo mặc định sang tiếng Việt** (đổi so với bản plan cũ): thể lệ nói mâu thuẫn — phần đầu ghi *"so sánh chính xác **về mặt ngữ nghĩa**"*, mục "Lưu ý quan trọng" lại ghi *"Answer (Q&A) sẽ được so sánh dưới dạng **chuỗi chính xác**"*. Bản verbatim tiếng Việt đúng dưới **cả hai** cách chấm; bản dịch tiếng Anh chỉ đúng dưới cách chấm ngữ nghĩa. Chọn phương án không thua ở kịch bản xấu.

> **Hedge bằng số dòng.** Có 100 dòng thì không phải chọn một: các dòng đầu = answer tiếng Việt verbatim (nhiều `frame_idx` khác nhau), các dòng sau = **cùng frame, answer dịch tiếng Anh**. Không có penalty cho dòng sai.

4. **Hard truncate ≤ 100 ký tự.** Nếu câu thơ dài hơn 100 ký tự → cắt ở ranh giới từ và ưu tiên giữ phần đặc trưng nhất. Log warning để review tay.
5. Rows: dòng 1–N = answer chính với các frame_idx khác nhau (cùng answer, khác frame → tăng cơ hội trúng frame). Dòng tiếp = answer hạng 2. Đa dạng hóa cả frame **và** answer.

### 6.5 TRAKE — Cell 12 (QĐ-6)

```
Bước 1 — Video-level retrieval
  Query = visual_desc_en tổng + hợp của mọi events[].desc_en
  Chạy full pipeline Stage 1-2, rồi aggregate về video:
     video_score = sum of top-10 kf scores của video đó
  Lấy top M = 5 video.

Bước 2 — Score matrix / video
  Với mỗi video v (N keyframe), mỗi event e_k (k = 1..K):
     S[i, k] = a * cos(clip_txt(e_k.visual_desc_en), vision_vec[i])
             + b * cos(embed(e_k.desc_en),  caption_vec[i])
             + c * bm25(tok_vi(e_k.desc_vi), ocr_unit[i])
     (a, b, c) = (0.5, 0.35, 0.15); min-max normalize từng cột k

Bước 3 — DP monotonic alignment  (đây là phần cốt lõi)
  Tìm i_1 < i_2 < ... < i_K maximize sum_k S[i_k, k]
     dp[i][k] = S[i,k] + max(dp[j][k-1] for j < i)
  -> tiền tố max chạy dần nên O(N*K), N<=~600, K<=~6 => tức thời.
  Traceback ra đường đi tốt nhất.

Bước 4 — Beam để có nhiều hypothesis
  Giữ top-B = 5 đường đi khác nhau / video (beam search trên cùng DP,
  hoặc chạy lại DP sau khi cấm i_1 đã chọn).
  => 5 video x 5 path = 25 hàng. Nới B để tiến gần 100 hàng.

Bước 5 — VLM verify (tùy chọn, chỉ top-5 path)
  Gửi K ảnh của path + K mô tả event, hỏi "các frame này có khớp
  đúng thứ tự events không?" -> score 0-10, dùng để sort lại hàng.

Bước 6 — Emit
  row = (video_id, frame_idx[i_1], ..., frame_idx[i_K])
  ASSERT len == K == n_events  và  frame_idx tăng dần nghiêm ngặt.
```

> **DP là bắt buộc, không phải tối ưu hóa.** Lấy `argmax` từng event độc lập sẽ cho ra frame **không** theo thứ tự thời gian (event 3 trước event 1) → sai format → 0 điểm.

---

## 7. Cell 13 — `WRITER`

```python
import csv
Path(f"{WORK}/submission").mkdir(exist_ok=True)
# tên file = tên file query, chỉ đổi .txt -> .csv
#   query-p1-16-trake.txt -> query-p1-16-trake.csv
with open(f"{WORK}/submission/{query_id}.csv", "w",
          newline="", encoding="utf-8") as f:      # newline="" -> để csv tự xử lý
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    for r in rows[:MAX_ROWS_PER_CSV]:
        w.writerow(r)
```

Quy tắc format (theo `TheLeCuocThi/sotuyenAIC.md`):

| Điều | Giá trị |
|:--|:--|
| Header | **KHÔNG** có |
| Encoding | UTF-8 (không BOM) |
| Delimiter | `,` |
| Line ending | `\n` |
| Quoting | `QUOTE_MINIMAL` — module `csv` tự bọc ngoặc kép và escape `""` đúng chuẩn |
| `video_id` | không có `.mp4` |
| `frame_idx` | số nguyên, không khoảng trắng |
| Số dòng | ≤ 100 |

> Dùng module `csv`, **đừng** tự nối chuỗi bằng `",".join()`. Answer tiếng Việt có thể chứa dấu phẩy (`"Có 3 người, gồm nam và nữ"`) và module `csv` xử lý escape đúng còn nối tay thì không.

## 8. Cell 14 — `VALIDATOR` (Definition of Done)

Chạy trên thư mục `submission/` **trước mỗi lần nộp**. Fail bất kỳ check nào thì **không nộp**.

| # | Check |
|:-:|:--|
| 1 | Mọi file `.csv`, không có `.xlsx` / `.xls` |
| 2 | Tên file khớp 1-1 với tên file query đã nhận (`query-p1-N-<type>.csv`) |
| 3 | Đọc lại bằng `csv.reader` không lỗi; encoding UTF-8 hợp lệ |
| 4 | Không có header row (dòng 1 phải parse được thành dữ liệu) |
| 5 | `1 <= n_rows <= 100`, và `n_rows > 0` cho **mọi** query |
| 6 | `video_id` khớp regex `^L\d{2}_V\d{3}$` |
| 7 | `frame_idx` là int, `>= 0`, và **tồn tại thật** trong `keyframes[video_id].frame_idx` |
| 8 | KIS: đúng **2** cột |
| 9 | QA: đúng **3** cột; `len(answer) <= 100`; answer không rỗng |
| 10 | TRAKE: đúng `1 + n_events` cột; frame_idx **tăng dần nghiêm ngặt**; cùng 1 `video_id` |
| 11 | Không có dòng trùng lặp hoàn toàn |
| 12 | Không có ký tự `\r` lạc, không có dòng trắng ở cuối gây parse thành row rỗng |

Check #7 đắt nhưng quan trọng nhất: nó bắt được lỗi nộp `n` hoặc `pts_time` thay vì `frame_idx`.

## 9. Cell 15 — `ZIP`

```python
import shutil, os
os.chdir(WORK)
shutil.make_archive("team_XXX_round1", "zip", root_dir=WORK, base_dir="submission")
# -> team_XXX_round1.zip, BÊN TRONG có thư mục submission/
```

Verify lại:
```python
import zipfile
names = zipfile.ZipFile(f"{WORK}/team_XXX_round1.zip").namelist()
assert all(n.startswith("submission/") for n in names), "THIẾU thư mục submission/"
print(*names, sep="\n")
```

> Lỗi phổ biến nhất theo thể lệ: nén trực tiếp các file CSV thay vì nén **thư mục** `submission`. `base_dir="submission"` là chỗ quyết định điều đó.

## 10. Chiến lược 3 lần nộp

| Lần | Cấu hình | Mục tiêu |
|:-:|:--|:--|
| 1 | Pipeline đầy đủ, weight mặc định, đã pass 12 check validator | **Baseline an toàn.** Có điểm trên bảng, biết format đúng |
| 2 | Sau khi xem Public LB + review tay `parsed_queries.json` (sửa `ocr_hints`), tune weight | Cải tiến có định hướng |
| 3 | Bản tốt nhất, review tay answer QA + verify TRAKE ordering | **Lần cuối được tính điểm** — chỉ nộp khi chắc chắn tốt hơn lần 2 |

> Public LB chỉ tính **50%** đáp án; Private tính 100%. Đừng overfit vào Public LB — nếu lần 2 chỉ hơn lần 1 một chút thì đó có thể là nhiễu, không phải cải tiến thật.
