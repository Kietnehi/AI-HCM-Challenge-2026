# 02 — NB01: INDEX BUILDER

**Tên notebook Kaggle:** `aic26-01-index-builder`
**Input datasets:** `kitnehi1211/feature-aic-2026` **duy nhất** (KHÔNG attach `fatle542/aic-dataset` — NB01 không cần ảnh, mount 115GB chỉ làm chậm)
**Path:** copy `PATHS` block + `resolve()` từ [`05_KAGGLE_PATHS.md §2–3`](./05_KAGGLE_PATHS.md)
**Output:** publish `/kaggle/working/index/` → Kaggle Dataset `aic26-index`
**Accelerator:** `None` (CPU) nếu `EMBED_PROVIDER="openai"` · `GPU T4 x2` nếu `EMBED_PROVIDER="local"`
**Internet:** **BẮT BUỘC ON** (gọi API embedding)
**Thời gian dự kiến:** 3–4h

> Chạy **một lần duy nhất**. Mỗi gói đề mới chỉ chạy NB02.

---

## Cấu trúc cell

| Cell | Tên | Nội dung | Thời gian |
|:-:|:--|:--|--:|
| 1 | `SETUP` | pip install, import, CONFIG block | 2m |
| 2 | `PATH_VERIFY` | assert mọi path input tồn tại, đếm file | 1m |
| 3 | `PHASE_A1` | build `keyframes.parquet` | 5m |
| 4 | `PHASE_A2` | build `videos.parquet` | 10m |
| 5 | `PHASE_A3` | build `text_units.parquet` (5 channel) | 25m |
| 6 | `PHASE_A4` | build `objects.parquet` + `objects_matrix.npz` | 30m |
| 7 | `PHASE_E` | build `vision.faiss` từ CLIP `.npy` | 5m |
| 8 | `PHASE_D` | build 6 BM25 index | 20m |
| 9 | `EMBED_CLIENT` | hàm `embed_texts()` (2 provider) + smoke test | 2m |
| 10 | `PHASE_B` | embed ~319K unit, checkpoint resumable | **1.5–3h** |
| 11 | `PHASE_C` | build 5 `text_*.faiss` từ checkpoint | 10m |
| 12 | `MANIFEST` | ghi `BUILD_MANIFEST.json` | 1m |
| 13 | `SELF_TEST` | 8 assertion + 3 truy vấn thử | 5m |

---

## Cell 1 — `SETUP`

```python
!pip install -q bm25s faiss-cpu pyvi PyStemmer pyarrow
# EMBED_PROVIDER="local" thì thêm: !pip install -q sentence-transformers
```

Import + **paste nguyên CONFIG block từ `00_MASTER_PLAN.md §5`** + **PATHS block & `resolve()` từ `05_KAGGLE_PATHS.md §2–3`**.

Thêm helper:
```python
import os, json, glob, re, unicodedata, time, random
from pathlib import Path
os.makedirs(f"{WORK}/index/faiss", exist_ok=True)
os.makedirs(f"{WORK}/index/bm25", exist_ok=True)
os.makedirs(f"{WORK}/ckpt", exist_ok=True)

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()
```

> `nfc()` phải áp dụng cho **mọi** text tiếng Việt khi ghi vào corpus. Xem `01_DATA_CONTRACTS.md §6`.

## Cell 2 — `PATH_VERIFY`

Assert và in ra số lượng, **fail ngay nếu lệch số kỳ vọng**. Dùng các biến `P_*` từ PATHS block:

```python
EXPECT = {
    "clip":    (f"{P_CLIP}/*.npy",              873),
    "mapkf":   (f"{P_MAPKF}/*.csv",             873),
    "caption": (f"{P_CAPTION}/*.json",          873),
    "media":   (f"{P_MEDIA}/*.json",            873),
    "summary": (f"{P_SUMMARY}/*.json",          866),   # 865 that + 1 sentinel _failed.json
    "asr_en":  (f"{P_ASR_EN}/*/video/*.json",   873),
    "asr_vi":  (f"{P_ASR_VI}/*/video/*.json",   873),
    "ocr_dir": (f"{P_OCR_DIR}/*.json",          700),
}
for k, (pat, n) in EXPECT.items():
    got = len(glob.glob(pat))
    assert got == n, f"{k}: thay {got}, ky vong {n}  (pattern: {pat})"

assert sum(1 for _ in open(P_OCR_JSONL, encoding="utf-8")) == 128_664

# detected_classes.txt: 587 dong CRLF = 2 comment + 1 dong trong + 584 class.
# KHONG dung .split("\n") (moi ten se dinh \r) va KHONG enumerate(splitlines()) (giu lai header).
OBJ_CLASSES = [l.strip() for l in open(P_OBJ_CLS, encoding="utf-8")]
OBJ_CLASSES = [l for l in OBJ_CLASSES if l and not l.startswith("#")]
assert len(OBJ_CLASSES) == 584, len(OBJ_CLASSES)

assert len(glob.glob(f"{P_ASR_EN}/*")) == 14                # 14 batch ASR
```

> Path đã verify thật (`05_KAGGLE_PATHS.md`) — chú ý **`FEAT_ROOT` có thêm lớp `Feature_Dataset/`** và mount là dạng `/kaggle/input/datasets/<owner>/<slug>`. Hàm `resolve()` xử lý cả 2 dạng nên không cần hard-code.

**Bốn bất thường ĐÃ VERIFY, đừng debug lại:**
1. `video_id` **không liên tục** — `L21_V004` và `L21_V020` không tồn tại. Đừng sinh id bằng `range()`.
2. **`Summary_video` có 866 file `.json` nhưng chỉ 865 là thật** — file còn lại là sentinel **`_failed.json`**, không phải video. Số video thiếu summary là **8**: `L26_V072`–`L26_V079`. Mọi vòng lặp trên thư mục này phải skip stem bắt đầu bằng `_`, nếu không sẽ sinh ra một "video" tên `_failed` trong `videos.parquet`.
3. **`detected_classes.txt` có header** — 587 dòng CRLF = 2 comment + 1 dòng trống + 584 class. Xem đoạn parse ở cell trên.
4. **`ocr_index.jsonl` đã là 1 dòng / keyframe** — 128,664 dòng = 128,664 cặp `(video_id, keyframe)` unique. Không có gì để gộp.

## Cell 3 — `PHASE_A1`: `keyframes.parquet`

1. Đọc 873 CSV, thêm cột `video_id` từ tên file.
2. `n = df.n.astype("int32")`, `frame_idx = df.frame_idx.astype("int32")`.
3. `kf_id = video_id + "#" + n.map("{:03d}".format)`.
4. `kf_path = video_id + "/" + n.map("{:03d}".format) + ".jpg"`.
5. Sort `(video_id, n)`, `reset_index`, `vis_row = np.arange(len(df), dtype="int32")`.

**Assertion bắt buộc (invariant §1 của data contracts):**
```python
for vid, g in kf.groupby("video_id"):
    n_npy = np.load(f"{P_CLIP}/{vid}.npy", mmap_mode="r").shape[0]
    if len(g) != n_npy:
        mismatch.append((vid, len(g), n_npy))
assert not mismatch, mismatch      # ghi vào manifest, KHÔNG tự sửa
```

Kỳ vọng: **177,321 dòng**.

## Cell 4 — `PHASE_A2`: `videos.parquet`

Đọc `{P_MEDIA}`, `{P_SUMMARY}`, `{P_ASR_EN}` full-text. Áp fallback `summary_en → description → ""`. Set `has_ocr` / `has_summary` / `has_transcript`.

> ASR nằm rải trong **14 thư mục batch** `Videos_L21_a` … `Videos_L30_a`. Dùng `glob(f"{P_ASR_EN}/*/video/*.json")` rồi lập dict `video_id -> path`, đừng giả định thư mục nào chứa video nào.

## Cell 5 — `PHASE_A3`: `text_units.parquet`

Implement **đúng** 5 luật build ở `01_DATA_CONTRACTS.md §3.2`. Viết 5 hàm riêng, mỗi hàm trả về `list[dict]`:

```python
def build_caption_units(kf_df) -> list[dict]: ...
def build_ocr_units(ocr_jsonl_path, kf_df) -> list[dict]: ...
def build_asr_units(kf_df, window=25.0, stride=10.0) -> list[dict]: ...
def build_summary_units(kf_df) -> list[dict]: ...   # skip _failed.json; parse moc [MM:SS-MM:SS] trong chunk_summaries
def build_meta_units(videos_df) -> list[dict]: ...
```

Năm điểm dễ sai nhất:

1. **Caption duplicate:** `duplicate_of` là **tên file** (`"009.jpg"`), không phải `kf_id` — parse ra `n` rồi mới dựng `f"{video_id}#{n:03d}"`. Dòng có `duplicate_of != null` vẫn tạo unit, `text_en` copy từ canonical, `emb_row` sẽ được gán bằng `emb_row` của canonical **ở Cell 11** (không phải bây giờ). Tạm set `emb_row = -2` để đánh dấu "dup, cần patch sau" — phân biệt với `-1` (không embed).

2. **ASR windowing + dedupe:** cửa sổ trượt trên trục **thời gian**, không phải trên số segment. **Bước dedupe là bắt buộc**, không phải tối ưu hoá.
```python
# t = 0, 10, 20, ... ; window = [t, t+25)
# gom mọi segment giao với window; bỏ window rỗng
# t_start = min(seg.start), t_end = max(seg.end) của các segment thực tế lấy được
#
# BAT BUOC: dedupe theo tap segment
#   key = tuple(sorted(seg["id"] for seg in hit))
#   neu key da thay -> bo window nay
```
> Segment ASR ở bộ này dài trung bình **24.2s**, gần bằng đúng cửa sổ 25s, nên stride 10s làm cùng một segment lặp lại ở 2–3 window liên tiếp. Đo trên 40 video: 1,600 window / chỉ 985 tập segment khác nhau (**dư 1.62×**). Không dedupe thì vừa phí ~38% tiền embed ASR, vừa để một đoạn text tự khuếch đại điểm RRF của chính nó vì xuất hiện ở nhiều rank.

3. **ASR → `frame_idx`:** dùng `np.searchsorted` trên `pts_time` đã sort của từng video. **Đừng** dùng vòng lặp Python trên ~21K window × 200 keyframe.

4. **OCR: không gộp gì cả.** `ocr_index.jsonl` đã là 1 dòng / keyframe (128,664 dòng = 128,664 keyframe unique) và đã có sẵn `frame_idx` + `pts_time`. Đọc thẳng, chỉ assert lại giá trị khớp `keyframes.parquet`.

5. **Summary: skip `_failed.json`.** Và mốc thời gian cho TRAKE nằm trong chuỗi `chunk_summaries` (`"- [00:00-02:00] ..."`), **không** phải trong `evidence` — trường đó là dict metadata, không có timestamp.

Kỳ vọng: **~319,000 dòng**. In bảng `value_counts()` theo `channel` và so với ước lượng ở `00_MASTER_PLAN.md §6`; lệch > 25% thì có bug. Số kỳ vọng từng channel: caption ~163,000 · **ocr 128,664** · asr ~21,500 · summary ~5,200 · meta 873.

## Cell 6 — `PHASE_A4`: objects

177,321 file JSON nhỏ → I/O bound nặng. Dùng `concurrent.futures.ThreadPoolExecutor(max_workers=16)`.

```python
# detection_scores la CHUOI ("0.79673874") -> BAT BUOC float() truoc khi so sanh
# giữ class có float(score) >= 0.30, dedupe giữ max score
# -> objects.parquet (kf_id, classes:list, scores:list)
# -> objects_matrix.npz : scipy.sparse.csr_matrix bool [177321, 584]
#    hàng i = vis_row, cột j = index trong detected_classes.txt
```

Ghi `OBJ_CLASSES` (đã parse ở Cell 2, 584 dòng sạch) → `index/object_classes.txt`. **Đừng copy nguyên `{P_OBJ_CLS}`** — file gốc còn 2 dòng comment + 1 dòng trống + CRLF, NB02 đọc vào sẽ lệch index cột 3 đơn vị.

## Cell 7 — `PHASE_E`: `vision.faiss`

```python
import faiss
mats = []
for vid in sorted(video_ids):                       # THỨ TỰ PHẢI KHỚP kf_df
    a = np.load(f"{P_CLIP}/{vid}.npy").astype("float32") # fp16 -> fp32 BẮT BUỘC
    mats.append(a)
X = np.vstack(mats)
assert X.shape == (len(kf_df), VISUAL_DIM)          # (177321, 512)
faiss.normalize_L2(X)                               # BẮT BUỘC cho cosine
idx = faiss.IndexFlatIP(VISUAL_DIM); idx.add(X)
faiss.write_index(idx, f"{WORK}/index/faiss/vision.faiss")
np.save(f"{WORK}/index/faiss/vision_rowmap.npy", kf_df.kf_id.values)
```

> Thứ tự `vstack` phải **y hệt** thứ tự sort của `kf_df` (`sorted(video_id)`, rồi `n` tăng dần). Lệch 1 video là toàn bộ kênh visual sai mà không hề báo lỗi.

## Cell 8 — `PHASE_D`: BM25

```python
import bm25s
from pyvi import ViTokenizer

def tok_vi(s):  return ViTokenizer.tokenize(nfc(s).lower()).split()
def tok_en(s):  return re.sub(r"[^\w\s]", " ", (s or "").lower()).split()
```

Build 6 index theo bảng `01_DATA_CONTRACTS.md §6`. Mỗi index: `bm25s.BM25().index(tokenized)` + `retriever.save(dir)` + `np.save(dir/ids.npy, unit_ids)`.

> `pyvi` đôi khi lỗi trên Kaggle. Bọc `try/except` và fallback về `tok_en`, nhưng **ghi vào manifest** đã dùng tokenizer nào — vì NB02 phải dùng **cùng** tokenizer cho query.

## Cell 9 — `EMBED_CLIENT`

Một interface, hai backend (QĐ-1):

```python
def _embed_openai(texts: list[str]) -> np.ndarray:
    # POST {EMBED_BASE_URL}/embeddings
    # body: {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"}
    # retry 429/5xx: exponential backoff 2**k + jitter, max EMBED_MAX_RETRY
    # LƯU Ý: giữ nguyên thứ tự -> sort theo response["data"][i]["index"]

def _embed_local(texts: list[str]) -> np.ndarray:
    # SentenceTransformer(LOCAL_EMBED_MODEL, device="cuda")
    # .encode(texts, normalize_embeddings=True, batch_size=64)

def embed_texts(texts: list[str]) -> np.ndarray:   # -> float32 [len(texts), EMBED_DIM]
    ...
```

**Smoke test bắt buộc** trước khi tiêu $:
```python
v = embed_texts(["a red car on the street", "một chiếc xe đỏ trên phố"])
assert v.shape == (2, EMBED_DIM) and np.isfinite(v).all()
print("cos(en,vi) =", float(v[0] @ v[1] / (np.linalg.norm(v[0])*np.linalg.norm(v[1]))))
# kỳ vọng > 0.5 -> model hiểu được tiếng Việt, kênh OCR embed trực tiếp VI là hợp lý
```

Nếu cos < 0.35 → `text-embedding-3-small` không align VI/EN tốt → **đổi sang `EMBED_PROVIDER="local"` với `bge-m3`** và ghi lại quyết định. Đây là kiểm tra rẻ nhưng quyết định cả chất lượng kênh OCR.

## Cell 10 — `PHASE_B`: embed toàn corpus (bước dài nhất)

**Yêu cầu resumable — đây là điều kiện sống còn vì Kaggle timeout 12h.**

```python
# Với mỗi channel:
#   texts = text_units[(channel==c) & (emb_row != -2)].text_embed
#   chia thành batch EMBED_BATCH=256
#   với mỗi batch b: nếu WORK/ckpt/{c}/{b:05d}.npy tồn tại -> SKIP
#                    ngược lại -> embed, lưu .npy (float32), lưu kèm ids .npy
#   ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY=8)
#   in tiến độ + token/USD đã tiêu mỗi 50 batch
```

Checkpoint layout:
```
WORK/ckpt/caption/00000.npy , 00000.ids.npy , 00001.npy , ...
WORK/ckpt/ocr/...   asr/...   summary/...   meta/...
```

> **Nếu session bị kill:** commit notebook để giữ `/kaggle/working`, hoặc publish `WORK/ckpt` thành dataset tạm rồi attach lại. Chạy lại cell là tự skip phần đã xong.

**Thứ tự chạy channel:** `meta` (873) → `summary` (5K) → `asr` (44K) → `ocr` (85K) → `caption` (163K). Từ nhỏ đến lớn để phát hiện lỗi sớm với chi phí thấp nhất.

## Cell 11 — `PHASE_C`: build `text_*.faiss`

Cho mỗi channel:
1. `np.vstack` các checkpoint **theo thứ tự batch index tăng dần** (`sorted(glob)` — chú ý zero-pad 5 chữ số nên sort chuỗi là đúng).
2. Assert số hàng == số unit dự kiến, và `ids` khớp từng phần tử với `text_units`.
3. `faiss.normalize_L2` → `IndexFlatIP` → `write_index`.
4. Gán `emb_row` vào `text_units`.
5. **Patch caption duplicate:** với dòng `emb_row == -2`, set `emb_row = emb_row của kf_id canonical`. ⚠️ `duplicate_of` là **tên file** (`"009.jpg"`) chứ không phải `kf_id` — phải `int(Path(duplicate_of).stem)` rồi dựng `f"{video_id}#{n:03d}"` mới join được. Nếu canonical cũng không có → `-1`.
   *(Đã verify trên `L21_V001`: 28/28 caption của keyframe dup **giống hệt** canonical → không mất thông tin khi bỏ embed.)*
6. Ghi lại `text_units.parquet` (đã có `emb_row` đầy đủ).

## Cell 12 — `MANIFEST`

Ghi `BUILD_MANIFEST.json` theo schema `01_DATA_CONTRACTS.md §9`. Điền thật các list `videos_missing_ocr`, `videos_missing_summary`, `videos_keyframe_count_mismatch`, và `tokenizer_vi` đã dùng.

## Cell 13 — `SELF_TEST` (Definition of Done)

| # | Assertion |
|:-:|:--|
| 1 | `len(keyframes) == 177321` và `kf_id` unique |
| 2 | `keyframes.frame_idx.dtype == int32`, không có NaN |
| 3 | `vision.faiss.ntotal == len(keyframes)` |
| 4 | Với mỗi channel: `text_X.faiss.ntotal == (text_units.channel==X & emb_row>=0).nunique(emb_row)` |
| 5 | Mọi `emb_row >= 0` trong `text_units` đều `< faiss.ntotal` của channel đó |
| 6 | 6 thư mục BM25 tồn tại, mỗi cái load được và `len(ids) == n_docs` |
| 7 | `objects_matrix.shape == (177321, 584)` |
| 8 | Toàn bộ `index/` < 20 GB (giới hạn dataset an toàn) |
| 9 | `len(object_classes.txt) == 584`, không dòng nào bắt đầu bằng `#` hoặc chứa `\r` |
| 10 | `videos.parquet` **không** có `video_id == "_failed"`; `has_summary.sum() == 865` |
| 11 | `text_units[channel=="ocr"]` có đúng **128,664** dòng |
| 12 | Không có 2 unit `asr` nào cùng `(video_id, text_en)` — chứng minh dedupe window đã chạy |

**3 truy vấn thử (sanity, phải nhìn kết quả bằng mắt):**

| Query | Channel | Kỳ vọng |
|:--|:--|:--|
| `"FANA charity club giving gifts"` | `bm25_ocr_vi` với `q_vi="FANA"` | Phải ra frame có chữ FANA — đây chính là `query-p1-15-qa` |
| `"lion dance on poles"` | `vision.faiss` (CLIP text) | Top-10 phải là ảnh múa lân |
| `"Nguyễn Trung Trực"` | `bm25_asr_vi` + `bm25_meta` | Phải ra video về đình thần Nguyễn Trung Trực — chính là `query-p1-19-qa` |

> Nếu 3 truy vấn này fail thì **đừng chạy NB02**. Sửa index trước. Ba query này chọn có chủ đích: chúng đến từ bộ đề thật và test đúng 3 kênh khác nhau.

## Publish dataset

`Save & Version` → `Data` tab → `New Dataset` từ `/kaggle/working/index`, tên **`aic26-index`**, private. Version sau thì `New Version` cùng dataset để NB02 không phải sửa path.
