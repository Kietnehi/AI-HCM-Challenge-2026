# 05 — KAGGLE PATHS (đã verify qua Kaggle API, 2026-08-20)

> **Đây là nguồn sự thật duy nhất về path.** Mọi notebook copy `PATHS` block ở §2. Không hard-code path ở nơi khác.

## 1. Ba dataset cần attach

| Dataset | Slug | Path mount | Size | Dùng ở |
|:--|:--|:--|--:|:--|
| **Feature** | `kitnehi1211/feature-aic-2026` | `/kaggle/input/datasets/kitnehi1211/feature-aic-2026/Feature_Dataset` | ~1 GB | NB01, NB02 |
| **Đề thi thử** | `kitnehi1211/dethithunghiem` | `/kaggle/input/datasets/kitnehi1211/dethithunghiem` | ~9 KB | NB02 |
| **Keyframe ảnh** | `fatle542/aic-dataset` | `/kaggle/input/datasets/fatle542/aic-dataset` | **115.75 GB** | NB02 (Stage 4 VLM) |
| **Index (output NB01)** | `kitnehi1211/aic26-index` | `/kaggle/input/datasets/kitnehi1211/aic26-index` | ~2.5 GB | NB02 |

### Hai dạng mount — resolver phải thử cả hai

Kaggle mount dataset theo 1 trong 2 dạng tuỳ cách attach:

```
/kaggle/input/<slug>                        # dạng cổ điển
/kaggle/input/datasets/<owner>/<slug>       # dạng owner-qualified (user đang dùng)
```

Bằng chứng cho dạng thứ hai: notebook `ntloi131205/aic2026-siglip-embedding` dùng `/kaggle/input/datasets/dtdat1725/aic-hcmc-2025-batch-1/video/L21_a`.

→ **Luôn dùng hàm `resolve()` ở §3**, đừng hard-code 1 dạng rồi debug 30 phút vì `FileNotFoundError`.

## 2. PATHS block (copy vào cell SETUP của mọi notebook)

```python
# ============================================================
#  PATHS - da verify qua Kaggle API 2026-08-20 (xem Planning/05_KAGGLE_PATHS.md)
# ============================================================
KG = "/kaggle/input"

# --- Feature dataset: CO them 1 lop "Feature_Dataset/" ben trong ---
FEAT_ROOT  = f"{KG}/datasets/kitnehi1211/feature-aic-2026/Feature_Dataset"

# --- De thi: cac file .txt nam PHANG ngay goc, khong co subfolder ---
QUERY_ROOT = f"{KG}/datasets/kitnehi1211/dethithunghiem"

# --- Anh keyframe: pattern Keyframes_<batch>/keyframes/<video_id>/<nnn>.jpg ---
KEYFRAME_DS = f"{KG}/datasets/fatle542/aic-dataset"

# --- Index do NB01 sinh ra ---
INDEX_ROOT = f"{KG}/datasets/kitnehi1211/aic26-index"

WORK = "/kaggle/working"

# ---- Sub-path cua FEAT_ROOT (da verify ton tai) ----
P_CLIP     = f"{FEAT_ROOT}/clip-features-32-aic25-b1/clip-features-32"   # <vid>.npy      873
P_MAPKF    = f"{FEAT_ROOT}/map-keyframes-aic25-b1/map-keyframes"         # <vid>.csv      873
P_CAPTION  = f"{FEAT_ROOT}/Image_captioning"                             # <vid>.json     873
P_MEDIA    = f"{FEAT_ROOT}/media-info-aic25-b1/media-info"               # <vid>.json     873
P_SUMMARY  = f"{FEAT_ROOT}/Summary_video"                                # 866 json = 865 that + _failed.json
P_ASR_VI   = f"{FEAT_ROOT}/Transcript_Extract"                           # */video/*.json 873
P_ASR_EN   = f"{FEAT_ROOT}/Transcript_Translated"                        # */video/*.json 873
P_OCR_DIR  = f"{FEAT_ROOT}/OCR_EasyOCR_VietOCR"                          # <vid>.json     700
P_OCR_JSONL= f"{FEAT_ROOT}/ocr_index.jsonl"                              # 128,664 dong = 1 dong/keyframe
P_OBJ_DIR  = f"{FEAT_ROOT}/objects-aic25-b1/objects"                     # <vid>/<nnn>.json
P_OBJ_CLS  = f"{FEAT_ROOT}/objects-aic25-b1/detected_classes.txt"        # 587 dong CRLF -> 584 class
```

## 3. Resolver — chống sai dạng mount (dán ngay sau PATHS block)

```python
import os, glob

def resolve(owner: str, slug: str, inner: str = "") -> str:
    """Tra ve path mount thuc te cua dataset, thu ca 2 dang."""
    for base in (f"{KG}/datasets/{owner}/{slug}", f"{KG}/{slug}"):
        p = os.path.join(base, inner) if inner else base
        if os.path.exists(p):
            return p
    # dang 3: tu do bang glob (truong hop slug bi Kaggle doi ten)
    hits = glob.glob(f"{KG}/**/{slug}", recursive=True)
    if hits:
        p = os.path.join(hits[0], inner) if inner else hits[0]
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Khong tim thay {owner}/{slug}. Da attach dataset chua?\n"
        f"Co san trong /kaggle/input: {os.listdir(KG) if os.path.isdir(KG) else 'KHONG CO'}"
    )

FEAT_ROOT   = resolve("kitnehi1211", "feature-aic-2026", "Feature_Dataset")
QUERY_ROOT  = resolve("kitnehi1211", "dethithunghiem")
KEYFRAME_DS = resolve("fatle542",    "aic-dataset")
# INDEX_ROOT = resolve("kitnehi1211", "aic26-index")   # chi trong NB02
```

## 4. Ảnh keyframe — pattern & auto-discovery ⚠️

**Đã verify:** `Keyframes_L21/keyframes/L21_V001/001.jpg` (135 KB).
→ Pattern: `<KEYFRAME_DS>/Keyframes_{batch}/keyframes/{video_id}/{n:03d}.jpg`

**CHƯA verify:** tên thư mục batch cho L26. Feature dataset có 14 batch ASR là `Videos_L21_a … Videos_L26_a…e … Videos_L30_a`, nên `Keyframes_L26` có thể là `Keyframes_L26_a` … `Keyframes_L26_e`. **Không đoán — build map một lần rồi cache:**

```python
import glob, os, json

def build_kf_index(keyframe_ds: str, cache=f"{WORK}/kf_dir_map.json") -> dict:
    """video_id -> thu muc chua anh keyframe. Quet 1 lan roi cache."""
    if os.path.exists(cache):
        return json.load(open(cache))
    m = {}
    # chi quet den do sau thu muc video, KHONG glob toi tung file .jpg
    # (115GB / ~177k file -> glob toi file se rat cham)
    for d in glob.glob(f"{keyframe_ds}/*/keyframes/*"):
        if os.path.isdir(d):
            m[os.path.basename(d)] = d
    json.dump(m, open(cache, "w"))
    return m

KF_DIR = build_kf_index(KEYFRAME_DS)

def kf_image_path(video_id: str, n: int) -> str:
    return f"{KF_DIR[video_id]}/{n:03d}.jpg"
```

**DoD cho T0.2:** `len(KF_DIR) == 873`. Nếu ít hơn → in ra `sorted(set(all_video_ids) - set(KF_DIR))` và báo lại; những video đó sẽ không VLM-rerank được.

> **Lưu ý về 115.75 GB:** dataset này gần như chắc chắn chứa cả file `.mp4`, không chỉ ảnh. Mount một dataset lớn với ~177K file nhỏ **rất chậm** khi khởi động notebook. NB02 chỉ cần ~500 ảnh (20 kf x 24 query), nên vẫn chấp nhận được — nhưng **đừng attach nó vào NB01** (NB01 không cần ảnh, chỉ cần `.npy`).
>
> *Tối ưu tuỳ chọn (T4.6):* tạo 1 notebook resize toàn bộ 177K keyframe xuống 384px WEBP q80 (~25 KB/ảnh ≈ **4.4 GB**), publish thành `aic26-keyframes-small`. NB02 mount nhanh hơn nhiều và VLM cũng chỉ cần ≤768px. Chỉ làm sau khi đã có baseline.

## 5. Đề thi — đọc & parse

```python
import glob, os, re, unicodedata

def load_queries(query_root: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(f"{query_root}/**/*.txt", recursive=True)):
        stem  = os.path.basename(p)[:-4]              # "query-p1-16-trake"
        qtype = stem.rsplit("-", 1)[-1].lower()       # kis | qa | trake
        assert qtype in ("kis", "qa", "trake"), f"hau to la: {stem}"
        q_vi  = unicodedata.normalize("NFC", open(p, encoding="utf-8").read().strip())
        out.append({
            "query_id":   stem,
            "query_file": os.path.basename(p),
            "type":       qtype,
            "q_vi":       q_vi,
            # dem event bang regex, KHONG tin LLM (data contracts §7)
            "n_events":   len(re.findall(r"^\s*E\s*(\d+)\s*[:.]", q_vi, flags=re.M)),
        })
    return out
```

**Đã verify cấu trúc `dethithunghiem`:** file `.txt` nằm **phẳng ngay gốc**, không có subfolder. Dùng `recursive=True` vẫn an toàn cho gói đề sau (BTC có thể zip có folder).

**Nội dung gói đề thử (24 file):**

| Type | Số lượng | query_id |
|:--|--:|:--|
| `kis` | **18** | p1-1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 21, 23, 24, 25 |
| `qa` | **3** | p1-15, 19, 22 |
| `trake` | **3** | p1-4, 16, 18 |

> ⚠️ **`query-p1-18-trake.txt` có lỗi đánh máy của BTC:** các dòng event là `E1:`, `E2:`, **`E2:`**, `E4:` — trùng `E2`, thiếu `E3`. Đếm theo **số dòng** ra 4 (đúng); dedupe theo con số ra 3 (sai → 0 điểm). Hàm `load_queries()` ở trên đếm bằng `len(re.findall(...))` nên đã đúng — nhưng mọi code downstream phải giữ **thứ tự dòng**, đừng sort hay đánh khoá theo con số sau chữ `E`.

> `query-p1-3` **không tồn tại** — số thứ tự query có lỗ. Đừng generate tên file bằng `range(1, 26)`; luôn `glob`. Bộ đề thật của BTC cũng sẽ có lỗ tương tự.

## 6. Đặc điểm dữ liệu đã verify — dễ gây bug

### 6.1 `video_id` KHÔNG liên tục
`L21_V004` và `L21_V020` **không tồn tại**. Tuyệt đối không sinh video_id bằng vòng lặp số; luôn lấy từ `glob` hoặc từ `keyframes.parquet`.

### 6.2 Phân bố 873 video theo prefix

| Prefix | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 | Tổng |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Số video | 29 | 31 | 25 | 43 | 88 | **498** | 16 | 24 | 23 | 96 | **873** |

L26 chiếm 57% dataset → khi sample để test, **phải sample stratified theo prefix**, nếu không sẽ chỉ test L26.

### 6.3 OCR coverage chính xác — 700/873

| Prefix | L21 | L22 | L23 | L24 | L25 | L26 | L27 | L28 | L29 | L30 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Có OCR | 29 | 31 | 25 | 43 | **58** | 498 | 16 | **0** | **0** | **0** |
| Tổng | 29 | 31 | 25 | 43 | 88 | 498 | 16 | 24 | 23 | 96 |

**173 video thiếu OCR** = 30 video của L25 + toàn bộ L28 (24), L29 (23), L30 (96).

> Sửa lại mô tả cũ "OCR có đủ L21→L27": **không đúng** — L25 chỉ có 58/88. `has_ocr` phải tính từ `ocr_index.jsonl` thực tế, đừng suy ra từ prefix.
>
> Folder `OCR_EasyOCR_VietOCR/` và `ocr_index.jsonl` khớp nhau **hoàn hảo** (cùng 700 video, sai khác 0) → chỉ cần đọc `ocr_index.jsonl`, bỏ qua 700 file JSON con.
>
> **`ocr_index.jsonl` đã là 1 dòng / keyframe.** Verify: 128,664 dòng ↔ 128,664 cặp `(video_id, keyframe)` unique, max 1 dòng/keyframe. Mỗi dòng đã có sẵn `frame_idx` + `pts_time`. **Không có bước gộp nào cả** — kênh OCR có **128,664 unit**, không phải ~85,000 như ước lượng ban đầu.

### 6.4 ASR nằm trong 14 thư mục batch không đều

```
Videos_L21_a  Videos_L22_a  Videos_L23_a  Videos_L24_a  Videos_L25_a
Videos_L26_a  Videos_L26_b  Videos_L26_c  Videos_L26_d  Videos_L26_e
Videos_L27_a  Videos_L28_a  Videos_L29_a  Videos_L30_a
```
L26 bị chia 5 batch (a–e). → Dùng `glob(f"{P_ASR_EN}/*/video/*.json")` rồi lập `dict[video_id] = path`. **Đừng** suy ra tên batch từ `video_id`.

## 7. ⚠️ Notebook SigLIP hiện có KHÔNG dùng lại được cho T4.1

Đã đọc source `ntloi131205/aic2026-siglip-embedding`. Bốn vấn đề:

| # | Vấn đề | Hệ quả |
|:-:|:--|:--|
| 1 | Nó **tự extract keyframe từ video** (`sim_threshold=0.95`, `skip_frames=5`), đặt tên `keyframe_{frame_id}.webp` | Ra **tập keyframe HOÀN TOÀN KHÁC** tập chính thức `001.jpg` → không align với `map-keyframes`, không align `keyframes.parquet` |
| 2 | Chỉ chạy `input_folder=".../video/L21_a"` | Mới xử lý **1 batch**, không phải 873 video |
| 3 | Model là `google/siglip-so400m-patch14-384` | Không phải `siglip2-giant` (bản tốt nhất theo `Model_Embedding.txt`) |
| 4 | Lưu **1 file `.pt` / keyframe** | 177K file nhỏ → I/O cực chậm, không dùng được với FAISS |

**Kết luận cho T4.1:** viết notebook SigLIP2 **mới**, embed đúng **ảnh keyframe chính thức** (`Keyframes_*/keyframes/<vid>/<nnn>.jpg`) theo **đúng thứ tự `keyframes.parquet`**, xuất **1 file `.npy` / video** giống `clip-features-32` để giữ nguyên data contract. Có thể tham khảo phần `DataParallel` 2×T4 của notebook cũ, phần còn lại viết lại.

## 8. Bảng path đối chiếu nhanh

| Cần gì | Path |
|:--|:--|
| CLIP `.npy` của `L21_V001` | `{P_CLIP}/L21_V001.npy` |
| Map keyframe | `{P_MAPKF}/L21_V001.csv` |
| Caption | `{P_CAPTION}/L21_V001.json` |
| Media info | `{P_MEDIA}/L21_V001.json` |
| Summary | `{P_SUMMARY}/L21_V001.json` |
| ASR tiếng Việt | `{P_ASR_VI}/Videos_L21_a/video/L21_V001.json` |
| ASR tiếng Anh | `{P_ASR_EN}/Videos_L21_a/video/L21_V001.json` |
| OCR index gộp | `{P_OCR_JSONL}` |
| Object 1 keyframe | `{P_OBJ_DIR}/L21_V001/001.json` |
| 584 class | `{P_OBJ_CLS}` — file 587 dòng CRLF, phải bỏ 2 dòng `#` + 1 dòng trống + `.strip()` |
| **Ảnh keyframe** | `{KF_DIR['L21_V001']}/001.jpg` → `{KEYFRAME_DS}/Keyframes_L21/keyframes/L21_V001/001.jpg` |
| File đề | `{QUERY_ROOT}/query-p1-16-trake.txt` |
