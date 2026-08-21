# %% [markdown] title
# # NB01 — INDEX BUILDER (AIC 2026, vòng sơ tuyển)
#
# Spec: `Planning/02_NB01_INDEX_BUILDER.md` · Contracts: `Planning/01_DATA_CONTRACTS.md` · Paths: `Planning/05_KAGGLE_PATHS.md`
#
# | | |
# |:--|:--|
# | **Input dataset** | `kitnehi1211/feature-aic-2026` **duy nhất** (KHÔNG attach `fatle542/aic-dataset`) |
# | **Output** | `/kaggle/working/index/` → publish thành Kaggle Dataset `aic26-index` |
# | **Accelerator** | `None` (CPU) nếu `EMBED_PROVIDER="openai"` · `GPU T4 x2` nếu `"local"` |
# | **Internet** | **BẮT BUỘC ON** |
# | **Thời gian** | ~3–4h (Phase B chiếm 1.5–3h) |
#
# > Chạy **một lần duy nhất**. Mỗi gói đề mới chỉ chạy NB02.
#
# **Bốn bất thường của dữ liệu đã verify — đừng debug lại:**
# 1. `video_id` không liên tục (`L21_V004`, `L21_V020` không tồn tại) → luôn `glob`, đừng `range()`.
# 2. `Summary_video/` có 866 `.json` nhưng 1 file là sentinel `_failed.json` → 865 video thật, 8 video thiếu (`L26_V072`–`L26_V079`).
# 3. `detected_classes.txt` = 587 dòng CRLF = 2 comment + 1 dòng trống + **584 class**.
# 4. `ocr_index.jsonl` **đã là 1 dòng / keyframe** (128,664 dòng) → không gộp gì cả.

# %% Cell 1 — SETUP (pip install)
# Chay 1 lan; neu da cai roi thi bo qua rat nhanh.
# sentence-transformers can cho CA HAI truong hop:
#   - Cell 13b: truy van thu #2 (CLIP text tower) - la 1 trong 3 cong DoD, khong duoc skip
#   - EMBED_PROVIDER="local": chay bge-m3
!pip install -q bm25s faiss-cpu pyvi PyStemmer pyarrow sentence-transformers

# %% Cell 1b — CONFIG (copy tu Planning/00_MASTER_PLAN.md §5)
# ============================================================
#  CONFIG - user tu dien API key, KHONG commit key vao git
# ============================================================
OPENROUTER_API_KEY = ""   # dung cho CA embedding va LLM/VLM neu EMBED_PROVIDER="openrouter"
OPENAI_API_KEY     = ""   # chi can khi EMBED_PROVIDER="openai"

# ---- Text embedding ----
# CAP NHAT 2026-08-21: OpenRouter DA co endpoint /api/v1/embeddings.
# Da verify: POST /api/v1/embeddings tra 401 (y nhu chat/completions) con path bia tra
# HTML 404 -> endpoint ton tai that. Planning/00_MASTER_PLAN.md QD-1 ("OpenRouter KHONG
# co endpoint embeddings") da LOI THOI. Loi ich: chi can 1 key duy nhat cho ca pipeline.
EMBED_PROVIDER    = "openrouter"              # "openrouter" | "openai" | "local"
EMBED_MODEL       = "text-embedding-3-small"  # ten "thuan", KHONG kem prefix provider
EMBED_DIM         = 1536
OPENAI_BASE_URL     = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_EMBED_MODEL = "BAAI/bge-m3"             # dung khi EMBED_PROVIDER="local" (dim=1024)
OR_SITE_URL       = ""                        # optional, cho bang xep hang openrouter.ai
OR_SITE_NAME      = "AIC2026"                 # optional
EMBED_BATCH       = 256
EMBED_MAX_RETRY   = 6
EMBED_CONCURRENCY = 8

# ---- Che do "Save & Version" (chay khong nguoi truc) ----
# Cell 9 smoke test KHONG duoc giet batch run vi mot nguong chat luong mem.
#   "warn"  = in canh bao roi chay tiep (mac dinh, hop voi Save & Version)
#   "abort" = SystemExit (chi dung khi chay tuong tac va muon chan chi tieu)
EMBED_ON_LOW_QUALITY = "warn"

# Cell 9b: so sanh vai model embedding tren cung bo cap VI-EN de co so lieu QUYET DINH
# cho version sau. Ton vai chuc token, gan nhu mien phi. Tat neu muon chay that nhanh.
EMBED_BENCHMARK = True
EMBED_CANDIDATES = [                # deu co tren OpenRouter /api/v1/embeddings
    ("openai/text-embedding-3-small", 1536, 0.02),
    ("baai/bge-m3",                   1024, 0.01),
    ("qwen/qwen3-embedding-4b",       2560, 0.02),
    ("intfloat/multilingual-e5-large", 1024, 0.01),
]

# ---- Resume Phase B sau khi mot lan "Save & Version" bi fail/timeout ----
# Kaggle CHI luu /kaggle/working khi run KET THUC THANH CONG. Run fail = mat sach
# checkpoint. Cach chua: publish WORK/ckpt thanh dataset roi attach lai, dien slug
# vao day - Cell 10 se seed tu do va skip phan da xong.
CKPT_INPUT_OWNER = ""               # vd "kitnehi1211"
CKPT_INPUT_SLUG  = ""               # vd "aic26-ckpt"

# ---- Visual embedding (QD-2) - PHAI khop model da embed keyframes ----
VISUAL_MODEL = "clip-ViT-B-32"                # tuong lai: "google/siglip2-giant-opt-patch16-384"
VISUAL_DIM   = 512

# ---- ASR windowing (data contracts §3.2) ----
ASR_WINDOW_SEC = 25.0
ASR_STRIDE_SEC = 10.0

# ---- Nguong khac ----
OCR_MIN_LEN            = 2
OBJECT_SCORE_THRESHOLD = 0.30
N_OBJECT_CLASSES       = 584

# ---- So luong ky vong (fail-fast) ----
N_VIDEOS_EXPECT    = 873
N_KEYFRAMES_EXPECT = 177_321
N_OCR_LINES_EXPECT = 128_664

WORK = "/kaggle/working"

# bge-m3 co dim 1024, khong phai 1536
if EMBED_PROVIDER == "local" and LOCAL_EMBED_MODEL == "BAAI/bge-m3":
    EMBED_DIM = 1024

def embed_endpoint():
    """-> (base_url, api_key, model_id_gui_di). OpenRouter can prefix 'openai/'."""
    if EMBED_PROVIDER == "openrouter":
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY, f"openai/{EMBED_MODEL}"
    if EMBED_PROVIDER == "openai":
        return OPENAI_BASE_URL, OPENAI_API_KEY, EMBED_MODEL
    return None, None, LOCAL_EMBED_MODEL

_bu, _k, _m = embed_endpoint()
print(f"EMBED_PROVIDER={EMBED_PROVIDER}  model={_m}  EMBED_DIM={EMBED_DIM}  base_url={_bu}")

# %% Cell 1c — PATHS + resolve() (copy tu Planning/05_KAGGLE_PATHS.md §2-3)
import os, json, glob, re, io, sys, time, math, random, unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# ============================================================
#  PATHS - da verify qua Kaggle API 2026-08-20
# ============================================================
KG = "/kaggle/input"

def resolve(owner: str, slug: str, inner: str = "") -> str:
    """Tra ve path mount thuc te cua dataset, thu ca 2 dang mount cua Kaggle."""
    for base in (f"{KG}/datasets/{owner}/{slug}", f"{KG}/{slug}"):
        p = os.path.join(base, inner) if inner else base
        if os.path.exists(p):
            return p
    hits = glob.glob(f"{KG}/**/{slug}", recursive=True)
    if hits:
        p = os.path.join(hits[0], inner) if inner else hits[0]
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Khong tim thay {owner}/{slug}. Da attach dataset chua?\n"
        f"Co san trong /kaggle/input: {os.listdir(KG) if os.path.isdir(KG) else 'KHONG CO'}"
    )

FEAT_ROOT = resolve("kitnehi1211", "feature-aic-2026", "Feature_Dataset")

# ---- Sub-path cua FEAT_ROOT (da verify ton tai) ----
P_CLIP      = f"{FEAT_ROOT}/clip-features-32-aic25-b1/clip-features-32"   # <vid>.npy      873
P_MAPKF     = f"{FEAT_ROOT}/map-keyframes-aic25-b1/map-keyframes"         # <vid>.csv      873
P_CAPTION   = f"{FEAT_ROOT}/Image_captioning"                             # <vid>.json     873
P_MEDIA     = f"{FEAT_ROOT}/media-info-aic25-b1/media-info"               # <vid>.json     873
P_SUMMARY   = f"{FEAT_ROOT}/Summary_video"                                # 866 = 865 that + _failed.json
P_ASR_VI    = f"{FEAT_ROOT}/Transcript_Extract"                           # */video/*.json 873
P_ASR_EN    = f"{FEAT_ROOT}/Transcript_Translated"                        # */video/*.json 873
P_OCR_DIR   = f"{FEAT_ROOT}/OCR_EasyOCR_VietOCR"                          # <vid>.json     700
P_OCR_JSONL = f"{FEAT_ROOT}/ocr_index.jsonl"                              # 128,664 dong = 1 dong/keyframe
P_OBJ_DIR   = f"{FEAT_ROOT}/objects-aic25-b1/objects"                     # <vid>/<nnn>.json
P_OBJ_CLS   = f"{FEAT_ROOT}/objects-aic25-b1/detected_classes.txt"        # 587 dong CRLF -> 584 class

OUT      = f"{WORK}/index"
OUT_FAI  = f"{OUT}/faiss"
OUT_BM25 = f"{OUT}/bm25"
CKPT     = f"{WORK}/ckpt"
for _d in (OUT, OUT_FAI, OUT_BM25, CKPT):
    os.makedirs(_d, exist_ok=True)

def nfc(s) -> str:
    """NFC-normalize + strip. BAT BUOC cho moi text tieng Viet (data contracts §6)."""
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s)).strip()

def make_kf_id(video_id: str, n) -> str:
    """Khoa chinh toan he thong: L21_V001#001"""
    return f"{video_id}#{int(n):03d}"

print("FEAT_ROOT =", FEAT_ROOT)

# %% Cell 2 — PATH_VERIFY (fail-fast, ~1 phut)
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
for _k, (_pat, _n) in EXPECT.items():
    _got = len(glob.glob(_pat))
    assert _got == _n, f"{_k}: thay {_got}, ky vong {_n}  (pattern: {_pat})"
    print(f"  {_k:8s} {_got:6d} OK")

n_ocr_lines = sum(1 for _ in open(P_OCR_JSONL, encoding="utf-8"))
assert n_ocr_lines == N_OCR_LINES_EXPECT, f"ocr_index.jsonl: {n_ocr_lines} dong, ky vong {N_OCR_LINES_EXPECT}"
print(f"  ocr_jsnl {n_ocr_lines:6d} OK")

def load_object_classes(path: str) -> list:
    """detected_classes.txt: 587 dong CRLF = 2 comment + 1 dong trong + 584 class.

    .strip() bo luon '\\r'. KHONG dung enumerate(splitlines()) tho (giu lai header
    -> lech index 3 cot) va KHONG dung split('\\n') (moi ten dinh '\\r').
    """
    lines = [l.strip() for l in open(path, encoding="utf-8")]
    cls = [l for l in lines if l and not l.startswith("#")]
    assert len(cls) == N_OBJECT_CLASSES, f"parse ra {len(cls)} class, ky vong {N_OBJECT_CLASSES}"
    return cls

OBJ_CLASSES = load_object_classes(P_OBJ_CLS)
OBJ_CLS_IDX = {c: i for i, c in enumerate(OBJ_CLASSES)}
print(f"  obj_cls  {len(OBJ_CLASSES):6d} OK   (vd: {OBJ_CLASSES[:3]})")

n_asr_batch = len(glob.glob(f"{P_ASR_EN}/*"))
assert n_asr_batch == 14, f"ASR batch: {n_asr_batch}, ky vong 14"
print(f"  asr_btch {n_asr_batch:6d} OK")

# ASR nam rai trong 14 thu muc batch khong deu -> lap dict, DUNG suy ra ten batch tu video_id
ASR_EN_PATH = {Path(p).stem: p for p in glob.glob(f"{P_ASR_EN}/*/video/*.json")}
ASR_VI_PATH = {Path(p).stem: p for p in glob.glob(f"{P_ASR_VI}/*/video/*.json")}
assert len(ASR_EN_PATH) == 873 and len(ASR_VI_PATH) == 873

VIDEO_IDS = sorted(Path(p).stem for p in glob.glob(f"{P_MAPKF}/*.csv"))
VIDEO_SET = set(VIDEO_IDS)
assert len(VIDEO_IDS) == N_VIDEOS_EXPECT

print(f"\nOK - {len(VIDEO_IDS)} video.")
print(f"video_id KHONG lien tuc (da verify): 'L21_V004' in VIDEO_SET = {'L21_V004' in VIDEO_SET}  (ky vong False)")
print("Phan bo theo prefix:", dict(sorted(Counter(v.split('_')[0] for v in VIDEO_IDS).items())))

# %% [markdown] phase-a1
# ## Cell 3 — PHASE_A1: `keyframes.parquet` (177,321 dòng)
#
# Invariant bắt buộc: với mỗi `video_id`, số dòng phải **==** `np.load(clip/<vid>.npy).shape[0]`.
# Lệch → visual FAISS lệch hàng, toàn bộ kênh visual sai **mà không báo lỗi**. Dừng lại, đừng tự sửa.

# %% Cell 3 — PHASE_A1: keyframes.parquet
t0 = time.time()
_frames = []
for vid in VIDEO_IDS:
    d = pd.read_csv(f"{P_MAPKF}/{vid}.csv")
    d["video_id"] = vid
    _frames.append(d)
kf = pd.concat(_frames, ignore_index=True)
del _frames

kf["n"]         = kf["n"].astype("int32")          # cast tu float (1.0 -> 1)
kf["frame_idx"] = kf["frame_idx"].astype("int32")  # GIA TRI NOP BAI
kf["pts_time"]  = kf["pts_time"].astype("float32")
kf["fps"]       = kf["fps"].astype("float32")

kf = kf.sort_values(["video_id", "n"], kind="mergesort").reset_index(drop=True)
_nnn         = kf["n"].map("{:03d}".format)
kf["kf_id"]   = kf["video_id"] + "#" + _nnn
kf["kf_path"] = kf["video_id"] + "/" + _nnn + ".jpg"
kf["vis_row"] = np.arange(len(kf), dtype="int32")
kf = kf[["kf_id", "video_id", "n", "frame_idx", "pts_time", "fps", "kf_path", "vis_row"]]

assert len(kf) == N_KEYFRAMES_EXPECT, f"{len(kf)} dong, ky vong {N_KEYFRAMES_EXPECT}"
assert kf["kf_id"].is_unique, "kf_id KHONG unique"
assert kf[["n", "frame_idx", "pts_time", "fps"]].notna().all().all(), "co NaN"

# ---- Invariant: so keyframe == so hang trong .npy CLIP ----
mismatch = []
for vid, g in kf.groupby("video_id", sort=False, observed=True):
    n_npy = np.load(f"{P_CLIP}/{vid}.npy", mmap_mode="r").shape[0]
    if len(g) != n_npy:
        mismatch.append((vid, int(len(g)), int(n_npy)))
assert not mismatch, f"LECH SO KEYFRAME (ghi vao manifest, KHONG tu sua): {mismatch}"

kf.to_parquet(f"{OUT}/keyframes.parquet", index=False)
KF_MISMATCH = mismatch
print(f"keyframes.parquet: {len(kf):,} dong · {kf.video_id.nunique()} video · {time.time()-t0:.0f}s")
kf.head(3)

# %% [markdown] phase-a2
# ## Cell 4 — PHASE_A2: `videos.parquet` (873 dòng)
#
# Fallback bắt buộc: `summary_en` null → `description` → `""` (và `has_summary=False`).
# Skip mọi stem bắt đầu bằng `_` trong `Summary_video/` (sentinel `_failed.json`).

# %% Cell 4 — PHASE_A2: videos.parquet
t0 = time.time()

def _read_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ---- Summary: SKIP stem bat dau bang '_' ----
SUMMARY = {}
for p in glob.glob(f"{P_SUMMARY}/*.json"):
    stem = Path(p).stem
    if stem.startswith("_"):
        continue                     # sentinel _failed.json - KHONG phai video
    SUMMARY[stem] = _read_json(p)
assert "_failed" not in SUMMARY
print(f"summary that: {len(SUMMARY)} (ky vong 865)")

# ---- OCR coverage: tinh tu ocr_index.jsonl THAT, dung suy ra tu prefix ----
OCR_VIDEOS = set()
with open(P_OCR_JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                OCR_VIDEOS.add(json.loads(line)["video_id"])
            except Exception:
                pass
print(f"video co OCR: {len(OCR_VIDEOS)} (ky vong 700)")

def _asr_fulltext(path):
    """Tra ve (text_vi, text_en) ghep tu segments."""
    d = _read_json(path)
    segs = d.get("segments") or []
    vi = " ".join(nfc(s.get("text")) for s in segs if s.get("text"))
    en = " ".join((s.get("text_en") or "").strip() for s in segs if s.get("text_en"))
    if not vi:
        vi = nfc(d.get("text"))
    if not en:
        en = (d.get("text_en") or "").strip()
    return vi, en

kf_agg = kf.groupby("video_id", observed=True).agg(
    n_keyframes=("n", "size"),
    max_pts=("pts_time", "max"),
    fps_mode=("fps", lambda s: float(s.mode().iloc[0]) if len(s.mode()) else float(s.iloc[0])),
)

rows = []
for vid in VIDEO_IDS:
    mi  = _read_json(f"{P_MEDIA}/{vid}.json")
    sm  = SUMMARY.get(vid, {})
    tvi, ten = _asr_fulltext(ASR_EN_PATH[vid])   # Transcript_Translated co ca text va text_en
    if not tvi:                                   # fallback sang Transcript_Extract
        tvi, _ = _asr_fulltext(ASR_VI_PATH[vid])

    desc       = nfc(mi.get("description"))
    summary_en = (sm.get("summary") or "").strip()
    has_summary = bool(summary_en)
    if not summary_en:
        summary_en = desc                        # fallback bat buoc

    kws = mi.get("keywords") or []
    if isinstance(kws, str):
        kws = [k.strip() for k in kws.split(",") if k.strip()]

    length = mi.get("length")
    try:
        duration = float(length)
    except (TypeError, ValueError):
        duration = float(kf_agg.at[vid, "max_pts"])

    rows.append({
        "video_id":     vid,
        "n_keyframes":  int(kf_agg.at[vid, "n_keyframes"]),
        "duration_sec": np.float32(duration),
        "fps":          np.float32(kf_agg.at[vid, "fps_mode"]),
        "title":        nfc(mi.get("title")),
        "description":  desc,
        "keywords":     [nfc(k) for k in kws],
        "author":       nfc(mi.get("author")),
        "publish_date": str(mi.get("publish_date") or ""),
        "watch_url":    str(mi.get("watch_url") or ""),
        "summary_en":   summary_en,
        "topics":       [str(t).strip() for t in (sm.get("topics") or [])],
        "entities":     [nfc(e) for e in (sm.get("entities") or [])],
        "transcript_vi": tvi,
        "transcript_en": ten,
        "has_ocr":       vid in OCR_VIDEOS,
        "has_summary":   has_summary,
        "has_transcript": len(ten) > 0,
    })

videos = pd.DataFrame(rows)
videos["n_keyframes"] = videos["n_keyframes"].astype("int32")

assert len(videos) == N_VIDEOS_EXPECT
assert "_failed" not in set(videos.video_id), "sentinel _failed lot vao videos.parquet"
assert int(videos.has_ocr.sum()) == 700, int(videos.has_ocr.sum())
assert int(videos.has_summary.sum()) == 865, int(videos.has_summary.sum())

VIDEOS_MISSING_OCR     = sorted(videos.loc[~videos.has_ocr, "video_id"])
VIDEOS_MISSING_SUMMARY = sorted(videos.loc[~videos.has_summary, "video_id"])
videos.to_parquet(f"{OUT}/videos.parquet", index=False)

print(f"videos.parquet: {len(videos)} dong · {time.time()-t0:.0f}s")
print(f"  thieu OCR:     {len(VIDEOS_MISSING_OCR)}  -> {VIDEOS_MISSING_OCR[:3]} ...")
print(f"  thieu summary: {len(VIDEOS_MISSING_SUMMARY)} -> {VIDEOS_MISSING_SUMMARY}")

# %% [markdown] phase-a3
# ## Cell 5 — PHASE_A3: `text_units.parquet` (~319,000 dòng)
#
# Năm điểm dễ sai nhất (`01_DATA_CONTRACTS.md §3.2`):
#
# 1. **Caption duplicate** — `duplicate_of` là **tên file** (`"009.jpg"`), không phải `kf_id`. Dòng dup vẫn tạo unit, `emb_row = -2` (patch ở Cell 11).
# 2. **ASR windowing + dedupe** — segment dài trung bình 24.2s ≈ đúng cửa sổ 25s, stride 10s làm cùng segment lặp ở 2–3 window. **Dedupe là bắt buộc**, không phải tối ưu hoá.
# 3. **ASR → `frame_idx`** dùng `np.searchsorted`, không loop Python.
# 4. **OCR không gộp gì cả** — jsonl đã 1 dòng/keyframe, đã có sẵn `frame_idx` + `pts_time`.
# 5. **Summary** — mốc thời gian nằm trong `chunk_summaries` (`"- [00:00-02:00] ..."`), **KHÔNG** trong `evidence` (đó là dict metadata provenance, không có timestamp).

# %% Cell 5a — helper: tra cuu keyframe theo video (dung cho ASR/summary)
# pts_time da sort tang dan theo n trong tung video -> searchsorted O(log N)
KF_BY_VIDEO = {}
for vid, g in kf.groupby("video_id", sort=False, observed=True):
    KF_BY_VIDEO[vid] = (
        g["pts_time"].to_numpy(dtype="float32"),
        g["frame_idx"].to_numpy(dtype="int32"),
        g["kf_id"].to_numpy(),
    )

def nearest_frame_idx(video_id: str, t: float):
    """frame_idx cua keyframe co pts_time gan t nhat. None neu khong xac dinh."""
    if video_id not in KF_BY_VIDEO or t is None or not np.isfinite(t):
        return None
    pts, fidx, _ = KF_BY_VIDEO[video_id]
    i = int(np.searchsorted(pts, t))
    if i <= 0:
        j = 0
    elif i >= len(pts):
        j = len(pts) - 1
    else:
        j = i if abs(pts[i] - t) < abs(pts[i - 1] - t) else i - 1
    return int(fidx[j])

KF_FRAME_IDX = dict(zip(kf.kf_id, kf.frame_idx))
KF_PTS       = dict(zip(kf.kf_id, kf.pts_time))
print("helper ready")

# %% Cell 5b — build_caption_units  (~163,000 unit)
def build_caption_units() -> list:
    """Image_captioning/<vid>.json -> keyframes[].

    duplicate_of la TEN FILE ("009.jpg") -> parse ra n roi moi dung kf_id.
    Dong dup van tao unit (van la ket qua nop bai hop le), text_en = caption
    cua canonical, emb_row = -2 (danh dau 'dup, patch sau' - phan biet voi -1
    la 'khong embed').
    """
    out, n_dup = [], 0
    for vid in VIDEO_IDS:
        d = _read_json(f"{P_CAPTION}/{vid}.json")
        items = d.get("keyframes") or d.get("results") or []
        if isinstance(items, dict):
            items = list(items.values())

        # pass 1: map n -> caption (de dong dup lay text cua canonical)
        cap_by_n, parsed = {}, []
        for it in items:
            name = it.get("keyframe") or it.get("file") or it.get("image") or it.get("filename") or ""
            n = it.get("n")
            if n is None:
                m = re.search(r"(\d+)", str(name))
                if not m:
                    continue
                n = int(m.group(1))
            n = int(n)
            cap = (it.get("caption") or it.get("text") or "").strip()
            dup = it.get("duplicate_of")
            dup_n = None
            if dup:
                m = re.search(r"(\d+)", str(dup))
                if m:
                    dup_n = int(m.group(1))
            if not dup_n and cap:
                cap_by_n[n] = cap
            parsed.append((n, cap, dup_n))

        for n, cap, dup_n in parsed:
            kid = make_kf_id(vid, n)
            if kid not in KF_FRAME_IDX:
                continue                       # keyframe khong co trong map-keyframes
            if dup_n is not None:
                text = cap or cap_by_n.get(dup_n, "")
                emb_row, dup_of = -2, make_kf_id(vid, dup_n)
                n_dup += 1
            else:
                text, emb_row, dup_of = cap, -3, None   # -3 = cho embed
            pts = float(KF_PTS[kid])
            out.append({
                "unit_id": f"cap:{kid}", "channel": "caption", "video_id": vid,
                "kf_id": kid, "frame_idx": int(KF_FRAME_IDX[kid]),
                "t_start": pts, "t_end": pts,
                "text_en": text, "text_vi": None, "text_embed": text,
                "lang_native": "en", "emb_row": emb_row, "dup_of": dup_of,
            })
    print(f"  caption: {len(out):,} unit ({n_dup:,} duplicate = {n_dup/max(len(out),1):.1%})")
    return out

t0 = time.time()
U_CAPTION = build_caption_units()
print(f"  {time.time()-t0:.0f}s")

# %% Cell 5c — build_ocr_units  (128,664 unit, 1 unit / keyframe)
def build_ocr_units() -> list:
    """ocr_index.jsonl: {video_id, frame_idx, pts_time, keyframe, text}.

    DA LA 1 dong / keyframe (128,664 dong <-> 128,664 cap unique). KHONG gop gi ca.
    Moi dong da co san frame_idx + pts_time -> khong can join keyframes.parquet,
    nhung van assert de bat lech du lieu.
    """
    out, n_short, n_mismatch, seen = [], 0, 0, set()
    with open(P_OCR_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vid = r["video_id"]
            name = str(r.get("keyframe", ""))
            m = re.search(r"(\d+)", name)
            if not m:
                continue
            n = int(m.group(1))
            kid = make_kf_id(vid, n)
            if kid in seen:
                continue                      # bao ve, du da verify la unique
            seen.add(kid)

            text = nfc(r.get("text"))
            if len(text) < OCR_MIN_LEN:       # loc rac, nhung DUNG loc manh tay
                n_short += 1
                continue

            fidx = int(r.get("frame_idx", -1))
            if kid in KF_FRAME_IDX and int(KF_FRAME_IDX[kid]) != fidx:
                n_mismatch += 1
                fidx = int(KF_FRAME_IDX[kid])  # tin keyframes.parquet
            t = float(r.get("pts_time", KF_PTS.get(kid, 0.0)))
            out.append({
                "unit_id": f"ocr:{kid}", "channel": "ocr", "video_id": vid,
                "kf_id": kid, "frame_idx": fidx, "t_start": t, "t_end": t,
                "text_en": None, "text_vi": text, "text_embed": text,
                "lang_native": "vi", "emb_row": -3, "dup_of": None,
            })
    print(f"  ocr: {len(out):,} unit  (bo {n_short:,} chuoi < {OCR_MIN_LEN} ky tu · "
          f"{n_mismatch} frame_idx lech so voi keyframes.parquet)")
    return out

t0 = time.time()
U_OCR = build_ocr_units()
print(f"  {time.time()-t0:.0f}s")

# %% Cell 5d — build_asr_units  (~21,500 unit sau dedupe)
def build_asr_units(window=None, stride=None) -> list:
    """Cua so truot tren truc THOI GIAN (khong phai tren so segment).

    DEDUPE BAT BUOC: segment ASR dai trung binh 24.2s ~ dung cua so 25s, nen voi
    stride 10s cung mot segment roi vao 2-3 cua so lien tiep va sinh unit GIONG HET
    nhau. Do tren 40 video: 1,600 window / chi 985 tap segment khac nhau (du 1.62x).
    Khong dedupe thi vua phi ~38% tien embed ASR, vua de mot doan text tu khuech dai
    diem RRF cua chinh no vi xuat hien o nhieu rank.
    Hai lop dedupe (do lop 1 chua du - da do tren du lieu that):
      1. tuple(sorted(seg_id))  - cung tap segment
      2. text_embed             - tap segment KHAC nhau van co the ra text GIONG HET
                                  (vd cac segment them vao co text rong)
    Giu window dau tien.
    """
    window = window or ASR_WINDOW_SEC
    stride = stride or ASR_STRIDE_SEC
    out, n_raw = [], 0
    for vid in VIDEO_IDS:
        d = _read_json(ASR_EN_PATH[vid])
        segs = d.get("segments") or []
        segs = [s for s in segs if s.get("start") is not None and s.get("end") is not None]
        if not segs:
            continue
        for i, s in enumerate(segs):
            s.setdefault("id", i)
        segs.sort(key=lambda s: float(s["start"]))
        s_start = np.array([float(s["start"]) for s in segs], dtype="float64")
        s_end   = np.array([float(s["end"])   for s in segs], dtype="float64")
        v_end   = float(s_end.max())

        seen, seen_text, w = set(), set(), 0
        t = 0.0
        while True:
            lo, hi = t, t + window
            # segment giao voi cua so
            sel = np.nonzero((s_end > lo) & (s_start < hi))[0]
            if len(sel):
                n_raw += 1
                key = tuple(sorted(int(segs[j]["id"]) for j in sel))
                if key not in seen:
                    seen.add(key)
                    hit = [segs[j] for j in sel]
                    t_start = float(min(float(s["start"]) for s in hit))
                    t_end   = float(max(float(s["end"])   for s in hit))
                    text_vi = nfc(" ".join((s.get("text") or "").strip() for s in hit))
                    text_en = " ".join((s.get("text_en") or "").strip() for s in hit).strip()
                    embed_t = text_en or text_vi
                    if embed_t and embed_t not in seen_text:   # lop dedupe 2
                        seen_text.add(embed_t)
                        out.append({
                            "unit_id": f"asr:{vid}#{w:04d}", "channel": "asr", "video_id": vid,
                            "kf_id": None,
                            "frame_idx": nearest_frame_idx(vid, (t_start + t_end) / 2.0),
                            "t_start": t_start, "t_end": t_end,
                            "text_en": text_en or None, "text_vi": text_vi or None,
                            "text_embed": embed_t, "lang_native": "vi",
                            "emb_row": -3, "dup_of": None,
                        })
                        w += 1
            t += stride
            if t >= v_end:
                break
    ratio = n_raw / max(len(out), 1)
    print(f"  asr: {len(out):,} unit (tu {n_raw:,} window tho -> dedupe {ratio:.2f}x)")
    return out

t0 = time.time()
U_ASR = build_asr_units()
print(f"  {time.time()-t0:.0f}s")

# %% Cell 5e — build_summary_units  (~5,200 unit)
CHUNK_TS_RE = re.compile(
    r"\s*-?\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\]"
)

def _parse_chunk_ts(s: str):
    """'- [00:00-02:00] A Vietnamese news report ...' -> (t_start, t_end) giay.

    Day la NGUON TEXT-THEO-THOI-GIAN DUY NHAT o cap video. Khong match -> (None, None),
    van giu unit.
    """
    m = CHUNK_TS_RE.match(s or "")
    if not m:
        return None, None
    a1, a2, a3, b1, b2, b3 = m.groups()
    if a3 is None:   # dang MM:SS
        t0 = int(a1) * 60 + int(a2)
        t1 = int(b1) * 60 + int(b2)
    else:            # dang HH:MM:SS
        t0 = int(a1) * 3600 + int(a2) * 60 + int(a3)
        t1 = int(b1) * 3600 + int(b2) * 60 + int(b3)
    return float(t0), float(t1)

def build_summary_units() -> list:
    """Summary_video/<vid>.json -> 1 unit video-level + 1 unit / chunk_summaries[i].

    KHONG dung 'evidence': da verify no la dict metadata provenance
    ({visual_count, speech_count, has_caption, ...}), KHONG co truong thoi gian nao.
    """
    out, n_ts = [], 0
    for vid, sm in SUMMARY.items():
        if vid not in VIDEO_SET:
            continue
        head = (sm.get("summary") or "").strip()
        topics   = [str(t).strip() for t in (sm.get("topics") or [])]
        entities = [nfc(e) for e in (sm.get("entities") or [])]
        parts = [p for p in [head, " | ".join(topics), " | ".join(entities)] if p]
        if parts:
            out.append({
                "unit_id": f"sum:{vid}", "channel": "summary", "video_id": vid,
                "kf_id": None, "frame_idx": None, "t_start": None, "t_end": None,
                "text_en": " | ".join(parts), "text_vi": None,
                "text_embed": " | ".join(parts), "lang_native": "en",
                "emb_row": -3, "dup_of": None,
            })
        for i, ch in enumerate(sm.get("chunk_summaries") or []):
            ch = str(ch).strip()
            if not ch:
                continue
            ts, te = _parse_chunk_ts(ch)
            if ts is not None:
                n_ts += 1
            out.append({
                "unit_id": f"sum:{vid}#c{i:02d}", "channel": "summary", "video_id": vid,
                "kf_id": None,
                "frame_idx": nearest_frame_idx(vid, (ts + te) / 2.0) if ts is not None else None,
                "t_start": ts, "t_end": te,
                "text_en": ch, "text_vi": None, "text_embed": ch,
                "lang_native": "en", "emb_row": -3, "dup_of": None,
            })
    print(f"  summary: {len(out):,} unit ({n_ts:,} chunk parse duoc moc thoi gian)")
    return out

def build_meta_units() -> list:
    out = []
    for r in videos.itertuples(index=False):
        kws = ", ".join(r.keywords) if len(r.keywords) else ""
        txt = nfc(f"{r.title}. {r.description}. Keywords: {kws}")
        out.append({
            "unit_id": f"meta:{r.video_id}", "channel": "meta", "video_id": r.video_id,
            "kf_id": None, "frame_idx": None, "t_start": None, "t_end": None,
            "text_en": txt, "text_vi": txt, "text_embed": txt,
            "lang_native": "mixed", "emb_row": -3, "dup_of": None,
        })
    print(f"  meta: {len(out):,} unit")
    return out

U_SUMMARY = build_summary_units()
U_META    = build_meta_units()

# %% Cell 5f — hop nhat -> text_units.parquet
t0 = time.time()
units = pd.DataFrame(U_CAPTION + U_OCR + U_ASR + U_SUMMARY + U_META)
del U_CAPTION, U_OCR, U_ASR, U_SUMMARY, U_META

units["channel"]     = units["channel"].astype("category")
units["video_id"]    = units["video_id"].astype("category")
units["lang_native"] = units["lang_native"].astype("category")
units["frame_idx"]   = units["frame_idx"].astype("Int32")
units["t_start"]     = units["t_start"].astype("float32")
units["t_end"]       = units["t_end"].astype("float32")
units["emb_row"]     = units["emb_row"].astype("int32")

assert units["unit_id"].is_unique, "unit_id KHONG unique"
n_ocr_units = int((units.channel == "ocr").sum())

counts = units.channel.value_counts()
# Ky vong DA HIEU CHINH theo do dac tren Feature_Dataset that (tests/smoke_nb01.py).
# Khac voi uoc luong o 00_MASTER_PLAN §6 o 2 cho, ca hai deu KHONG phai bug:
#   - caption: 1 unit / keyframe = 177,321 (plan ghi ~163,000 vi da tru dup; nhung theo
#     data contracts §3.2 dong dup VAN tao unit). Ty le dup that ~17.9%, khong phai 8.2%
#     -> so unit CAN EMBED moi la ~145,500.
#   - summary: ~1,355 (plan uoc ~5,200). chunk_summaries thuc te chi ~0.6 chunk/video.
EXPECT_UNITS = {"caption": 177_321, "ocr": 128_300, "asr": 23_600, "summary": 1_355, "meta": 873}
print("value_counts theo channel:")
for c, exp in EXPECT_UNITS.items():
    got = int(counts.get(c, 0))
    dev = abs(got - exp) / exp
    flag = "OK " if dev <= 0.25 else "!! LECH > 25% - kiem tra lai"
    print(f"  {c:8s} {got:>8,}  (ky vong ~{exp:,}, lech {dev:+.0%})  {flag}")
n_to_embed = int((units.emb_row == -3).sum())
print(f"\n  can embed: {n_to_embed:,} unit "
      f"(bo {int((units.emb_row == -2).sum()):,} caption duplicate)")

# Cac assert cung
assert n_ocr_units <= N_OCR_LINES_EXPECT
assert int((units.channel == "meta").sum()) == 873
assert units.loc[units.channel == "asr", "frame_idx"].notna().all(), "ASR co frame_idx null"
# Chung minh dedupe window da chay: khong 2 unit asr nao cung (video_id, text_embed).
# Dung text_embed chu KHONG phai text_en: text_en co the None (fallback sang text_vi)
# va nhieu None trong cung video se bi tinh nham la trung.
_asr = units[units.channel == "asr"]
assert not _asr.duplicated(subset=["video_id", "text_embed"]).any(), "ASR CHUA dedupe"

units.to_parquet(f"{OUT}/text_units.parquet", index=False)
print(f"\ntext_units.parquet: {len(units):,} dong · {time.time()-t0:.0f}s")

# %% [markdown] phase-a4
# ## Cell 6 — PHASE_A4: `objects.parquet` + `objects_matrix.npz`
#
# 177,321 file JSON nhỏ → I/O bound nặng, dùng `ThreadPoolExecutor`.
# ⚠️ `detection_scores` trong JSON là **chuỗi** (`"0.79673874"`) — so sánh trực tiếp với float sẽ `TypeError`.

# %% Cell 6 — PHASE_A4: objects
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()

def _read_obj(args):
    """-> (vis_row, [class_idx], [max_score]) — da loc score >= threshold, dedupe max."""
    vis_row, vid, n = args
    p = f"{P_OBJ_DIR}/{vid}/{n:03d}.json"
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return vis_row, [], []
    ents   = d.get("detection_class_entities") or []
    scores = d.get("detection_scores") or []
    best = {}
    for e, s in zip(ents, scores):
        try:
            s = float(s)                    # JSON luu dang CHUOI -> BAT BUOC cast
        except (TypeError, ValueError):
            continue
        if s < OBJECT_SCORE_THRESHOLD:
            continue
        e = str(e).strip()
        if e in OBJ_CLS_IDX and s > best.get(e, -1.0):
            best[e] = s
    if not best:
        return vis_row, [], []
    items = sorted(best.items(), key=lambda kv: -kv[1])
    return vis_row, [k for k, _ in items], [v for _, v in items]

_tasks = list(zip(kf.vis_row.tolist(), kf.video_id.astype(str).tolist(), kf.n.tolist()))
results = [None] * len(_tasks)
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, res in enumerate(ex.map(_read_obj, _tasks, chunksize=256)):
        results[res[0]] = res
        if (i + 1) % 20000 == 0:
            print(f"  {i+1:,}/{len(_tasks):,}  ({time.time()-t0:.0f}s)")

obj_classes_col, obj_scores_col, rows_i, cols_j = [], [], [], []
for vis_row, cls, scs in results:
    obj_classes_col.append(cls)
    obj_scores_col.append(np.asarray(scs, dtype="float32"))
    for c in cls:
        rows_i.append(vis_row)
        cols_j.append(OBJ_CLS_IDX[c])
del results

objects = pd.DataFrame({
    "kf_id":   kf.kf_id.values,
    "classes": obj_classes_col,
    "scores":  obj_scores_col,
})
objects.to_parquet(f"{OUT}/objects.parquet", index=False)

OBJ_M = sp.csr_matrix(
    (np.ones(len(rows_i), dtype=bool), (rows_i, cols_j)),
    shape=(len(kf), N_OBJECT_CLASSES), dtype=bool,
)
sp.save_npz(f"{OUT}/objects_matrix.npz", OBJ_M)

# Ghi object_classes.txt SACH 584 dong (DUNG copy nguyen file goc - con 2 comment + CRLF)
with open(f"{OUT}/object_classes.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(OBJ_CLASSES) + "\n")

assert OBJ_M.shape == (N_KEYFRAMES_EXPECT, N_OBJECT_CLASSES), OBJ_M.shape
# spot-check: kf co 'Person' trong classes thi matrix cung True tai cot Person
_pi = OBJ_CLS_IDX.get("Person")
if _pi is not None:
    _hit = [i for i, c in enumerate(obj_classes_col[:5000]) if "Person" in c]
    if _hit:
        assert all(OBJ_M[i, _pi] for i in _hit[:50]), "matrix lech so voi classes"
        print(f"  spot-check Person: {len(_hit)} kf trong 5000 dau, matrix khop")

print(f"objects: nnz={OBJ_M.nnz:,} · shape={OBJ_M.shape} · {time.time()-t0:.0f}s")

# %% [markdown] phase-e
# ## Cell 7 — PHASE_E: `vision.faiss` (512-d, 177,321 vector)
#
# Thứ tự `vstack` phải **y hệt** thứ tự sort của `kf` (`video_id` tăng dần, rồi `n`).
# Lệch 1 video là toàn bộ kênh visual sai **mà không hề báo lỗi**.

# %% Cell 7 — PHASE_E: vision.faiss
import faiss

t0 = time.time()
mats = []
for vid in VIDEO_IDS:                                    # THU TU PHAI KHOP kf (da sort theo video_id)
    a = np.load(f"{P_CLIP}/{vid}.npy").astype("float32")  # fp16 -> fp32 BAT BUOC, FAISS khong nhan fp16
    mats.append(a)
X = np.vstack(mats)
del mats

assert X.shape == (len(kf), VISUAL_DIM), f"{X.shape} != {(len(kf), VISUAL_DIM)}"
faiss.normalize_L2(X)                                    # BAT BUOC cho cosine qua inner-product
vis_index = faiss.IndexFlatIP(VISUAL_DIM)
vis_index.add(X)
faiss.write_index(vis_index, f"{OUT_FAI}/vision.faiss")
np.save(f"{OUT_FAI}/vision_rowmap.npy", kf.kf_id.values.astype(object), allow_pickle=True)

assert vis_index.ntotal == len(kf)
print(f"vision.faiss: ntotal={vis_index.ntotal:,} dim={VISUAL_DIM} · {time.time()-t0:.0f}s")
del X

# %% [markdown] phase-d
# ## Cell 8 — PHASE_D: 6 BM25 index
#
# `bm25s` (sparse scipy) thay `rank_bm25` (pure-Python, hàng giây/query → không chấp nhận được).
#
# ⚠️ **NFC-normalize bắt buộc** cho cả corpus và query: tiếng Việt có 2 dạng dựng sẵn/tổ hợp
# (`ế` = 1 codepoint hoặc `e` + dấu) — không normalize thì BM25 miss hoàn toàn.

# %% Cell 8 — PHASE_D: BM25 x 6
import bm25s

TOKENIZER_VI = "pyvi"
try:
    from pyvi import ViTokenizer
    _ = ViTokenizer.tokenize("kiem tra")
except Exception as e:
    print(f"[WARN] pyvi loi ({e}) -> fallback tokenizer EN cho tieng Viet")
    ViTokenizer = None
    TOKENIZER_VI = "fallback_regex"

def tok_en(s) -> list:
    return re.sub(r"[^\w\s]", " ", (s or "").lower()).split()

def tok_vi(s) -> list:
    s = nfc(s).lower()
    if ViTokenizer is None:
        return re.split(r"\W+", s) and [w for w in re.split(r"\W+", s) if w]
    try:
        return ViTokenizer.tokenize(s).split()
    except Exception:
        return [w for w in re.split(r"\W+", s) if w]

BM25_SPEC = [
    ("bm25_caption_en", "caption", "text_en",    tok_en),
    ("bm25_ocr_vi",     "ocr",     "text_vi",    tok_vi),
    ("bm25_asr_vi",     "asr",     "text_vi",    tok_vi),
    ("bm25_asr_en",     "asr",     "text_en",    tok_en),
    ("bm25_summary_en", "summary", "text_en",    tok_en),
    ("bm25_meta",       "meta",    "text_embed", tok_vi),
]

t0 = time.time()
BM25_SIZES = {}
for name, channel, field, tok in BM25_SPEC:
    sub = units.loc[units.channel == channel, ["unit_id", field]]
    sub = sub[sub[field].notna() & (sub[field].astype(str).str.strip() != "")]
    ids  = sub["unit_id"].to_numpy()
    corp = [tok(t) for t in sub[field].astype(str).tolist()]
    r = bm25s.BM25()
    r.index(corp)
    d = f"{OUT_BM25}/{name}"
    r.save(d)
    np.save(f"{d}/ids.npy", ids.astype(object), allow_pickle=True)
    BM25_SIZES[name] = int(len(ids))
    print(f"  {name:18s} {len(ids):>8,} doc")

print(f"BM25 xong · tokenizer_vi={TOKENIZER_VI} · {time.time()-t0:.0f}s")

# %% [markdown] embed-client
# ## Cell 9 — `EMBED_CLIENT`: một interface, hai backend (QĐ-1)
#
# OpenRouter **không có** endpoint `/v1/embeddings` — chỉ proxy chat completions.
# Nên `EMBED_PROVIDER="openai"` gọi thẳng OpenAI (~$0.26 cho toàn corpus), hoặc `"local"` chạy `bge-m3` trên T4.
#
# **Smoke test bắt buộc trước khi tiêu $.** `cos(en, vi) > 0.5` → model hiểu tiếng Việt, embed OCR trực tiếp VI là hợp lý.
# Nếu `< 0.35` → đổi sang `EMBED_PROVIDER="local"` + `bge-m3`.

# %% Cell 9 — EMBED_CLIENT
import requests

_ST_MODEL = None

def _embed_http(texts: list) -> np.ndarray:
    """POST {base}/embeddings - dung CHUNG cho openai va openrouter (cung schema).

    Retry 429/5xx voi exponential backoff + jitter.
    """
    base, key, model = embed_endpoint()
    assert key, f"Chua dien API key cho EMBED_PROVIDER='{EMBED_PROVIDER}' o cell CONFIG"
    url = f"{base}/embeddings"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if EMBED_PROVIDER == "openrouter":          # optional, cho bang xep hang openrouter.ai
        if OR_SITE_URL:
            hdr["HTTP-Referer"] = OR_SITE_URL
        if OR_SITE_NAME:
            hdr["X-Title"] = OR_SITE_NAME
    body = {"model": model, "input": texts, "encoding_format": "float"}
    last = None
    for k in range(EMBED_MAX_RETRY):
        try:
            r = requests.post(url, headers=hdr, json=body, timeout=180)
            if r.status_code == 200:
                data = r.json()["data"]
                data.sort(key=lambda d: d["index"])       # GIU NGUYEN THU TU
                v = np.asarray([d["embedding"] for d in data], dtype="float32")
                assert v.shape == (len(texts), EMBED_DIM), v.shape
                return v
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(min(2 ** k + random.random(), 60))
    raise RuntimeError(f"embed that bai sau {EMBED_MAX_RETRY} lan: {last}")

def _embed_local(texts: list) -> np.ndarray:
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(LOCAL_EMBED_MODEL, device="cuda")
    v = _ST_MODEL.encode(texts, normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(v, dtype="float32")

def embed_texts(texts: list) -> np.ndarray:
    """-> float32 [len(texts), EMBED_DIM]. Doi provider KHONG duoc sua code downstream."""
    texts = [(t if (t and str(t).strip()) else " ") for t in texts]   # API tu choi chuoi rong
    if EMBED_PROVIDER in ("openai", "openrouter"):
        return _embed_http(texts)
    if EMBED_PROVIDER == "local":
        return _embed_local(texts)
    raise ValueError(EMBED_PROVIDER)

# ============================================================
#  SMOKE TEST - thiet ke cho "Save & Version" (chay khong nguoi truc)
# ============================================================
# Chi HARD-FAIL khi model that su HONG (sai shape / NaN / API loi). Chat luong VI-EN
# la tin hieu MEM -> chi canh bao + ghi vao manifest, KHONG SystemExit, vi giet mot
# batch run 3h chi vi mot nguong mem thi thiet hai hon nhieu.
# Muon no dung han thi dat EMBED_ON_LOW_QUALITY = "abort".

PAIRS_ALIGNED = [                      # cung nghia -> cos phai CAO
    ("a red car on the street",            "một chiếc xe đỏ trên phố"),
    ("lion dance performance on poles",    "múa lân trên cột"),
    ("a temple festival in Kien Giang",    "lễ hội đình thần ở Kiên Giang"),
]
PAIRS_UNRELATED = [                    # khac nghia -> cos phai THAP
    ("a red car on the street",            "công thức nấu món phở bò"),
    ("lion dance performance on poles",    "giá vàng hôm nay tăng mạnh"),
]

def quality_probe(embed_fn) -> dict:
    """Do do TACH BIET, khong chi do cos tho.

    Mot model suy bien cho cos cao voi MOI cap -> nhin rieng cos(en,vi) se bi lua.
    separation = mean(cos cap cung nghia) - mean(cos cap khac nghia) moi la thu
    phan anh model co thuc su hieu noi dung hay khong.
    """
    def cos_of(pairs):
        out = []
        for a, b in pairs:
            v = embed_fn([a, b])
            out.append(float(v[0] @ v[1] / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))))
        return out
    al, un = cos_of(PAIRS_ALIGNED), cos_of(PAIRS_UNRELATED)
    return {"aligned": float(np.mean(al)), "unrelated": float(np.mean(un)),
            "separation": float(np.mean(al) - np.mean(un))}

# ---- 1. Kiem tra CUNG: model co chay duoc khong ----
v = embed_texts(["a red car on the street", "một chiếc xe đỏ trên phố"])
assert v.shape == (2, EMBED_DIM), f"shape {v.shape}, ky vong (2, {EMBED_DIM})"
assert np.isfinite(v).all(), "embedding co NaN/Inf"
print(f"[OK] {EMBED_MODEL} chay duoc, shape={v.shape}")

# ---- 2. Kiem tra MEM: chat luong VI-EN ----
Q = quality_probe(embed_texts)
cos_envi = Q["aligned"]
print(f"\n  cung nghia   : {Q['aligned']:.3f}")
print(f"  khac nghia   : {Q['unrelated']:.3f}")
print(f"  TACH BIET    : {Q['separation']:.3f}   <- day moi la con so quan trong")

if Q["separation"] < 0.05:
    msg = (f"separation={Q['separation']:.3f} qua thap - model gan nhu khong phan biet "
           f"duoc noi dung. Doi EMBED_MODEL sang 'baai/bge-m3' hoac "
           f"'qwen/qwen3-embedding-8b' (deu co tren OpenRouter, deu RE HON).")
    if EMBED_ON_LOW_QUALITY == "abort":
        raise SystemExit(msg)
    print(f"\n[!!! CANH BAO NANG] {msg}")
elif cos_envi < 0.5:
    print(f"\n[WARN] cos cung nghia {cos_envi:.3f} < 0.5 - kenh OCR (embed truc tiep VI) "
          f"se yeu. Can nhac 'baai/bge-m3' (1024-d, $0.01/1M - re hon mot nua).")
else:
    print(f"\n[OK] chat luong VI-EN dat. Chay tiep Cell 10 duoc.")

# %% [markdown] bench
# ## Cell 9b — Benchmark model embedding (tuỳ chọn, ~vài chục token)
#
# Trong chế độ **Save & Version** bạn không dừng lại quyết định được. Cell này chạy cùng
# bộ cặp VI–EN qua vài model ứng viên rồi in bảng so sánh — **không đổi model đang dùng**,
# chỉ cho bạn số liệu để chọn cho version sau.
#
# Cột **TACH BIET** (`cos(cùng nghĩa) − cos(khác nghĩa)`) mới là cột đáng nhìn: một model
# suy biến cho `cos` cao với *mọi* cặp, nhìn riêng `cos(en,vi)` sẽ bị đánh lừa.

# %% Cell 9b — benchmark cac model embedding ung vien
if EMBED_BENCHMARK and EMBED_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
    _base, _key, _ = embed_endpoint()

    def _probe_model(model_id):
        def f(texts):
            hdr = {"Authorization": f"Bearer {_key}", "Content-Type": "application/json"}
            r = requests.post(f"{_base}/embeddings", headers=hdr, timeout=120,
                              json={"model": model_id, "input": texts,
                                    "encoding_format": "float"})
            r.raise_for_status()
            d = sorted(r.json()["data"], key=lambda x: x["index"])
            return np.asarray([x["embedding"] for x in d], dtype="float32")
        return f

    print(f"{'model':38s} {'dim':>5s} {'$/1M':>6s} {'cung':>6s} {'khac':>6s} {'TACH BIET':>10s}")
    print("-" * 78)
    for mid, dim, usd in EMBED_CANDIDATES:
        try:
            q = quality_probe(_probe_model(mid))
            mark = "  <-- dang dung" if mid.split("/")[-1] == EMBED_MODEL else ""
            print(f"{mid:38s} {dim:5d} {usd:6.3f} {q['aligned']:6.3f} "
                  f"{q['unrelated']:6.3f} {q['separation']:10.3f}{mark}")
        except Exception as e:
            print(f"{mid:38s}  LOI: {str(e)[:34]}")
    print("-" * 78)
    print("Doi model: sua EMBED_MODEL + EMBED_DIM o Cell CONFIG roi chay lai TU DAU")
    print("(index cu khong dung lai duoc - khac model = khac khong gian vector)")
else:
    print("[SKIP] benchmark (EMBED_BENCHMARK=False, khong phai openrouter, hoac chua co key)")

# %% [markdown] phase-b
# ## Cell 10 — PHASE_B: embed toàn corpus (~319K unit, 1.5–3h)
#
# **Resumable là điều kiện sống còn** — Kaggle timeout 12h khi "Save & Version".
# Chạy lại cell lần 2 phải skip 100% và xong trong < 1 phút.
#
# Thứ tự channel: `meta` (873) → `summary` (5K) → `asr` (23K) → `ocr` (128K) → `caption` (146K).
# Từ nhỏ đến lớn để phát hiện lỗi sớm với chi phí thấp nhất.
#
# ---
#
# ### ⚠️ Cạm bẫy của "Save & Version": run FAIL = mất sạch checkpoint
#
# Kaggle **chỉ lưu `/kaggle/working` khi run kết thúc thành công**. Nếu batch run bị
# timeout hoặc lỗi ở giữa Phase B, toàn bộ checkpoint bay hết → lần sau embed lại từ
# đầu, **trả tiền hai lần**. Cơ chế resumable ở dưới chỉ tự cứu được khi bạn chạy
# **tương tác** (interactive session giữ nguyên `/kaggle/working` giữa các lần chạy cell).
#
# **Ba cách xử lý, chọn một:**
#
# | Cách | Làm gì | Khi nào dùng |
# |:--|:--|:--|
# | **A. Chạy tương tác Phase B** (khuyến nghị lần đầu) | Mở interactive session, chạy tới Cell 10, đợi xong, rồi mới `Save & Version` | An toàn nhất. Chạy lại cell = skip phần đã xong |
# | **B. Chuỗi checkpoint dataset** | Version 1 embed được bao nhiêu hay bấy nhiêu → publish `/kaggle/working/ckpt` thành dataset `aic26-ckpt` → attach lại + điền `CKPT_INPUT_OWNER/SLUG` → Version 2 chạy tiếp | Khi buộc phải chạy batch |
# | **C. Tách NB01A / NB01B** | NB01A = Phase A+D+E (CPU, ~1h). NB01B = Phase B+C, chain dataset từ NB01A | Nếu tổng thời gian chạm mốc 12h |
#
# Với `EMBED_PROVIDER="openrouter"`, Phase B ước tính **1.5–3h** nên **cách A** là đủ và
# đơn giản nhất — không chạm giới hạn 12h.

# %% Cell 10 — PHASE_B: embed resumable
from concurrent.futures import ThreadPoolExecutor, as_completed

EMBED_ORDER = ["meta", "summary", "asr", "ocr", "caption"]

# ---- Seed checkpoint tu dataset (chua cho truong hop "Save & Version" bi fail) ----
# Kaggle CHI luu /kaggle/working khi run ket thuc THANH CONG. Run bi timeout/fail thi
# mat sach checkpoint -> lan sau embed lai tu dau, ton tien va thoi gian lan hai.
if CKPT_INPUT_SLUG:
    import shutil as _sh
    try:
        _src = resolve(CKPT_INPUT_OWNER, CKPT_INPUT_SLUG)
        _n = 0
        for _p in glob.glob(f"{_src}/**/*.npy", recursive=True):
            _rel = os.path.relpath(_p, _src)
            _dst = os.path.join(CKPT, _rel)
            if not os.path.exists(_dst):
                os.makedirs(os.path.dirname(_dst), exist_ok=True)
                _sh.copyfile(_p, _dst)
                _n += 1
        print(f"[RESUME] seed {_n} file checkpoint tu {CKPT_INPUT_OWNER}/{CKPT_INPUT_SLUG}")
    except FileNotFoundError as _e:
        print(f"[WARN] khong tim thay dataset checkpoint: {_e}")
else:
    print("[INFO] CKPT_INPUT_SLUG rong -> embed tu dau. Neu run nay fail giua chung, "
          "xem huong dan resume o markdown cell ngay tren.")
USD_PER_1M_TOKEN = 0.02          # gia OpenAI cho text-embedding-3-small.
                                 # Qua OpenRouter co the co markup -> con so USD in ra chi
                                 # la UOC LUONG; xem so thuc te o openrouter.ai/activity
_tok_est = lambda s: max(1, len(s) // 4)

def embed_channel(channel: str):
    """Embed 1 channel voi checkpoint theo batch. Bo qua batch da co file .npy."""
    sub = units.loc[(units.channel == channel) & (units.emb_row == -3),
                    ["unit_id", "text_embed"]].reset_index(drop=True)
    n = len(sub)
    if n == 0:
        print(f"  [{channel}] 0 unit can embed")
        return
    cdir = f"{CKPT}/{channel}"
    os.makedirs(cdir, exist_ok=True)

    batches = [(b, i, min(i + EMBED_BATCH, n))
               for b, i in enumerate(range(0, n, EMBED_BATCH))]
    todo = [(b, i, j) for (b, i, j) in batches if not os.path.exists(f"{cdir}/{b:05d}.npy")]
    print(f"  [{channel}] {n:,} unit · {len(batches)} batch · con lai {len(todo)}")
    if not todo:
        return

    done, tok_total, t0 = 0, 0, time.time()

    def run(task):
        b, i, j = task
        texts = sub["text_embed"].iloc[i:j].astype(str).tolist()
        v = embed_texts(texts)
        np.save(f"{cdir}/{b:05d}.npy", v.astype("float32"))
        np.save(f"{cdir}/{b:05d}.ids.npy", sub["unit_id"].iloc[i:j].to_numpy().astype(object),
                allow_pickle=True)
        return sum(_tok_est(t) for t in texts)

    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        futs = {ex.submit(run, t): t for t in todo}
        for fu in as_completed(futs):
            tok_total += fu.result()
            done += 1
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                eta = el / done * (len(todo) - done)
                print(f"    {done}/{len(todo)} batch · ~{tok_total:,} tok "
                      f"· ~${tok_total/1e6*USD_PER_1M_TOKEN:.3f} · {el:.0f}s · ETA {eta/60:.0f}m")

for ch in EMBED_ORDER:
    embed_channel(ch)

_n_ck = sum(len(glob.glob(f"{CKPT}/{c}/*.npy")) - len(glob.glob(f"{CKPT}/{c}/*.ids.npy"))
            for c in EMBED_ORDER)
print(f"\nPHASE_B xong. Chay lai cell nay lan 2 phai skip 100% trong < 1 phut.")

# %% [markdown] phase-c
# ## Cell 11 — PHASE_C: build 5 `text_*.faiss` + patch caption duplicate
#
# Patch dup: `duplicate_of` là **tên file** (`"009.jpg"`) → `int(Path(dup).stem)` rồi dựng `f"{vid}#{n:03d}"`.
# Sau bước này **không còn `emb_row == -2`** nào.

# %% Cell 11 — PHASE_C: text_*.faiss
t0 = time.time()
FAISS_NTOTAL = {}
emb_row_map = {}          # unit_id -> emb_row

for ch in EMBED_ORDER:
    cdir = f"{CKPT}/{ch}"
    parts = sorted(p for p in glob.glob(f"{cdir}/*.npy") if not p.endswith(".ids.npy"))
    if not parts:
        print(f"  [{ch}] khong co checkpoint - BO QUA")
        continue
    V   = np.vstack([np.load(p) for p in parts]).astype("float32")   # zero-pad 5 -> sort chuoi dung
    IDS = np.concatenate([np.load(p.replace(".npy", ".ids.npy"), allow_pickle=True) for p in parts])

    expect = int(((units.channel == ch) & (units.emb_row == -3)).sum())
    assert len(V) == len(IDS) == expect, f"[{ch}] {len(V)} vector / {len(IDS)} id / ky vong {expect}"

    faiss.normalize_L2(V)
    idx = faiss.IndexFlatIP(EMBED_DIM)
    idx.add(V)
    faiss.write_index(idx, f"{OUT_FAI}/text_{ch}.faiss")
    np.save(f"{OUT_FAI}/text_{ch}_rowmap.npy", IDS.astype(object), allow_pickle=True)
    FAISS_NTOTAL[ch] = int(idx.ntotal)
    for r, uid in enumerate(IDS):
        emb_row_map[uid] = r
    print(f"  text_{ch}.faiss: ntotal={idx.ntotal:,}")
    del V

# ---- Gan emb_row ----
units["emb_row"] = units["unit_id"].map(emb_row_map).fillna(units["emb_row"]).astype("int32")
units.loc[units.emb_row == -3, "emb_row"] = -1          # -3 chua embed duoc -> -1

# ---- Patch caption duplicate (-2 -> emb_row cua canonical) ----
cap = units[units.channel == "caption"]
kfid2row = dict(zip(cap.kf_id, cap.emb_row))
dup_mask = units.emb_row == -2
n_dup = int(dup_mask.sum())
patched = units.loc[dup_mask, "dup_of"].map(lambda k: kfid2row.get(k, -1))
patched = patched.where(patched >= 0, -1)
units.loc[dup_mask, "emb_row"] = patched.astype("int32")
n_fail = int((units.loc[dup_mask, "emb_row"] < 0).sum())

assert not (units.emb_row == -2).any(), "van con emb_row == -2"
assert not (units.emb_row == -3).any(), "van con emb_row == -3"

# emb_row hop le phai < ntotal cua channel do
for ch, nt in FAISS_NTOTAL.items():
    bad = units[(units.channel == ch) & (units.emb_row >= nt)]
    assert bad.empty, f"[{ch}] {len(bad)} unit co emb_row >= ntotal={nt}"

units.to_parquet(f"{OUT}/text_units.parquet", index=False)
N_EMBEDDED = {ch: int(nt) for ch, nt in FAISS_NTOTAL.items()}
print(f"\npatch dup: {n_dup:,} dong ({n_fail} khong tim thay canonical -> -1) · {time.time()-t0:.0f}s")

# %% Cell 12 — MANIFEST
from datetime import datetime, timezone

manifest = {
    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "embed_provider": EMBED_PROVIDER,
    # Ghi ten CANONICAL (khong prefix provider): "text-embedding-3-small" du di qua
    # openai hay openrouter. Cung mot model = cung khong gian vector, nen NB02 doi
    # provider van dung index nay duoc - guard o NB02 so sanh theo ten canonical.
    "embed_model": LOCAL_EMBED_MODEL if EMBED_PROVIDER == "local" else EMBED_MODEL,
    "embed_dim": int(EMBED_DIM),
    "visual_model": VISUAL_MODEL,
    "visual_dim": int(VISUAL_DIM),
    "n_videos": int(len(videos)),
    "n_keyframes": int(len(kf)),
    "n_text_units": {c: int((units.channel == c).sum()) for c in EMBED_ORDER},
    "n_embedded": N_EMBEDDED,
    "asr_window_sec": ASR_WINDOW_SEC,
    "asr_stride_sec": ASR_STRIDE_SEC,
    "asr_dedup_by_segment_set": True,
    "ocr_min_len": OCR_MIN_LEN,
    "object_score_threshold": OBJECT_SCORE_THRESHOLD,
    "n_object_classes": N_OBJECT_CLASSES,
    "tokenizer_vi": TOKENIZER_VI,
    "bm25_sizes": BM25_SIZES,
    # Ghi ca 3 con so de sau nay doi chieu duoc giua cac lan build khac model
    "quality_probe": {k: round(v, 4) for k, v in Q.items()},
    "videos_missing_ocr": VIDEOS_MISSING_OCR,
    "videos_missing_summary": VIDEOS_MISSING_SUMMARY,
    "videos_keyframe_count_mismatch": KF_MISMATCH,
}
with open(f"{OUT}/BUILD_MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in manifest.items()
                  if k not in ("videos_missing_ocr", "videos_missing_summary")},
                 ensure_ascii=False, indent=2))

# %% [markdown] self-test
# ## Cell 13 — `SELF_TEST` (Definition of Done)
#
# 12 assertion + 3 truy vấn thử. **Nếu 3 truy vấn thử fail thì đừng chạy NB02** — sửa index trước.

# %% Cell 13a — SELF_TEST: 12 assertion
def _size_gb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9

checks = []
def chk(i, desc, cond):
    checks.append((i, desc, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {i:2d}. {desc}")

chk(1,  "len(keyframes) == 177321 va kf_id unique",
        len(kf) == N_KEYFRAMES_EXPECT and kf.kf_id.is_unique)
chk(2,  "frame_idx dtype int32, khong NaN",
        str(kf.frame_idx.dtype) == "int32" and kf.frame_idx.notna().all())
chk(3,  "vision.faiss.ntotal == len(keyframes)",
        faiss.read_index(f"{OUT_FAI}/vision.faiss").ntotal == len(kf))
chk(4,  "moi channel: ntotal == so unit da embed",
        all(FAISS_NTOTAL[c] == int(((units.channel == c) & (units.emb_row >= 0)
            & (units.dup_of.isna())).sum()) for c in FAISS_NTOTAL))
chk(5,  "moi emb_row >= 0 deu < ntotal cua channel do",
        all(units[(units.channel == c) & (units.emb_row >= 0)].emb_row.max() < nt
            for c, nt in FAISS_NTOTAL.items()))
chk(6,  "6 thu muc BM25 load duoc va len(ids) == n_docs",
        all(os.path.isdir(f"{OUT_BM25}/{n}") and
            len(np.load(f"{OUT_BM25}/{n}/ids.npy", allow_pickle=True)) == BM25_SIZES[n]
            for n, *_ in BM25_SPEC))
chk(7,  "objects_matrix.shape == (177321, 584)",
        OBJ_M.shape == (N_KEYFRAMES_EXPECT, N_OBJECT_CLASSES))
_gb = _size_gb(OUT)
chk(8,  f"index/ < 20 GB (dang la {_gb:.2f} GB)", _gb < 20)
_ocls = open(f"{OUT}/object_classes.txt", encoding="utf-8").read().splitlines()
chk(9,  "object_classes.txt = 584 dong sach (khong '#', khong '\\r')",
        len(_ocls) == 584 and not any(l.startswith("#") or "\r" in l for l in _ocls))
chk(10, "videos khong co '_failed' va has_summary.sum() == 865",
        "_failed" not in set(videos.video_id) and int(videos.has_summary.sum()) == 865)
chk(11, "text_units[channel=='ocr'] gan 128,664 dong",
        abs(int((units.channel == 'ocr').sum()) - N_OCR_LINES_EXPECT) <= 2000)
chk(12, "khong 2 unit asr nao cung (video_id, text_embed) - dedupe da chay",
        not units[units.channel == 'asr'].duplicated(subset=['video_id', 'text_embed']).any())

n_fail = sum(1 for _, _, ok in checks if not ok)
print(f"\n{len(checks)-n_fail}/{len(checks)} PASS")
assert n_fail == 0, f"{n_fail} check FAIL - sua truoc khi publish dataset"

# %% Cell 13b — SELF_TEST: 3 truy van thu (nhin bang MAT)
UNIT_TEXT = dict(zip(units.unit_id, units.text_embed))
UNIT_VID  = dict(zip(units.unit_id, units.video_id.astype(str)))

def try_bm25(name, query_tokens, k=10):
    r = bm25s.BM25.load(f"{OUT_BM25}/{name}", load_corpus=False)
    ids = np.load(f"{OUT_BM25}/{name}/ids.npy", allow_pickle=True)
    # bm25s nhan list-of-list-of-token; dung np.array(dtype=object) de bi suy ra 2D
    res, sc = r.retrieve([list(query_tokens)], k=min(k, len(ids)))
    return [(ids[i], float(s)) for i, s in zip(res[0], sc[0])]

print("=" * 70)
print("[1] bm25_ocr_vi  <- 'FANA'   (chinh la query-p1-15-qa)")
for uid, s in try_bm25("bm25_ocr_vi", tok_vi("FANA")):
    print(f"    {s:6.2f}  {uid:28s}  {str(UNIT_TEXT.get(uid))[:70]}")

print("\n[2] vision.faiss <- CLIP text 'lion dance on poles'")
try:
    from sentence_transformers import SentenceTransformer
    _clip = SentenceTransformer(VISUAL_MODEL)
    qv = _clip.encode(["lion dance on poles"], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)
    D, I = faiss.read_index(f"{OUT_FAI}/vision.faiss").search(qv, 10)
    for d, i in zip(D[0], I[0]):
        print(f"    {d:.3f}  {kf.kf_id.iloc[i]}")
    print("    -> MO ANH RA XEM: top-10 phai la anh mua lan")
except Exception as e:
    print(f"    [SKIP] khong load duoc {VISUAL_MODEL}: {e}")

print("\n[3] bm25_asr_vi + bm25_meta <- 'Nguyen Trung Truc'  (query-p1-19-qa)")
for name in ("bm25_asr_vi", "bm25_meta"):
    print(f"  {name}:")
    for uid, s in try_bm25(name, tok_vi("Nguyễn Trung Trực"), k=5):
        print(f"    {s:6.2f}  {UNIT_VID.get(uid):10s}  {str(UNIT_TEXT.get(uid))[:60]}")
print("=" * 70)
print("Neu 3 truy van nay khong hop ly -> DUNG chay NB02, sua index truoc.")

# %% [markdown] publish
# ## Publish dataset
#
# `Save & Version` → tab `Data` → `New Dataset` từ `/kaggle/working/index`, tên **`aic26-index`**, private.
#
# Version sau thì dùng `New Version` **cùng dataset** để NB02 không phải sửa path.
#
# ```
# aic26-index/
# ├── keyframes.parquet · videos.parquet · text_units.parquet
# ├── objects.parquet · objects_matrix.npz · object_classes.txt
# ├── faiss/   vision.faiss + 5 text_*.faiss + 6 *_rowmap.npy
# ├── bm25/    6 thư mục bm25s + ids.npy
# └── BUILD_MANIFEST.json
# ```

# %% Cell 14 — liet ke output truoc khi publish
total = 0
for p in sorted(Path(OUT).rglob("*")):
    if p.is_file():
        sz = p.stat().st_size
        total += sz
        print(f"  {sz/1e6:10.1f} MB  {p.relative_to(OUT)}")
print(f"\nTONG: {total/1e9:.2f} GB  ->  publish thanh Kaggle Dataset 'aic26-index' (private)")
