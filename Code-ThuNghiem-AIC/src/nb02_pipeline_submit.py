# %% [markdown] title
# # NB02 — END-TO-END PIPELINE & SUBMISSION (AIC 2026)
#
# Spec: `Planning/03_NB02_PIPELINE_SUBMIT.md` · Contracts: `Planning/01_DATA_CONTRACTS.md` · Paths: `Planning/05_KAGGLE_PATHS.md`
#
# **Input datasets (4):**
# `kitnehi1211/aic26-index` (output NB01) · `kitnehi1211/feature-aic-2026` ·
# `kitnehi1211/dethithunghiem` (gói đề) · `fatle542/aic-dataset` (ảnh keyframe, 115.75 GB)
#
# **Output:** `/kaggle/working/team_XXX_roundN.zip` chứa thư mục `submission/`
# **Accelerator:** `GPU T4 x2` (cho `RERANK_MODE="local"`) · **Internet:** BẮT BUỘC ON · **Thời gian:** 1–3h / gói 24 query
#
# ```
# Stage 0  query understanding (LLM)        -> parsed_queries.json
# Stage 1  12-channel retrieval             -> ranks
# Stage 2  RRF fusion + video prior + object bonus -> top-1000
# Stage 3  text rerank (Qwen3-Reranker)     -> top-100
# Stage 4  VLM rerank + QA answer           -> top-20
# KIS / QA / TRAKE  -> WRITER -> VALIDATOR (12 check) -> ZIP
# ```

# %% Cell 1 — SETUP (pip install)
!pip install -q bm25s faiss-cpu pyvi PyStemmer pyarrow sentence-transformers

# %% Cell 1b — CONFIG (copy tu Planning/00_MASTER_PLAN.md §5)
# ============================================================
#  CONFIG - user tu dien API key, KHONG commit key vao git
# ============================================================
OPENROUTER_API_KEY = ""   # MiMo-V2.5 (query rewrite, VLM rerank, QA answer) VA embedding
OPENAI_API_KEY     = ""   # chi can khi EMBED_PROVIDER="openai"

# ---- Text embedding - PHAI khop BUILD_MANIFEST cua NB01 ----
# OpenRouter DA co /api/v1/embeddings (verify 2026-08-21) -> chi can 1 key cho ca pipeline.
EMBED_PROVIDER    = "openrouter"              # "openrouter" | "openai" | "local"
EMBED_MODEL       = "text-embedding-3-small"  # ten canonical, KHONG kem prefix provider
EMBED_DIM         = 1536
OPENAI_BASE_URL   = "https://api.openai.com/v1"
LOCAL_EMBED_MODEL = "BAAI/bge-m3"
OR_SITE_URL       = ""                        # optional
OR_SITE_NAME      = "AIC2026"                 # optional
EMBED_MAX_RETRY   = 6

# ---- Visual embedding (QD-2) ----
VISUAL_MODEL = "clip-ViT-B-32"
VISUAL_DIM   = 512

# ---- LLM / VLM qua OpenRouter ----
# Da verify 2026-08-21 qua GET /api/v1/models (public):
#   xiaomi/mimo-v2.5      -> ["text","audio","image","video"]  NHAN ANH  <-- dung cai nay
#   xiaomi/mimo-v2.5-pro  -> ["text"]                          TEXT-ONLY <-- BAY, dung doi sang
# Fallback da xac nhan nhan anh: qwen/qwen3-vl-{8b,30b-a3b,32b,235b-a22b}-{instruct,thinking},
#                                google/gemini-2.5-flash, google/gemini-2.5-flash-lite
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "xiaomi/mimo-v2.5"                # query understanding
VLM_MODEL = "xiaomi/mimo-v2.5"                # final rerank + QA answer
VLM_IMAGE_MAX_SIDE = 768

# ---- Reranker (QD-3) ----
RERANK_MODE        = "local"                   # "local" | "openrouter_llm"
RERANK_MODEL_LOCAL = "Qwen/Qwen3-Reranker-4B"  # 8B fp16 = 16GB -> KHONG vua T4

def embed_endpoint():
    """-> (base_url, api_key, model_id_gui_di). OpenRouter can prefix 'openai/'."""
    if EMBED_PROVIDER == "openrouter":
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY, f"openai/{EMBED_MODEL}"
    if EMBED_PROVIDER == "openai":
        return OPENAI_BASE_URL, OPENAI_API_KEY, EMBED_MODEL
    return None, None, LOCAL_EMBED_MODEL

# ---- Retrieval / Fusion ----
TOPK_PER_CHANNEL = 500
TOPK_FUSED       = 1000
TOPK_TEXT_RERANK = 100
TOPK_VLM_RERANK  = 20
RRF_K            = 60
MAX_ROWS_PER_CSV = 100

CHANNEL_WEIGHTS = {          # tune tren bo de thu (kitnehi1211/dethithunghiem)
    "vision": 1.00, "caption": 0.90, "ocr": 0.70,
    "asr": 0.60, "summary_prior": 0.50, "meta_prior": 0.35,
    "bm25_ocr_vi": 0.80, "bm25_caption_en": 0.50,
    "bm25_asr_vi": 0.55, "bm25_asr_en": 0.45, "bm25_meta": 0.40,
    "bm25_summary_en": 0.40,
    "object_bonus": 0.30,
}
W = CHANNEL_WEIGHTS

# ---- TRAKE ----
TRAKE_TOP_VIDEOS = 5      # M
TRAKE_BEAM       = 5      # B duong di / video
TRAKE_ABC        = (0.50, 0.35, 0.15)   # (vision, caption, ocr) trong score matrix

# ---- Nop bai ----
TEAM_NAME = "team_XXX"    # <-- DOI THANH TEN DOI THAT
ROUND     = 1

# ---- Co dieu khien ----
SKIP_VLM       = False    # tu bat True neu khong tim duoc anh keyframe
RELOAD_PARSED  = False    # True = doc lai parsed_queries.json da sua tay, khong goi LLM

WORK = "/kaggle/working"
print(f"LLM={LLM_MODEL} · VLM={VLM_MODEL} · RERANK_MODE={RERANK_MODE}")

# %% Cell 1c — PATHS + resolve() (copy tu Planning/05_KAGGLE_PATHS.md §2-4)
import os, json, glob, re, io, sys, time, math, random, base64, unicodedata, csv, shutil, zipfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

KG = "/kaggle/input"

def resolve(owner: str, slug: str, inner: str = "") -> str:
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

INDEX_ROOT  = resolve("kitnehi1211", "aic26-index")
QUERY_ROOT  = resolve("kitnehi1211", "dethithunghiem")
try:
    KEYFRAME_DS = resolve("fatle542", "aic-dataset")
except FileNotFoundError as e:
    print(f"[WARN] {e}\n-> SKIP_VLM = True")
    KEYFRAME_DS, SKIP_VLM = None, True

SUB_DIR = f"{WORK}/submission"
DBG_DIR = f"{WORK}/debug"
for _d in (SUB_DIR, DBG_DIR):
    os.makedirs(_d, exist_ok=True)

def nfc(s) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s)).strip()

print("INDEX_ROOT  =", INDEX_ROOT)
print("QUERY_ROOT  =", QUERY_ROOT)
print("KEYFRAME_DS =", KEYFRAME_DS)

# %% [markdown] preflight
# ## Cell 2 — `PREFLIGHT` (đừng bỏ qua)
#
# Ba thứ chặn cả pipeline nếu sai, kiểm hết ở đây với chi phí ~30 giây:
# **(1)** manifest guard — index build bằng model nào (QĐ-2) · **(2)** keyframe path discovery (QĐ-4, R2) ·
# **(3)** VLM model có tồn tại và **thật sự nhận ảnh** không (R1).

# %% Cell 2a — PREFLIGHT 1/3: manifest guard (QD-2)
mf = json.load(open(f"{INDEX_ROOT}/BUILD_MANIFEST.json", encoding="utf-8"))

def _canon(m: str) -> str:
    """Bo prefix provider: 'openai/text-embedding-3-small' -> 'text-embedding-3-small'.

    Cung mot model qua openai hay openrouter la CUNG khong gian vector, nen doi
    provider giua NB01 va NB02 la an toan. Chi khac model moi phai dung.
    """
    return str(m).split("/")[-1]

_mf_embed = LOCAL_EMBED_MODEL if EMBED_PROVIDER == "local" else EMBED_MODEL
assert _canon(mf["embed_model"]) == _canon(_mf_embed), (
    f"Index build bang '{mf['embed_model']}' nhung CONFIG dat '{_mf_embed}'. "
    "Hai khong gian vector khac nhau -> DUNG NGAY.")
if mf.get("embed_provider") != EMBED_PROVIDER:
    print(f"[INFO] NB01 embed qua '{mf.get('embed_provider')}', NB02 dung '{EMBED_PROVIDER}' "
          f"- cung model '{_canon(_mf_embed)}' nen cung khong gian vector, OK.")
assert mf["visual_model"] == VISUAL_MODEL, f"Index build bang '{mf['visual_model']}'"

EMBED_DIM  = int(mf["embed_dim"])     # lay tu manifest, KHONG tin CONFIG
VISUAL_DIM = int(mf["visual_dim"])
TOK_VI     = mf.get("tokenizer_vi", "pyvi")
print(f"manifest OK · embed={mf['embed_model']}({EMBED_DIM}d) "
      f"visual={mf['visual_model']}({VISUAL_DIM}d) tokenizer_vi={TOK_VI}")
print(f"  built_at={mf['built_at']} · {mf['n_keyframes']:,} kf · thieu OCR {len(mf['videos_missing_ocr'])} video")

# %% Cell 2b — PREFLIGHT 2/3: keyframe path discovery (QD-4, R2)
def build_kf_index(keyframe_ds: str, cache=f"{WORK}/kf_dir_map.json") -> dict:
    """video_id -> thu muc chua anh keyframe. Quet 1 lan roi cache.

    Pattern da verify: Keyframes_L21/keyframes/L21_V001/001.jpg
    Ten batch dir cho L26 CHUA verify (co the Keyframes_L26_a...e) nen glob o muc
    */keyframes/* de tu xu ly moi cach dat ten.
    DUNG glob xuong toi .jpg: 177K file tren dataset 115GB se treo rat lau.
    """
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    m = {}
    for d in glob.glob(f"{keyframe_ds}/*/keyframes/*"):
        if os.path.isdir(d):
            m[os.path.basename(d)] = d
    json.dump(m, open(cache, "w", encoding="utf-8"))
    return m

KF_DIR = {}
if KEYFRAME_DS and not SKIP_VLM:
    t0 = time.time()
    KF_DIR = build_kf_index(KEYFRAME_DS)
    print(f"KF_DIR: {len(KF_DIR)}/873 video · {time.time()-t0:.0f}s")
    if len(KF_DIR) < 873:
        print("[WARN] thieu anh cho cac video sau (khong VLM-rerank duoc):")
        # danh sach day du in o Cell 3 sau khi co keyframes.parquet
    if len(KF_DIR) == 0:
        SKIP_VLM = True
        print("[WARN] KHONG tim thay anh nao -> SKIP_VLM=True. "
              "Stage 0-3 + KIS/TRAKE van chay (kem hon), QA gan nhu chac chan sai.")

def kf_image_path(video_id: str, n: int) -> str:
    return f"{KF_DIR[video_id]}/{int(n):03d}.jpg"

# %% Cell 2c — PREFLIGHT 3/3: model availability (R1)
import requests

def _or_headers():
    return {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

if OPENROUTER_API_KEY:
    r = requests.get(f"{OPENROUTER_BASE_URL}/models", headers=_or_headers(), timeout=60)
    ids = {m["id"] for m in r.json()["data"]}
    for tag, mid in (("LLM", LLM_MODEL), ("VLM", VLM_MODEL)):
        if mid not in ids:
            print(f"[ERROR] {tag}_MODEL='{mid}' KHONG co tren OpenRouter.")
            print("  Ung vien vision thay the:")
            print("   ", [i for i in sorted(ids)
                          if any(k in i for k in ("vl", "vision", "gemini", "mimo"))][:40])
            raise SystemExit("Chon lai model roi chay lai (giu no la CONFIG, dung hard-code)")
        print(f"  {tag}_MODEL='{mid}' OK")
else:
    print("[WARN] chua dien OPENROUTER_API_KEY - bo qua check model")

# %% Cell 2d — PREFLIGHT: xac nhan VLM THAT SU nhan anh
# Nhieu model text-only van tra 200 roi ignore anh -> phai thu bang anh that.
from PIL import Image

def _png_b64_solid(color=(255, 0, 0), size=(64, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

if OPENROUTER_API_KEY and not SKIP_VLM:
    body = {
        "model": VLM_MODEL, "temperature": 0, "max_tokens": 20,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What colour fills this image? Answer with one word."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_png_b64_solid()}"}},
        ]}],
    }
    r = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions",
                      headers=_or_headers(), json=body, timeout=120)
    if r.status_code != 200:
        print(f"[ERROR] VLM tra HTTP {r.status_code}: {r.text[:300]}")
        raise SystemExit("VLM_MODEL khong nhan vision input - chon model khac")
    ans = r.json()["choices"][0]["message"]["content"]
    print(f"  VLM vision test -> {ans!r}  (ky vong co chu 'red')")
    if "red" not in ans.lower():
        print("[WARN] model co the dang IGNORE anh. Kiem tra lai truoc khi chay Stage 4.")

# %% [markdown] load-index
# ## Cell 3 — `LOAD_INDEX`
#
# Load hết vào RAM (Kaggle 30GB, index ~2.5GB nên thoải mái). Mục tiêu: < 3 phút, RAM < 12GB.

# %% Cell 3 — LOAD_INDEX
import faiss, bm25s
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

t0 = time.time()
kf     = pd.read_parquet(f"{INDEX_ROOT}/keyframes.parquet")
videos = pd.read_parquet(f"{INDEX_ROOT}/videos.parquet")
units  = pd.read_parquet(f"{INDEX_ROOT}/text_units.parquet")

kf = kf.reset_index(drop=True)
KF_LIST   = kf.kf_id.to_numpy()
KF2ROW    = {k: i for i, k in enumerate(KF_LIST)}          # kf_id -> vis_row
KF2FIDX   = dict(zip(kf.kf_id, kf.frame_idx.astype(int)))  # GIA TRI NOP BAI
KF2VID    = dict(zip(kf.kf_id, kf.video_id.astype(str)))
KF2PTS    = dict(zip(kf.kf_id, kf.pts_time.astype(float)))
KF2N      = dict(zip(kf.kf_id, kf.n.astype(int)))
VID_FIDX_SET = {v: set(g.astype(int)) for v, g in kf.groupby(kf.video_id.astype(str))["frame_idx"]}

# keyframe cua tung video, da sort theo n
KF_BY_VIDEO = {}
for vid, g in kf.groupby(kf.video_id.astype(str), sort=False):
    KF_BY_VIDEO[vid] = (g.kf_id.to_numpy(), g.pts_time.to_numpy(dtype="float32"),
                        g.vis_row.to_numpy(dtype="int64"))

TEXT_CH = ["caption", "ocr", "asr", "summary", "meta"]
FAISS_IDX = {"vision": faiss.read_index(f"{INDEX_ROOT}/faiss/vision.faiss")}
ROWMAP    = {"vision": np.load(f"{INDEX_ROOT}/faiss/vision_rowmap.npy", allow_pickle=True)}
for c in TEXT_CH:
    FAISS_IDX[c] = faiss.read_index(f"{INDEX_ROOT}/faiss/text_{c}.faiss")
    ROWMAP[c]    = np.load(f"{INDEX_ROOT}/faiss/text_{c}_rowmap.npy", allow_pickle=True)

BM25_NAMES = ["bm25_caption_en", "bm25_ocr_vi", "bm25_asr_vi",
              "bm25_asr_en", "bm25_summary_en", "bm25_meta"]
BM25, BM25_IDS = {}, {}
for n in BM25_NAMES:
    d = f"{INDEX_ROOT}/bm25/{n}"
    BM25[n]     = bm25s.BM25.load(d, load_corpus=False)
    BM25_IDS[n] = np.load(f"{d}/ids.npy", allow_pickle=True)

OBJ_M = sp.load_npz(f"{INDEX_ROOT}/objects_matrix.npz").tocsc()

def load_object_classes(path: str) -> list:
    lines = [l.strip() for l in open(path, encoding="utf-8")]
    return [l for l in lines if l and not l.startswith("#")]

OBJ_CLASSES = load_object_classes(f"{INDEX_ROOT}/object_classes.txt")
OBJ_CLS = {c: i for i, c in enumerate(OBJ_CLASSES)}
assert len(OBJ_CLS) == 584, len(OBJ_CLS)

# ---- unit lookup ----
U_ID   = units.unit_id.to_numpy()
U_CH   = units.channel.astype(str).to_numpy()
U_VID  = units.video_id.astype(str).to_numpy()
U_KFID = units.kf_id.to_numpy()
U_TS   = units.t_start.to_numpy(dtype="float32")
U_TE   = units.t_end.to_numpy(dtype="float32")
U_POS  = {u: i for i, u in enumerate(U_ID)}

# emb_row -> unit_id cho tung channel (dung rowmap cua FAISS)
UNIT_TEXT_EN = dict(zip(units.unit_id, units.text_en))
UNIT_TEXT_VI = dict(zip(units.unit_id, units.text_vi))
# tra cuu nhanh: kf_id -> caption / ocr text (dung cho evidence card)
_cap = units[units.channel == "caption"]
CAP_BY_KF = dict(zip(_cap.kf_id, _cap.text_en))
_ocr = units[units.channel == "ocr"]
OCR_BY_KF = dict(zip(_ocr.kf_id, _ocr.text_vi))
# asr theo video, sort theo t_start (de tim window bao quanh 1 pts_time)
ASR_BY_VIDEO = {}
for vid, g in units[units.channel == "asr"].groupby(units.video_id.astype(str), sort=False):
    g = g.sort_values("t_start")
    ASR_BY_VIDEO[vid] = (g.t_start.to_numpy(dtype="float32"),
                         g.t_end.to_numpy(dtype="float32"),
                         g.text_en.fillna("").to_numpy(),
                         g.text_vi.fillna("").to_numpy())
VID_ROW = videos.set_index("video_id")

# objects theo kf (cho evidence card)
_obj = pd.read_parquet(f"{INDEX_ROOT}/objects.parquet")
OBJ_BY_KF = dict(zip(_obj.kf_id, _obj.classes))
del _obj, _cap, _ocr

clip_txt = SentenceTransformer(VISUAL_MODEL)      # text tower cho kenh visual (QD-2)

if KF_DIR and len(KF_DIR) < 873:
    missing = sorted(set(kf.video_id.astype(str)) - set(KF_DIR))
    print(f"[WARN] {len(missing)} video khong co anh: {missing[:20]}{' ...' if len(missing)>20 else ''}")

print(f"LOAD_INDEX xong: {len(kf):,} kf · {len(units):,} unit · "
      f"{len(FAISS_IDX)} faiss · {len(BM25)} bm25 · {time.time()-t0:.0f}s")

# %% Cell 3b — tokenizer (PHAI khop tokenizer NB01 da dung -> manifest)
if TOK_VI == "pyvi":
    from pyvi import ViTokenizer
else:
    ViTokenizer = None
    print(f"[INFO] NB01 dung tokenizer_vi='{TOK_VI}' -> NB02 dung dung cai do")

def tok_en(s) -> list:
    return re.sub(r"[^\w\s]", " ", (s or "").lower()).split()

def tok_vi(s) -> list:
    s = nfc(s).lower()                    # NFC BAT BUOC - khong thi BM25 miss hoan toan
    if ViTokenizer is None:
        return [w for w in re.split(r"\W+", s) if w]
    try:
        return ViTokenizer.tokenize(s).split()
    except Exception:
        return [w for w in re.split(r"\W+", s) if w]

print("tok_vi('Nguyễn Trung Trực') =", tok_vi("Nguyễn Trung Trực"))

# %% Cell 3c — embed_texts() cho query (cung provider voi NB01)
def _embed_http(texts: list) -> np.ndarray:
    """Dung CHUNG cho openai va openrouter - cung schema /embeddings."""
    base, key, model = embed_endpoint()
    assert key, f"Chua dien API key cho EMBED_PROVIDER='{EMBED_PROVIDER}'"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if EMBED_PROVIDER == "openrouter":
        if OR_SITE_URL:
            hdr["HTTP-Referer"] = OR_SITE_URL
        if OR_SITE_NAME:
            hdr["X-Title"] = OR_SITE_NAME
    body = {"model": model, "input": texts, "encoding_format": "float"}
    last = None
    for k in range(EMBED_MAX_RETRY):
        try:
            r = requests.post(f"{base}/embeddings", headers=hdr, json=body, timeout=120)
            if r.status_code == 200:
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return np.asarray([d["embedding"] for d in data], dtype="float32")
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
            else:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(min(2 ** k + random.random(), 60))
    raise RuntimeError(f"embed that bai: {last}")

_ST_EMB = None
def _embed_local(texts: list) -> np.ndarray:
    global _ST_EMB
    if _ST_EMB is None:
        _ST_EMB = SentenceTransformer(LOCAL_EMBED_MODEL, device="cuda")
    return np.asarray(_ST_EMB.encode(texts, normalize_embeddings=True,
                                     convert_to_numpy=True), dtype="float32")

_EMB_CACHE = {}
def embed_texts(texts: list) -> np.ndarray:
    """Co cache vi 1 query duoc embed lai nhieu lan qua cac channel."""
    texts = [(t if (t and str(t).strip()) else " ") for t in texts]
    miss = [t for t in texts if t not in _EMB_CACHE]
    if miss:
        v = _embed_local(miss) if EMBED_PROVIDER == "local" else _embed_http(miss)
        for t, e in zip(miss, v):
            _EMB_CACHE[t] = e
    out = np.stack([_EMB_CACHE[t] for t in texts]).astype("float32")
    faiss.normalize_L2(out)
    return out

def clip_encode(texts: list) -> np.ndarray:
    v = clip_txt.encode(texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(v)
    return v

# %% [markdown] stage0
# ## Cell 4 — `STAGE0`: Query Understanding
#
# ⚠️ `query-p1-3` **không tồn tại** — số thứ tự query có lỗ. Luôn `glob`, đừng `range(1, 26)`.
#
# ⚠️ `query-p1-18-trake.txt` **BTC đánh máy sai**: các dòng là `E1:`, `E2:`, **`E2:`**, `E4:`.
# Đếm theo **số dòng** ra 4 (đúng); `set()`/dedupe theo con số ra 3 (sai → 0 điểm).
# Thứ tự event **luôn là thứ tự dòng**, không bao giờ sort theo con số sau chữ `E`.

# %% Cell 4a — load_queries (Planning/05_KAGGLE_PATHS.md §5)
EVENT_RE = re.compile(r"^\s*E\s*(\d+)\s*[:.]", flags=re.M)

def load_queries(query_root: str) -> list:
    out = []
    for p in sorted(glob.glob(f"{query_root}/**/*.txt", recursive=True)):
        stem  = os.path.basename(p)[:-4]              # "query-p1-16-trake"
        qtype = stem.rsplit("-", 1)[-1].lower()       # kis | qa | trake
        assert qtype in ("kis", "qa", "trake"), f"hau to la: {stem}"
        q_vi  = nfc(open(p, encoding="utf-8").read())
        out.append({
            "query_id":   stem,
            "query_file": os.path.basename(p),
            "type":       qtype,
            "q_vi":       q_vi,
            # dem SO DONG khop regex, KHONG dedupe/sort theo con so sau chu E
            "n_events":   len(EVENT_RE.findall(q_vi)),
        })
    return out

def event_lines(q_vi: str) -> list:
    """Cac dong event theo DUNG THU TU DONG trong file (khong dedupe, khong sort)."""
    return [l.strip() for l in q_vi.splitlines() if EVENT_RE.match(l)]

QUERIES = load_queries(QUERY_ROOT)
print(f"{len(QUERIES)} query · " + str(dict(Counter(q['type'] for q in QUERIES))))
for q in QUERIES:
    if q["type"] == "trake":
        print(f"  {q['query_id']}: n_events={q['n_events']} · "
              f"so sau chu E = {EVENT_RE.findall(q['q_vi'])}")

# %% Cell 4b — SYSTEM_PROMPT (dung nguyen van - quyet dinh chat luong ca pipeline)
STAGE0_SYSTEM = """You are a query analyst for a Vietnamese TV-news video retrieval system
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
- Keep proper nouns unchanged in q_en (do not anglicise Vietnamese names/places)."""

def or_chat(messages, model=None, temperature=0.0, max_tokens=2048, json_mode=True, retry=5):
    """Goi OpenRouter chat completions, retry 429/5xx."""
    body = {"model": model or LLM_MODEL, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for k in range(retry):
        try:
            r = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions",
                              headers=_or_headers(), json=body, timeout=180)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 404 and json_mode:
                body.pop("response_format", None)   # model khong ho tro json_object
                json_mode = False
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(min(2 ** k + random.random(), 30))
    raise RuntimeError(f"OpenRouter that bai: {last}")

def parse_json_loose(s: str) -> dict:
    """Model hay tra kem ```json fence -> strip roi moi parse."""
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise

# %% Cell 4c — STAGE0: chay tren ca 24 query
PARSED_PATH = f"{WORK}/parsed_queries.json"
ALLOWED_CLS = ", ".join(sorted(OBJ_CLS))

def stage0_one(q: dict) -> dict:
    msg = [{"role": "system", "content": STAGE0_SYSTEM},
           {"role": "user", "content": f"ALLOWED_OBJECT_CLASSES:\n{ALLOWED_CLS}\n\nQUERY:\n{q['q_vi']}"}]
    p = parse_json_loose(or_chat(msg))

    # ---- Post-processing BAT BUOC: dung tin LLM ----
    # 1. type lay tu TEN FILE, khong tu LLM
    p["type"] = q["type"]
    # 2. n_events dem bang regex tren SO DONG
    p["n_events"] = q["n_events"]
    # 3. object_classes loc ve tap hop le (584 class)
    p["object_classes"] = [c for c in (p.get("object_classes") or []) if c in OBJ_CLS]
    # 4. visual_desc_en truncate <= 60 tu (gioi han 77 token cua CLIP)
    p["visual_desc_en"] = " ".join(nfc(p.get("visual_desc_en")).split()[:60])
    # 5. NFC-normalize moi field tieng Viet
    for k in ("q_en", "question_en"):
        p[k] = nfc(p.get(k)) or None
    for k in ("keywords_vi", "keywords_en", "ocr_hints", "named_entities"):
        p[k] = [nfc(x) for x in (p.get(k) or []) if nfc(x)]
    p["query_id"], p["query_file"], p["q_vi"] = q["query_id"], q["query_file"], q["q_vi"]

    # ---- TRAKE: so event PHAI khop so dong ----
    if q["type"] == "trake":
        assert q["n_events"] >= 2, f"{q['query_file']}: khong parse duoc event"
        evs = p.get("events") or []
        if len(evs) != q["n_events"]:
            # goi lai 1 lan voi ep buoc so luong
            force = msg + [{"role": "user", "content":
                            f"You returned {len(evs)} events but the query has EXACTLY "
                            f"{q['n_events']} lines matching 'E<number>:'. The numbers may "
                            f"repeat or skip - ignore them, use LINE ORDER. Return the JSON "
                            f"again with exactly {q['n_events']} events."}]
            try:
                evs = (parse_json_loose(or_chat(force)).get("events") or [])
            except Exception as e:
                print(f"  [WARN] {q['query_id']}: retry loi {e}")
        if len(evs) != q["n_events"]:
            # fallback: tu split theo dong, dich tung dong rieng
            print(f"  [WARN] {q['query_id']}: LLM tra {len(evs)} event, fallback split theo dong")
            lines = event_lines(q["q_vi"])
            evs = []
            for i, ln in enumerate(lines):
                body = EVENT_RE.sub("", ln).strip()
                try:
                    tr = parse_json_loose(or_chat([
                        {"role": "system", "content":
                         'Translate to English. Return {"desc_en": "...", "visual_desc_en": "<=25 words"}'},
                        {"role": "user", "content": body}]))
                except Exception:
                    tr = {}
                evs.append({"idx": i + 1, "desc_vi": nfc(body),
                            "desc_en": nfc(tr.get("desc_en")) or nfc(body),
                            "visual_desc_en": nfc(tr.get("visual_desc_en")) or nfc(body)})
        # THU TU = THU TU DONG. Khong sort, khong dedupe theo con so sau chu E.
        for i, e in enumerate(evs):
            e["idx"] = i + 1
            e["visual_desc_en"] = " ".join(nfc(e.get("visual_desc_en")).split()[:25])
        p["events"] = evs[:q["n_events"]]
        assert len(p["events"]) == q["n_events"]
    else:
        p["events"] = []

    if q["type"] == "qa" and not p.get("question_en"):
        p["question_en"] = p.get("q_en")
    return p

if RELOAD_PARSED and os.path.exists(PARSED_PATH):
    PARSED = json.load(open(PARSED_PATH, encoding="utf-8"))
    print(f"[RELOAD] doc lai {len(PARSED)} query da sua tay tu {PARSED_PATH}")
else:
    PARSED = []
    for i, q in enumerate(QUERIES, 1):
        t0 = time.time()
        PARSED.append(stage0_one(q))
        print(f"  [{i}/{len(QUERIES)}] {q['query_id']:24s} {time.time()-t0:.1f}s")
    json.dump(PARSED, open(PARSED_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

P_BY_ID = {p["query_id"]: p for p in PARSED}

# ---- Validate ngay (fail fast, data contracts §7) ----
for p in PARSED:
    assert p["type"] == p["query_file"].rsplit("-", 1)[-1][:-4]
    assert len(p["visual_desc_en"].split()) <= 60
    assert set(p["object_classes"]) <= set(OBJ_CLS)
    if p["type"] == "trake":
        assert p["n_events"] == len(p["events"]) >= 2, p["query_id"]
    if p["type"] == "qa":
        assert p["question_en"], p["query_id"]
print(f"\nSTAGE0 OK - {len(PARSED)} query validate pass")

# %% [markdown] stage0-review
# ## Cell 5 — `STAGE0_REVIEW` — 👤 ĐIỂM DỪNG CỦA CON NGƯỜI
#
# `ocr_hints` sai/thiếu là **nguyên nhân miss lớn nhất** với news video.
#
# Bắt buộc kiểm mắt: `query-p1-15-qa` phải có `"FANA"` trong `ocr_hints`;
# `query-p1-19-qa` phải có `"Nguyễn Trung Trực"` trong `named_entities`. Không có → sửa prompt.
#
# **Sửa tay:** mở `/kaggle/working/parsed_queries.json`, sửa, rồi đặt `RELOAD_PARSED = True` và chạy lại Cell 4c.

# %% Cell 5 — STAGE0_REVIEW
pd.set_option("display.max_colwidth", 60)
rev = pd.DataFrame([{
    "query_id": p["query_id"], "type": p["type"], "n_ev": p["n_events"],
    "ocr_hints": ", ".join(p["ocr_hints"])[:45],
    "objects": ", ".join(p["object_classes"])[:30],
    "visual_desc_en": p["visual_desc_en"][:60],
} for p in PARSED])
display(rev)

for qid, field, needle in [("query-p1-15-qa", "ocr_hints", "FANA"),
                           ("query-p1-19-qa", "named_entities", "Nguyễn Trung Trực")]:
    p = P_BY_ID.get(qid)
    if p:
        ok = any(needle.lower() in x.lower() for x in p[field])
        print(f"  [{'OK  ' if ok else 'MISS'}] {qid}.{field} chua '{needle}' -> {p[field]}")

# %% [markdown] retrievers
# ## Cell 6 — `RETRIEVERS`: 12 channel
#
# | Channel | Query input | Trả về |
# |:--|:--|:--|
# | `vision` | `clip_encode(visual_desc_en)` | kf-level |
# | `caption` / `asr` / `summary` / `meta` | `embed(q_en)` | kf / video prior |
# | `ocr` | `embed(q_vi)` ← **VI**, vì corpus OCR là VI | kf-level |
# | `bm25_ocr_vi` | `tok_vi(q_vi + ocr_hints × 3)` | kf-level |
# | `bm25_caption_en`, `bm25_asr_en`, `bm25_summary_en` | `tok_en(q_en)` | kf / prior |
# | `bm25_asr_vi`, `bm25_meta` | `tok_vi(q_vi)` | kf / prior |
#
# `ocr_hints` **nhân 3 lần** trong query BM25 để tăng trọng số term chính xác — với news video
# đó thường là tín hiệu mạnh nhất (vd `"FANA"` gần như định vị ngay video).

# %% Cell 6a — search primitives
def faiss_search(channel: str, qvec: np.ndarray, k: int):
    idx = FAISS_IDX[channel]
    k = min(k, idx.ntotal)
    D, I = idx.search(qvec.reshape(1, -1), k)
    return ROWMAP[channel][I[0]], D[0]

def bm25_search(name: str, tokens: list, k: int):
    ids = BM25_IDS[name]
    if not tokens:
        return np.array([], dtype=object), np.array([])
    k = min(k, len(ids))
    res, sc = BM25[name].retrieve([list(tokens)], k=k)   # list-of-list, khong dung np.array
    return ids[res[0]], sc[0]

def unit_to_kf_ranks(unit_ids, base_rank=0):
    """unit kf-level (caption/ocr) -> {kf_id: rank}. Giu rank tot nhat."""
    out = {}
    for r, uid in enumerate(unit_ids):
        i = U_POS.get(uid)
        if i is None:
            continue
        k = U_KFID[i]
        if k is not None and k == k and k in KF2FIDX and k not in out:
            out[k] = base_rank + r
    return out

def spread_time_units_to_kf(unit_ids, pad=2.0):
    """Unit asr/summary phu mot KHOANG THOI GIAN -> moi keyframe trong khoang
    [t_start-pad, t_end+pad] cua video do nhan CUNG rank (khong giam dan)."""
    out = {}
    for r, uid in enumerate(unit_ids):
        i = U_POS.get(uid)
        if i is None:
            continue
        vid = U_VID[i]
        if vid not in KF_BY_VIDEO:
            continue
        ts, te = U_TS[i], U_TE[i]
        kfids, pts, _ = KF_BY_VIDEO[vid]
        if not np.isfinite(ts) or not np.isfinite(te):
            continue
        sel = np.nonzero((pts >= ts - pad) & (pts <= te + pad))[0]
        for j in sel:
            k = kfids[j]
            if k not in out:
                out[k] = r
    return out

def unit_to_video_ranks(unit_ids):
    """summary/meta la video-level -> KHONG sinh candidate, chi boost."""
    out = {}
    for r, uid in enumerate(unit_ids):
        i = U_POS.get(uid)
        if i is None:
            continue
        v = U_VID[i]
        if v not in out:
            out[v] = r
    return out

# %% Cell 6b — retrieve_all: 12 channel cho 1 query
def retrieve_all(p: dict, k=None) -> dict:
    """-> {'kf': {channel: {kf_id: rank}}, 'video': {channel: {video_id: rank}}}"""
    k = k or TOPK_PER_CHANNEL
    q_vi, q_en = p["q_vi"], p.get("q_en") or p["q_vi"]
    vdesc = p.get("visual_desc_en") or q_en
    hints = p.get("ocr_hints") or []
    # ocr_hints nhan 3 lan de tang trong so term chinh xac
    ocr_q = q_vi + " " + " ".join(hints * 3)

    qv_en  = embed_texts([q_en])[0]
    qv_vi  = embed_texts([q_vi])[0]
    qv_vis = clip_encode([vdesc])[0]

    kf_ranks, vid_ranks = {}, {}

    ids, _ = faiss_search("vision", qv_vis, k)
    kf_ranks["vision"] = {kid: r for r, kid in enumerate(ids) if kid in KF2FIDX}

    ids, _ = faiss_search("caption", qv_en, k)
    kf_ranks["caption"] = unit_to_kf_ranks(ids)

    ids, _ = faiss_search("ocr", qv_vi, k)          # corpus OCR la tieng Viet
    kf_ranks["ocr"] = unit_to_kf_ranks(ids)

    ids, _ = faiss_search("asr", qv_en, k)
    kf_ranks["asr"] = spread_time_units_to_kf(ids)

    ids, _ = faiss_search("summary", qv_en, k)
    vid_ranks["summary_prior"] = unit_to_video_ranks(ids)
    ids, _ = faiss_search("meta", qv_en, k)
    vid_ranks["meta_prior"] = unit_to_video_ranks(ids)

    ids, _ = bm25_search("bm25_ocr_vi", tok_vi(ocr_q), k)
    kf_ranks["bm25_ocr_vi"] = unit_to_kf_ranks(ids)
    ids, _ = bm25_search("bm25_caption_en", tok_en(q_en), k)
    kf_ranks["bm25_caption_en"] = unit_to_kf_ranks(ids)
    ids, _ = bm25_search("bm25_asr_vi", tok_vi(q_vi), k)
    kf_ranks["bm25_asr_vi"] = spread_time_units_to_kf(ids)
    ids, _ = bm25_search("bm25_asr_en", tok_en(q_en), k)
    kf_ranks["bm25_asr_en"] = spread_time_units_to_kf(ids)

    ids, _ = bm25_search("bm25_summary_en", tok_en(q_en), k)
    vid_ranks["bm25_summary_en"] = unit_to_video_ranks(ids)
    ids, _ = bm25_search("bm25_meta", tok_vi(q_vi), k)
    vid_ranks["bm25_meta"] = unit_to_video_ranks(ids)

    return {"kf": kf_ranks, "video": vid_ranks}

# %% [markdown] fusion
# ## Cell 7 — `STAGE1_2`: retrieval + RRF fusion
#
# **Vì sao RRF (rank-based) chứ không phải weighted-sum của score thô:** score cosine của các
# channel khác nhau không cùng thang, và **173 video thiếu OCR** (R5) sẽ bị trừng phạt bất công
# nếu cộng score thô (OCR score = 0). RRF chỉ dùng thứ hạng nên channel vắng mặt đơn giản là
# không góp điểm, không kéo xuống.
#
# **Object bonus dùng SOFT, KHÔNG hard filter** — detector 584-class recall hạn chế;
# hard-filter `Dragon` sẽ loại sạch video múa lân nếu detector miss con rồng.

# %% Cell 7a — fusion
VID_PRIOR_CH = ["summary_prior", "meta_prior", "bm25_summary_en", "bm25_meta"]

def compute_video_prior(vid_ranks: dict) -> dict:
    prior = defaultdict(float)
    for ch in VID_PRIOR_CH:
        w = W.get(ch, 0.4)
        for vid, r in vid_ranks.get(ch, {}).items():
            prior[vid] += w / (RRF_K + r + 1)
    if prior:
        vals = np.fromiter(prior.values(), dtype="float64")
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            prior = {v: (s - lo) / (hi - lo) for v, s in prior.items()}   # min-max -> [0,1]
        else:
            prior = {v: 1.0 for v in prior}
    return dict(prior)

def compute_object_bonus(object_classes: list) -> np.ndarray:
    """SOFT bonus, vectorized tren toan bo 177K keyframe."""
    if not object_classes:
        return None
    cols = [OBJ_CLS[c] for c in object_classes]
    hit = np.asarray(OBJ_M[:, cols].sum(axis=1)).ravel().astype("float32")
    return W["object_bonus"] * (hit / len(cols))

KF_CHANNELS = ["vision", "caption", "ocr", "asr",
               "bm25_ocr_vi", "bm25_caption_en", "bm25_asr_vi", "bm25_asr_en"]

def fuse(p: dict, ret: dict) -> pd.DataFrame:
    kf_ranks   = ret["kf"]
    prior      = compute_video_prior(ret["video"])
    obj_bonus  = compute_object_bonus(p["object_classes"])

    score = defaultdict(float)
    for ch in KF_CHANNELS:
        w = W.get(ch, 0.0)
        if w <= 0:
            continue
        for kid, r in kf_ranks.get(ch, {}).items():
            score[kid] += w / (RRF_K + r + 1)

    # video prior boost ca keyframe cua video do (ke ca kf chua co diem kenh nao)
    for kid in list(score):
        score[kid] += prior.get(KF2VID[kid], 0.0)

    if not score:
        return pd.DataFrame(columns=["kf_id", "video_id", "frame_idx", "fused_score"])

    rows = []
    for kid, s in score.items():
        row = KF2ROW[kid]
        ob = float(obj_bonus[row]) if obj_bonus is not None else 0.0
        rows.append({
            "kf_id": kid, "video_id": KF2VID[kid], "frame_idx": KF2FIDX[kid],
            "pts_time": KF2PTS[kid],
            "fused_score": float(s + ob),
            "rank_vision":  kf_ranks.get("vision",  {}).get(kid, -1),
            "rank_caption": kf_ranks.get("caption", {}).get(kid, -1),
            "rank_ocr":     kf_ranks.get("ocr",     {}).get(kid, -1),
            "rank_asr":     kf_ranks.get("asr",     {}).get(kid, -1),
            "video_prior":  float(prior.get(KF2VID[kid], 0.0)),
            "object_bonus": ob,
            "rerank_text_score": np.nan,
            "rerank_vlm_score":  np.nan,
            "vlm_reason": None,
            "vlm_answer": None,
        })
    df = pd.DataFrame(rows).sort_values("fused_score", ascending=False)
    return df.head(TOPK_FUSED).reset_index(drop=True)

# %% Cell 7b — chay Stage 1-2 cho ca 24 query
CAND = {}
t_all = time.time()
for i, p in enumerate(PARSED, 1):
    t0 = time.time()
    ret = retrieve_all(p)
    df  = fuse(p, ret)
    CAND[p["query_id"]] = df
    df.to_parquet(f"{DBG_DIR}/candidates_{p['query_id']}.parquet", index=False)
    print(f"  [{i}/{len(PARSED)}] {p['query_id']:24s} {len(df):5d} cand · "
          f"{df.video_id.nunique():3d} video · {time.time()-t0:.1f}s")
print(f"STAGE1_2 xong · {time.time()-t_all:.0f}s")

# ---- Kiem R5: video thieu OCR van xuat hien duoc trong top-100 ----
no_ocr = set(mf["videos_missing_ocr"])
hit = sum(1 for d in CAND.values() if set(d.head(100).video_id) & no_ocr)
print(f"[R5] {hit}/{len(CAND)} query co video-thieu-OCR trong top-100 "
      f"-> RRF khong trung phat channel vang mat")

# %% [markdown] stage3
# ## Cell 8 — `STAGE3`: Text rerank (1000 → 100)
#
# Evidence card = "document" đưa vào reranker. Kết hợp:
# `final = 0.4 * norm(fused_score) + 0.6 * norm(rerank_text_score)` —
# giữ lại tín hiệu fusion để reranker không "quên" bằng chứng đa kênh.

# %% Cell 8a — evidence card
def asr_around(video_id: str, t: float, pad=6.0) -> str:
    if video_id not in ASR_BY_VIDEO or not np.isfinite(t):
        return ""
    ts, te, en, vi = ASR_BY_VIDEO[video_id]
    sel = np.nonzero((te >= t - pad) & (ts <= t + pad))[0]
    if not len(sel):
        j = int(np.argmin(np.abs((ts + te) / 2 - t)))
        sel = [j]
    txt = " ".join((en[j] or vi[j] or "") for j in sel[:2])
    return txt.strip()[:400]

def evidence_card(kf_id: str) -> str:
    vid = KF2VID[kf_id]
    v = VID_ROW.loc[vid] if vid in VID_ROW.index else None
    pts = KF2PTS[kf_id]
    objs = OBJ_BY_KF.get(kf_id, [])
    objs = list(objs)[:12] if objs is not None else []
    return "\n".join([
        f"[VIDEO] {'' if v is None else v.title}",
        f"[SUMMARY] {'' if v is None else str(v.summary_en)[:300]}",
        f"[FRAME t={pts:.1f}s] {CAP_BY_KF.get(kf_id) or ''}",
        f"[ON-SCREEN TEXT] {OCR_BY_KF.get(kf_id) or ''}",
        f"[SPEECH] {asr_around(vid, pts)}",
        f"[OBJECTS] {', '.join(objs)}",
    ])

def _minmax(a):
    a = np.asarray(a, dtype="float64")
    if len(a) == 0:
        return a
    lo, hi = np.nanmin(a), np.nanmax(a)
    return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)

print(evidence_card(CAND[PARSED[0]["query_id"]].kf_id.iloc[0])[:600])

# %% Cell 8b — reranker: local Qwen3-Reranker | openrouter listwise
_RR = None

def _load_local_reranker():
    global _RR
    if _RR is not None:
        return _RR
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tk = AutoTokenizer.from_pretrained(RERANK_MODEL_LOCAL, padding_side="left")
    md = AutoModelForCausalLM.from_pretrained(
        RERANK_MODEL_LOCAL, torch_dtype=torch.float16,
        device_map="cuda:0").eval()
    yes = tk.convert_tokens_to_ids("yes")
    no  = tk.convert_tokens_to_ids("no")
    _RR = (tk, md, yes, no, torch)
    return _RR

RR_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
             "the Query. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
             "<|im_start|>user\n")
RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def rerank_local(query: str, docs: list, batch=16, max_len=1024) -> np.ndarray:
    """Cross-encoder: P(yes) - P(no) tren token dau tien."""
    tk, md, yes, no, torch = _load_local_reranker()
    out = []
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        prompts = [f"{RR_PREFIX}<Query>: {query}\n<Document>: {d}{RR_SUFFIX}" for d in chunk]
        enc = tk(prompts, return_tensors="pt", padding=True, truncation=True,
                 max_length=max_len).to(md.device)
        with torch.no_grad():
            logits = md(**enc).logits[:, -1, :]
            lp = torch.nn.functional.log_softmax(
                torch.stack([logits[:, no], logits[:, yes]], dim=1), dim=1)
            out.extend(lp[:, 1].exp().float().cpu().numpy().tolist())
    return np.asarray(out, dtype="float32")

def rerank_openrouter(query: str, docs: list, chunk=20) -> np.ndarray:
    """Listwise: 20 candidate/call, model tra [{'id':i,'score':0-10}]."""
    scores = np.zeros(len(docs), dtype="float32")
    for i in range(0, len(docs), chunk):
        part = docs[i:i + chunk]
        listing = "\n\n".join(f"[{j}] {d[:800]}" for j, d in enumerate(part))
        try:
            r = parse_json_loose(or_chat([
                {"role": "system", "content":
                 'Rank documents by relevance to the query. Return JSON: '
                 '{"ranking": [{"id": <int>, "score": <0-10>}, ...]} for ALL documents.'},
                {"role": "user", "content": f"QUERY:\n{query}\n\nDOCUMENTS:\n{listing}"}]))
            for it in r.get("ranking", []):
                j = int(it["id"])
                if 0 <= j < len(part):
                    scores[i + j] = float(it["score"])
        except Exception as e:
            print(f"    [WARN] listwise rerank loi: {e}")
    return scores

def stage3(p: dict, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    q = f"{p.get('q_en') or ''} || {p['q_vi']}"
    docs = [evidence_card(k) for k in df.kf_id]
    sc = rerank_local(q, docs) if RERANK_MODE == "local" else rerank_openrouter(q, docs)
    df = df.copy()
    df["rerank_text_score"] = sc
    df["final_stage3"] = 0.4 * _minmax(df.fused_score) + 0.6 * _minmax(sc)
    return df.sort_values("final_stage3", ascending=False).head(TOPK_TEXT_RERANK).reset_index(drop=True)

# %% Cell 8c — chay Stage 3
t_all = time.time()
for i, p in enumerate(PARSED, 1):
    t0 = time.time()
    CAND[p["query_id"]] = stage3(p, CAND[p["query_id"]])
    print(f"  [{i}/{len(PARSED)}] {p['query_id']:24s} -> {len(CAND[p['query_id']])} · {time.time()-t0:.0f}s")
print(f"STAGE3 xong · {time.time()-t_all:.0f}s")

# %% [markdown] stage4
# ## Cell 9 — `STAGE4`: VLM rerank (100 → 20)
#
# 1 ảnh / call (nhiều ảnh cùng call làm model lẫn frame). 100 kf × 24 query = 2,400 call → **quá đắt**,
# nên chỉ VLM-rerank **top-20** sau Stage 3, và **dedupe kf cùng video cách nhau < 2s**.
#
# **Answer mặc định = verbatim tiếng Việt** (QĐ-5): thể lệ tự mâu thuẫn giữa "ngữ nghĩa" và
# "chuỗi chính xác" → bản verbatim VI đúng dưới **cả hai** cách chấm.

# %% Cell 9a — anh -> base64 (resize long side <= 768, JPEG q85)
def kf_b64(kf_id: str, max_side=None):
    max_side = max_side or VLM_IMAGE_MAX_SIDE
    vid, n = kf_id.split("#")
    if vid not in KF_DIR:
        return None
    try:
        im = Image.open(kf_image_path(vid, int(n))).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

VLM_PROMPT = """You are judging whether ONE video keyframe satisfies a retrieval query.

QUERY (Vietnamese): {q_vi}
QUERY (English):    {q_en}

Return JSON only:
{{"score": 0-10, "reason": "<= 25 words", "answer": "<see below>"}}

score: 10 = this exact frame is the answer; 7-9 = right scene/moment, maybe off by
a second; 4-6 = right video, wrong moment; 1-3 = related topic only; 0 = unrelated.
answer: only for Q&A queries -> the answer to "{question_en}", else null.
        DEFAULT TO VIETNAMESE, VERBATIM. If the answer is text visible in the frame
        (a place name, club name, banner, recipe title, a line of poetry) or spoken by
        someone, copy it EXACTLY as it appears - same wording, same diacritics, no
        translation, no paraphrase, no added words. Only answer in English when the
        question is purely descriptive (a count, a colour) AND no Vietnamese string on
        screen expresses it. Max 100 characters."""

def vlm_judge(p: dict, kf_id: str) -> dict:
    b64 = kf_b64(kf_id)
    if b64 is None:
        return {"score": 0.0, "reason": "no image", "answer": None}
    txt = VLM_PROMPT.format(q_vi=p["q_vi"], q_en=p.get("q_en") or "",
                            question_en=p.get("question_en") or "")
    try:
        raw = or_chat([{"role": "user", "content": [
            {"type": "text", "text": txt + "\n\nEVIDENCE:\n" + evidence_card(kf_id)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}], model=VLM_MODEL, max_tokens=300)
        d = parse_json_loose(raw)
        ans = d.get("answer")
        return {"score": float(d.get("score", 0)),
                "reason": nfc(d.get("reason"))[:150],
                "answer": nfc(ans) if ans else None}
    except Exception as e:
        return {"score": 0.0, "reason": f"error: {e}"[:150], "answer": None}

def dedupe_near(df: pd.DataFrame, gap=2.0, limit=None):
    """Bo kf trung: 2 kf cung video cach nhau < gap giay -> giu 1 dai dien (diem cao hon)."""
    keep, chosen = [], defaultdict(list)
    for r in df.itertuples(index=False):
        if any(abs(r.pts_time - t) < gap for t in chosen[r.video_id]):
            continue
        chosen[r.video_id].append(r.pts_time)
        keep.append(r.kf_id)
        if limit and len(keep) >= limit:
            break
    return keep

# %% Cell 9b — chay Stage 4
from concurrent.futures import ThreadPoolExecutor

if SKIP_VLM:
    print("[SKIP_VLM] bo qua Stage 4. QA gan nhu chac chan sai - BAO RO khi nop.")
else:
    t_all = time.time()
    for i, p in enumerate(PARSED, 1):
        df = CAND[p["query_id"]]
        if df.empty:
            continue
        targets = dedupe_near(df, gap=2.0, limit=TOPK_VLM_RERANK)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(lambda k: (k, vlm_judge(p, k)), targets))
        m = {k: v for k, v in res}
        df["rerank_vlm_score"] = df.kf_id.map(lambda k: m[k]["score"] if k in m else np.nan)
        df["vlm_reason"]       = df.kf_id.map(lambda k: m[k]["reason"] if k in m else None)
        df["vlm_answer"]       = df.kf_id.map(lambda k: m[k]["answer"] if k in m else None)
        CAND[p["query_id"]] = df
        df.to_parquet(f"{DBG_DIR}/candidates_{p['query_id']}.parquet", index=False)
        print(f"  [{i}/{len(PARSED)}] {p['query_id']:24s} {len(targets)} call · {time.time()-t0:.0f}s")
    print(f"STAGE4 xong · {time.time()-t_all:.0f}s")

# %% Cell 9c — thu tu xep hang cuoi cung
def final_order(df: pd.DataFrame) -> pd.DataFrame:
    """sort theo (vlm, text, fused) giam dan. NaN vlm -> xuong duoi nhung van giu."""
    d = df.copy()
    d["_v"] = d.rerank_vlm_score.fillna(-1.0)
    d["_t"] = d.rerank_text_score.fillna(-1.0)
    return d.sort_values(["_v", "_t", "fused_score"], ascending=False).reset_index(drop=True)

# %% [markdown] kis
# ## Cell 10 — KIS rows
#
# Được 100 dòng thì **dùng hết 100**. Không có penalty cho dòng sai, chỉ có phần thưởng cho dòng đúng.
#
# - **Diversity**: mỗi video tối đa 5 frame trong 30 dòng đầu (tránh dồn hết vào 1 video sai), 30 dòng sau nới lên 10.
# - **Temporal padding**: với mỗi kf top-10, thêm kf lân cận (`n-1`, `n+1`) — "khoảnh khắc đầu tiên" rất dễ lệch 1 keyframe.

# %% Cell 10 — KIS
def neighbour_kfs(kf_id: str, offs=(-1, 1)):
    vid, n = kf_id.split("#")
    out = []
    for o in offs:
        k = f"{vid}#{int(n) + o:03d}"
        if k in KF2FIDX:
            out.append(k)
    return out

def build_kis_rows(p: dict) -> list:
    """Duyet theo TIER voi cap tang dan.

    Cap PHAI theo tier, khong duoc viet 'cap = 5 if len(rows) < 30 else 10': khi moi
    video da cham 5 ma rows moi co 5*n_video < 30 thi khong dong nao them duoc nua
    -> ket deadlock, roi vong bu-quota (no cap) lai chen dong vao trong 30 dong dau
    va pha vo chinh rang buoc diversity.
    """
    d = final_order(CAND[p["query_id"]])
    order = list(d.kf_id)
    rows, seen, per_video = [], set(), Counter()

    def push(kid, limit, cap):
        if len(rows) >= limit:
            return False
        key = (KF2VID[kid], KF2FIDX[kid])
        if key in seen:
            return True
        if cap is not None and per_video[KF2VID[kid]] >= cap:
            return True
        seen.add(key)
        per_video[KF2VID[kid]] += 1
        rows.append([KF2VID[kid], int(KF2FIDX[kid])])
        return True

    # (so dong toi da cua tier, so frame toi da / video); None = khong gioi han
    for limit, cap in [(30, 5), (60, 10), (MAX_ROWS_PER_CSV, None)]:
        for i, kid in enumerate(order):
            if not push(kid, limit, cap):
                break
            if i < 10:                              # temporal padding cho top-10
                for nb in neighbour_kfs(kid):
                    if not push(nb, limit, cap):
                        break
        if len(rows) >= MAX_ROWS_PER_CSV:
            break
    return rows[:MAX_ROWS_PER_CSV]

# %% [markdown] qa
# ## Cell 11 — QA rows (QĐ-5, R6)
#
# 👤 **Review tay bắt buộc.** `query-p1-19-qa` (2 câu thơ) phải là **tiếng Việt nguyên văn**, không dịch.
# `query-p1-15-qa` (tên xã) phải là tên xã tiếng Việt. Đây là R6 — rủi ro cao nhất của dạng QA.
#
# **Hedge bằng số dòng:** dòng đầu = answer VI verbatim ở nhiều `frame_idx`; dòng sau = cùng frame,
# answer dịch EN. Dưới cách chấm ngữ nghĩa cả hai đều trúng; dưới exact-string thì bản VI còn cơ hội.

# %% Cell 11a — gom answer theo cum ngu nghia
def norm_ans(s: str) -> str:
    """lowercase + bo dau + bo ky tu khong phai chu/so -> khoa gom cum."""
    s = unicodedata.normalize("NFD", nfc(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", s).strip()

def truncate_100(s: str) -> str:
    """Hard truncate <= 100 ky tu, cat o ranh gioi tu."""
    s = nfc(s)
    if len(s) <= 100:
        return s
    cut = s[:100]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 40 else cut).strip()

def translate_en(s: str) -> str:
    """Dong hedge phu - chi de tang co hoi duoi cach cham ngu nghia."""
    try:
        return truncate_100(nfc(parse_json_loose(or_chat([
            {"role": "system", "content":
             'Translate Vietnamese to English. Keep proper nouns UNCHANGED. '
             'Return {"en": "..."} only, max 100 characters.'},
            {"role": "user", "content": s}])).get("en", "")))
    except Exception:
        return ""

def build_qa_rows(p: dict) -> list:
    d = final_order(CAND[p["query_id"]])
    have = d[d.vlm_answer.notna() & (d.vlm_answer.astype(str).str.strip() != "")]

    # ---- vote clustering ----
    clusters = defaultdict(lambda: {"votes": 0, "best": -1e9, "surface": Counter(), "kfs": []})
    for r in have.itertuples(index=False):
        k = norm_ans(r.vlm_answer)
        if not k:
            continue
        c = clusters[k]
        c["votes"] += 1
        sc = float(r.rerank_vlm_score if np.isfinite(r.rerank_vlm_score) else 0)
        c["best"] = max(c["best"], sc)
        c["surface"][nfc(r.vlm_answer)] += 1
        c["kfs"].append(r.kf_id)

    ranked = sorted(clusters.values(), key=lambda c: (-c["votes"], -c["best"]))
    if not ranked:                       # khong co answer nao -> van phai nop du dong
        print(f"  [WARN] {p['query_id']}: VLM khong tra answer nao")
        fallback = truncate_100(OCR_BY_KF.get(d.kf_id.iloc[0], "") or p.get("q_en") or "N/A")
        ranked = [{"votes": 0, "best": 0, "surface": Counter({fallback: 1}),
                   "kfs": list(d.kf_id[:20])}]

    rows, seen = [], set()
    quota = [60, 20, 10]                 # dong danh cho answer hang 1 / 2 / 3
    for ci, c in enumerate(ranked[:3]):
        ans = truncate_100(c["surface"].most_common(1)[0][0])
        if len(nfc(c["surface"].most_common(1)[0][0])) > 100:
            print(f"  [WARN] {p['query_id']}: answer bi cat tu "
                  f"{len(nfc(c['surface'].most_common(1)[0][0]))} -> 100 ky tu, REVIEW TAY")
        # cac frame diem cao nhat cua cum nay, roi bu them tu top chung
        kfs = list(dict.fromkeys(c["kfs"] + list(d.kf_id)))
        for kid in kfs:
            if len(rows) >= MAX_ROWS_PER_CSV:
                break
            key = (KF2VID[kid], KF2FIDX[kid], ans)
            if key in seen:
                continue
            seen.add(key)
            rows.append([KF2VID[kid], int(KF2FIDX[kid]), ans])
            if sum(1 for r in rows if r[2] == ans) >= quota[ci]:
                break

    # ---- hedge tieng Anh o CUOI file (cung frame, answer dich) ----
    if rows and len(rows) < MAX_ROWS_PER_CSV:
        main = rows[0][2]
        en = translate_en(main)
        if en and norm_ans(en) != norm_ans(main):
            for r in rows[:MAX_ROWS_PER_CSV - len(rows)]:
                key = (r[0], r[1], en)
                if key in seen:
                    continue
                seen.add(key)
                rows.append([r[0], r[1], en])
                if len(rows) >= MAX_ROWS_PER_CSV:
                    break
    return rows[:MAX_ROWS_PER_CSV]

# %% [markdown] trake
# ## Cell 12 — TRAKE: DP monotonic alignment (QĐ-6)
#
# **DP là bắt buộc, không phải tối ưu hóa.** Lấy `argmax` từng event độc lập sẽ cho frame
# **không** theo thứ tự thời gian (event 3 trước event 1) → sai format → 0 điểm.
#
# ```
# dp[i][k] = S[i,k] + max(dp[j][k-1] for j < i)     -> tiền tố max chạy dần, O(N*K)
# ```

# %% Cell 12a — DP monotonic alignment + beam
def dp_align(S: np.ndarray, banned_first=frozenset()):
    """S: [N, K]. Tim i_1 < i_2 < ... < i_K maximize sum_k S[i_k, k].
    Tra ve (path, score) hoac (None, -inf) neu N < K."""
    N, K = S.shape
    if N < K:
        return None, -np.inf
    NEG = -1e18
    dp   = np.full((N, K), NEG, dtype="float64")
    back = np.full((N, K), -1, dtype="int64")
    for i in range(N):
        if i not in banned_first:
            dp[i, 0] = S[i, 0]
    for k in range(1, K):
        best_val, best_j = NEG, -1
        for i in range(N):
            if i > 0 and dp[i - 1, k - 1] > best_val:      # tien to max cua cot k-1
                best_val, best_j = dp[i - 1, k - 1], i - 1
            if best_j >= 0:
                dp[i, k]   = S[i, k] + best_val
                back[i, k] = best_j
    end = int(np.argmax(dp[:, K - 1]))
    if dp[end, K - 1] <= NEG / 2:
        return None, -np.inf
    path, i = [end], end
    for k in range(K - 1, 0, -1):
        i = int(back[i, k])
        path.append(i)
    path.reverse()
    assert all(path[a] < path[a + 1] for a in range(len(path) - 1)), path
    return path, float(dp[end, K - 1])

# ---- unit test tren ma tran gia ----
_S = np.array([[9., 0., 0.], [0., 9., 0.], [0., 0., 9.], [8., 8., 8.]])
_p, _sc = dp_align(_S)
assert _p == [0, 1, 2] and abs(_sc - 27) < 1e-6, (_p, _sc)
_S2 = np.array([[0., 5.], [5., 0.]])     # argmax doc lap se ra [1,0] (SAI thu tu)
_p2, _ = dp_align(_S2)
assert _p2 == [0, 1], _p2
print("dp_align unit test OK — luon tra ve chi so TANG DAN NGHIEM NGAT")

# %% Cell 12b — score matrix + TRAKE rows
CAP_EMB_ROW = dict(zip(units.loc[units.channel == "caption", "kf_id"],
                       units.loc[units.channel == "caption", "emb_row"]))

def reconstruct(channel: str, rows: np.ndarray) -> np.ndarray:
    """Lay lai vector tu IndexFlat. reconstruct_batch khong co o faiss cu -> fallback."""
    idx = FAISS_IDX[channel]
    rows = np.asarray(rows, dtype="int64")
    try:
        return np.asarray(idx.reconstruct_batch(rows), dtype="float32")
    except AttributeError:
        return np.stack([idx.reconstruct(int(r)) for r in rows]).astype("float32")

def _cap_rows_for_video(vid):
    """emb_row caption cua tung keyframe trong video (-1 = khong co)."""
    kfids, _, _ = KF_BY_VIDEO[vid]
    return np.asarray([CAP_EMB_ROW.get(k, -1) for k in kfids], dtype="int64")

def trake_score_matrix(p: dict, vid: str) -> np.ndarray:
    """S[i, k] = a*cos(clip(e_k.visual), vision_i) + b*cos(embed(e_k.desc_en), caption_i)
                 + c*bm25(tok_vi(e_k.desc_vi), ocr_i);  min-max normalize tung cot k."""
    a, b, c = TRAKE_ABC
    kfids, pts, vis_rows = KF_BY_VIDEO[vid]
    N, evs = len(kfids), p["events"]
    K = len(evs)
    S = np.zeros((N, K), dtype="float32")

    Vvis = reconstruct("vision", vis_rows)
    Qvis = clip_encode([e["visual_desc_en"] or e["desc_en"] for e in evs])
    S += a * (Vvis @ Qvis.T)

    cap_rows = _cap_rows_for_video(vid)
    ok = cap_rows >= 0
    if ok.any():
        Vcap = reconstruct("caption", cap_rows[ok])
        Qcap = embed_texts([e["desc_en"] or e["desc_vi"] for e in evs])
        S[ok] += b * (Vcap @ Qcap.T)

    # OCR: BM25 tren pham vi video nay
    ocr_txt = [OCR_BY_KF.get(k) or "" for k in kfids]
    if any(ocr_txt):
        doc_tok = [set(tok_vi(t)) if t else set() for t in ocr_txt]   # tokenize 1 lan
        for k, e in enumerate(evs):
            qtok = set(tok_vi(e["desc_vi"] or e["desc_en"]))
            if not qtok:
                continue
            for i, dtok in enumerate(doc_tok):
                if dtok:
                    S[i, k] += c * len(qtok & dtok) / len(qtok)

    for k in range(K):                                     # min-max tung cot
        col = S[:, k]
        lo, hi = float(col.min()), float(col.max())
        S[:, k] = (col - lo) / (hi - lo) if hi > lo else 0.0
    return S

def build_trake_rows(p: dict) -> list:
    d = final_order(CAND[p["query_id"]])
    K = p["n_events"]
    # Buoc 1: video-level retrieval - tong diem top-10 kf cua moi video
    vscore = (d.assign(_s=d.fused_score)
                .groupby("video_id")["_s"]
                .apply(lambda s: float(s.nlargest(10).sum()))
                .sort_values(ascending=False))
    top_videos = list(vscore.index[:TRAKE_TOP_VIDEOS])

    cands = []
    for vid in top_videos:
        if vid not in KF_BY_VIDEO or len(KF_BY_VIDEO[vid][0]) < K:
            continue
        S = trake_score_matrix(p, vid)
        kfids = KF_BY_VIDEO[vid][0]
        banned = set()
        for _ in range(TRAKE_BEAM):                        # beam: cam i_1 da chon
            path, sc = dp_align(S, banned_first=frozenset(banned))
            if path is None:
                break
            banned.add(path[0])
            frames = [int(KF2FIDX[kfids[i]]) for i in path]
            if len(set(frames)) != K or any(frames[j] >= frames[j + 1] for j in range(K - 1)):
                continue                                   # frame_idx phai TANG DAN NGHIEM NGAT
            cands.append((sc + float(vscore[vid]) * 0.01, [vid] + frames))

    cands.sort(key=lambda x: -x[0])
    rows, seen = [], set()
    for _, row in cands:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= MAX_ROWS_PER_CSV:
            break

    if not rows:      # fallback: khong duoc de trong (0 diem chac chan)
        vid = top_videos[0] if top_videos else str(d.video_id.iloc[0])
        kfids = KF_BY_VIDEO[vid][0]
        step = max(1, len(kfids) // (K + 1))
        idxs = [min(len(kfids) - 1, (j + 1) * step) for j in range(K)]
        rows = [[vid] + [int(KF2FIDX[kfids[i]]) for i in idxs]]
        print(f"  [WARN] {p['query_id']}: khong sinh duoc path, fallback chia deu theo thoi gian")
    return rows

# %% Cell 12c — sinh rows cho ca 24 query
ROWS = {}
for i, p in enumerate(PARSED, 1):
    t0 = time.time()
    if p["type"] == "kis":
        ROWS[p["query_id"]] = build_kis_rows(p)
    elif p["type"] == "qa":
        ROWS[p["query_id"]] = build_qa_rows(p)
    else:
        ROWS[p["query_id"]] = build_trake_rows(p)
    print(f"  [{i}/{len(PARSED)}] {p['query_id']:24s} {p['type']:5s} "
          f"{len(ROWS[p['query_id']]):3d} dong · {time.time()-t0:.0f}s")

# %% Cell 12d — 👤 REVIEW TAY: answer QA (R6 - rui ro cao nhat)
for p in PARSED:
    if p["type"] != "qa":
        continue
    ans = Counter(r[2] for r in ROWS[p["query_id"]])
    print(f"\n{p['query_id']}  Q: {p.get('question_en')}")
    for a, c in ans.most_common(4):
        print(f"    {c:3d} dong · {len(a):3d} ky tu · {a!r}")
print("\n>>> KIEM MAT: cau tho / ten xa / tieu de mon an PHAI la tieng Viet NGUYEN VAN, khong dich.")

# %% [markdown] writer
# ## Cell 13 — `WRITER`
#
# Dùng module `csv`, **đừng** tự nối chuỗi bằng `",".join()`: answer tiếng Việt có thể chứa dấu phẩy
# và module `csv` xử lý escape đúng chuẩn còn nối tay thì không.
#
# Không header · UTF-8 không BOM · `\n` · `QUOTE_MINIMAL` · ≤ 100 dòng.

# %% Cell 13 — WRITER
shutil.rmtree(SUB_DIR, ignore_errors=True)
os.makedirs(SUB_DIR, exist_ok=True)

for p in PARSED:
    rows = ROWS[p["query_id"]][:MAX_ROWS_PER_CSV]
    out = f"{SUB_DIR}/{p['query_file'][:-4]}.csv"     # chi doi .txt -> .csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for r in rows:
            w.writerow(r)
print(f"da ghi {len(os.listdir(SUB_DIR))} file vao {SUB_DIR}")
print(open(f"{SUB_DIR}/{PARSED[0]['query_file'][:-4]}.csv", encoding="utf-8").read()[:200])

# %% [markdown] validator
# ## Cell 14 — `VALIDATOR` (Definition of Done)
#
# Chạy trước **mỗi** lần nộp. Fail bất kỳ check nào thì **không nộp**.
#
# Check #7 (frame_idx tồn tại thật) đắt nhất nhưng quan trọng nhất: nó bắt được lỗi nộp `n`
# hoặc `pts_time` thay vì `frame_idx`.

# %% Cell 14 — VALIDATOR: 12 check
VID_RE = re.compile(r"^L\d{2}_V\d{3}$")
problems = []

def bad(qid, n, msg):
    problems.append(f"[check {n:2d}] {qid}: {msg}")

files = sorted(glob.glob(f"{SUB_DIR}/*"))
expected = {p["query_file"][:-4] + ".csv" for p in PARSED}

# 1 + 2
for f in files:
    if not f.endswith(".csv"):
        bad(os.path.basename(f), 1, "khong phai .csv")
got = {os.path.basename(f) for f in files}
if got != expected:
    bad("-", 2, f"thieu {sorted(expected - got)} · thua {sorted(got - expected)}")

raw_all = {}
for p in PARSED:
    qid, path = p["query_id"], f"{SUB_DIR}/{p['query_file'][:-4]}.csv"
    if not os.path.exists(path):
        continue
    raw = open(path, "rb").read().decode("utf-8")           # 3: UTF-8 hop le
    raw_all[qid] = raw
    if raw.startswith("﻿"):
        bad(qid, 3, "co BOM")
    if "\r" in raw:                                          # 12
        bad(qid, 12, "co ky tu \\r lac")
    all_rows = list(csv.reader(io.StringIO(raw)))
    rows = [r for r in all_rows if r and any(c.strip() for c in r)]
    if len(rows) != len(all_rows):                           # 12
        bad(qid, 12, f"co {len(all_rows)-len(rows)} dong trang parse thanh row rong")

    if not rows:
        bad(qid, 5, "0 dong")
        continue
    if not (1 <= len(rows) <= 100):                          # 5
        bad(qid, 5, f"{len(rows)} dong")
    try:
        int(rows[0][1])                                       # 4: dong 1 phai la du lieu
    except (ValueError, IndexError):
        bad(qid, 4, f"dong 1 co ve la header: {rows[0]}")

    ncol_expect = {"kis": 2, "qa": 3}.get(p["type"], 1 + p["n_events"])
    for li, r in enumerate(rows, 1):
        if len(r) != ncol_expect:                             # 8 / 9 / 10
            bad(qid, 8 if p["type"] == "kis" else (9 if p["type"] == "qa" else 10),
                f"dong {li}: {len(r)} cot, ky vong {ncol_expect}")
            continue
        vid = r[0]
        if not VID_RE.match(vid):                             # 6
            bad(qid, 6, f"dong {li}: video_id '{vid}' sai format")
            continue
        try:
            fidxs = [int(x) for x in r[1:1 + (ncol_expect - (2 if p["type"] == "qa" else 1))]]
        except ValueError:
            bad(qid, 7, f"dong {li}: frame_idx khong phai int")
            continue
        for fi in fidxs:                                      # 7: TON TAI THAT
            if fi < 0 or fi not in VID_FIDX_SET.get(vid, ()):
                bad(qid, 7, f"dong {li}: frame_idx {fi} KHONG ton tai trong {vid}")
        if p["type"] == "qa":
            a = r[2]
            if not a.strip():
                bad(qid, 9, f"dong {li}: answer rong")
            if len(a) > 100:
                bad(qid, 9, f"dong {li}: answer {len(a)} ky tu > 100")
        if p["type"] == "trake":
            if any(fidxs[j] >= fidxs[j + 1] for j in range(len(fidxs) - 1)):
                bad(qid, 10, f"dong {li}: frame_idx KHONG tang dan nghiem ngat: {fidxs}")
    if len(set(map(tuple, rows))) != len(rows):               # 11
        bad(qid, 11, "co dong trung lap hoan toan")

print("=" * 70)
if problems:
    for x in problems[:80]:
        print("  " + x)
    print(f"\n{len(problems)} VAN DE -> KHONG NOP. Sua roi chay lai.")
else:
    print("  12/12 CHECK PASS - san sang dong goi")
print("=" * 70)
assert not problems, f"{len(problems)} check FAIL"

# %% [markdown] zip
# ## Cell 15 — `ZIP`
#
# Lỗi phổ biến nhất theo thể lệ: nén trực tiếp các file CSV thay vì nén **thư mục** `submission`.
# `base_dir="submission"` là chỗ quyết định điều đó.

# %% Cell 15 — ZIP
name = f"{TEAM_NAME}_round{ROUND}"
zip_path = f"{WORK}/{name}.zip"
if os.path.exists(zip_path):
    os.remove(zip_path)
shutil.make_archive(f"{WORK}/{name}", "zip", root_dir=WORK, base_dir="submission")

names = zipfile.ZipFile(zip_path).namelist()
assert all(n.startswith("submission/") for n in names), "THIEU thu muc submission/"
print(f"{zip_path}  ({os.path.getsize(zip_path)/1e3:.0f} KB · {len(names)} entry)")
print(*names[:10], sep="\n")
if TEAM_NAME == "team_XXX":
    print("\n[WARN] TEAM_NAME van la 'team_XXX' - DOI THANH TEN DOI THAT truoc khi nop")

# %% [markdown] submit-strategy
# ## Chiến lược 3 lần nộp
#
# | Lần | Cấu hình | Mục tiêu |
# |:-:|:--|:--|
# | 1 | Pipeline đầy đủ, weight mặc định, đã pass 12 check | **Baseline an toàn.** Có điểm trên bảng, biết format đúng |
# | 2 | Sau khi xem Public LB + review tay `parsed_queries.json` (sửa `ocr_hints`), tune weight | Cải tiến có định hướng |
# | 3 | Bản tốt nhất, review tay answer QA + verify TRAKE ordering | **Lần cuối được tính điểm** |
#
# > Public LB chỉ tính **50%** đáp án; Private tính 100%. Nếu lần 2 chỉ hơn lần 1 một chút thì
# > đó có thể là nhiễu, không phải cải tiến thật — **đừng overfit Public LB**.
