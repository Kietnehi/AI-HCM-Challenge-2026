#!/usr/bin/env python3
"""Chay THAT cac cell Phase A cua NB01 tren Feature_Dataset LOCAL.

Muc dich: bat loi runtime truoc khi ton 1 session Kaggle 4h.
Khac voi test_logic.py (kiem ham roi le), file nay exec DUNG source cua notebook.

Chay:  python tests/smoke_nb01.py            # Phase A1-A3 full + A4/E tren subset
       python tests/smoke_nb01.py --full     # them Phase A4 (objects, ~177K file, cham)

Bo qua: Cell 1 (pip), Cell 8 (BM25 - can bm25s), Cell 9-13 (can API key).
"""
import os, re, sys, time, glob, json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "src" / "nb01_index_builder.py"
ROOT = HERE.parents[1]          # tests/ -> Code-ThuNghiem-AIC/ -> ai-challenge-2026/
FEAT = ROOT / "Feature_Dataset"
FULL = "--full" in sys.argv

if not FEAT.is_dir():
    sys.exit(f"Khong thay {FEAT} - bo qua smoke test")


def load_cells(path):
    """-> [(label, source)] theo dung thu tu, chi lay code cell."""
    out, label, buf = [], None, []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r"^# %%(?:\s+\[(markdown)\])?\s*(.*)$", line)
        if m:
            if label is not None:
                out.append((label, "\n".join(buf)))
            label = None if m.group(1) else (m.group(2) or "?")
            buf = []
            if m.group(1):
                label = None
        elif label is not None:
            buf.append(line)
    if label is not None:
        out.append((label, "\n".join(buf)))
    return out


CELLS = dict(load_cells(SRC))
ORDER = [l for l, _ in load_cells(SRC)]
print(f"Doc {len(CELLS)} code cell tu {SRC.name}\n")

NS = {"__name__": "__smoke__"}

def run(label_prefix, patch=None):
    label = next(l for l in ORDER if l.startswith(label_prefix))
    src = CELLS[label]
    if patch:
        src = patch(src)
    t0 = time.time()
    print(f">>> {label}")
    exec(compile(src, f"<{label}>", "exec"), NS)
    print(f"    ({time.time()-t0:.1f}s)\n")


# ---- Cell 1b: CONFIG ----
run("Cell 1b")

# ---- Cell 1c: PATHS. Thay resolve() bang thu muc LOCAL, WORK -> thu muc tam ----
WORKDIR = HERE / "_smoke_out"
def patch_paths(src):
    src = src.replace('FEAT_ROOT = resolve("kitnehi1211", "feature-aic-2026", "Feature_Dataset")',
                      f'FEAT_ROOT = r"{FEAT}"')
    return src
NS["WORK"] = str(WORKDIR)
run("Cell 1c", patch_paths)
NS["WORK"] = str(WORKDIR)   # cell 1c khong ghi de WORK, nhung chac chan lai

# ---- Cell 2: PATH_VERIFY (chay nguyen ven - du lieu local la ban day du) ----
run("Cell 2")

# ---- Cell 3: keyframes.parquet (full 873 video, co assert invariant vs .npy) ----
run("Cell 3")
kf = NS["kf"]
print(f"    keyframes: {len(kf):,} dong · vd {kf.kf_id.iloc[0]} "
      f"frame_idx={kf.frame_idx.iloc[0]} kf_path={kf.kf_path.iloc[0]}\n")

# ---- Cell 4: videos.parquet ----
run("Cell 4")
videos = NS["videos"]
print(f"    videos: {len(videos)} · has_ocr={int(videos.has_ocr.sum())} "
      f"has_summary={int(videos.has_summary.sum())} "
      f"has_transcript={int(videos.has_transcript.sum())}\n")

# ---- Cell 5a-5f: text_units.parquet ----
for c in ("Cell 5a", "Cell 5b", "Cell 5c", "Cell 5d", "Cell 5e", "Cell 5f"):
    run(c)
units = NS["units"]

# ---- Phase A4 / E tren SUBSET (full qua cham tren may local) ----
import numpy as np
import pandas as pd
import scipy.sparse as sp

if FULL:
    run("Cell 6")
    run("Cell 7")
else:
    print(">>> Cell 6/7 chay tren SUBSET (dung --full de chay het)")
    t0 = time.time()
    sub_vids = sorted(set(kf.video_id.astype(str)))[:3]
    sub = kf[kf.video_id.astype(str).isin(sub_vids)].reset_index(drop=True)

    # --- logic Cell 6 (objects) tren subset ---
    OBJ_CLS_IDX = NS["OBJ_CLS_IDX"]
    _read_obj = None
    exec(CELLS[next(l for l in ORDER if l.startswith("Cell 6"))].split("_tasks =")[0],
         {**NS, "kf": sub}, (_g := {}))
    _read_obj = _g["_read_obj"]
    rows_i, cols_j, n_cls = [], [], 0
    for i, (vid, n) in enumerate(zip(sub.video_id.astype(str), sub.n)):
        _, cls, scs = _read_obj((i, vid, int(n)))
        n_cls += len(cls)
        for c in cls:
            rows_i.append(i); cols_j.append(OBJ_CLS_IDX[c])
    M = sp.csr_matrix((np.ones(len(rows_i), dtype=bool), (rows_i, cols_j)),
                      shape=(len(sub), 584), dtype=bool)
    assert M.shape == (len(sub), 584)
    print(f"    objects subset: {len(sub)} kf · {n_cls} (kf,class) qua nguong "
          f"{NS['OBJECT_SCORE_THRESHOLD']} · nnz={M.nnz} · {time.time()-t0:.1f}s")

    # --- logic Cell 7 (vision faiss) tren subset ---
    import faiss
    t0 = time.time()
    X = np.vstack([np.load(f"{NS['P_CLIP']}/{v}.npy").astype("float32") for v in sub_vids])
    assert X.shape == (len(sub), NS["VISUAL_DIM"]), (X.shape, len(sub))
    faiss.normalize_L2(X)
    idx = faiss.IndexFlatIP(NS["VISUAL_DIM"]); idx.add(X)
    assert idx.ntotal == len(sub)
    D, I = idx.search(X[:1], 3)
    assert abs(D[0][0] - 1.0) < 1e-4 and I[0][0] == 0, (D[0], I[0])
    print(f"    vision subset: ntotal={idx.ntotal} · self-search top1 cos={D[0][0]:.4f} "
          f"(=1.0 -> normalize dung) · {time.time()-t0:.1f}s\n")

# ---- Kiem tra tong hop ----
print("=" * 66)
ok = True
def c(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))

c("keyframes = 177,321 dong", len(kf) == 177_321, f"{len(kf):,}")
c("kf_id unique", kf.kf_id.is_unique)
c("frame_idx dtype int32", str(kf.frame_idx.dtype) == "int32", str(kf.frame_idx.dtype))
c("invariant kf vs .npy pass cho ca 873 video", NS["KF_MISMATCH"] == [], str(NS["KF_MISMATCH"]))
c("videos = 873, khong co '_failed'",
  len(videos) == 873 and "_failed" not in set(videos.video_id))
c("has_ocr = 700", int(videos.has_ocr.sum()) == 700, str(int(videos.has_ocr.sum())))
c("has_summary = 865", int(videos.has_summary.sum()) == 865, str(int(videos.has_summary.sum())))
c("thieu summary = L26_V072..079", NS["VIDEOS_MISSING_SUMMARY"] ==
  [f"L26_V{i:03d}" for i in range(72, 80)], str(NS["VIDEOS_MISSING_SUMMARY"]))
c("unit_id unique", units.unit_id.is_unique)
c("ocr units = 128,664 (khong loc mat gi dang ke)",
  abs(int((units.channel == "ocr").sum()) - 128_664) <= 2000,
  f"{int((units.channel=='ocr').sum()):,}")
c("meta units = 873", int((units.channel == "meta").sum()) == 873)
c("ASR frame_idx khong null",
  units.loc[units.channel == "asr", "frame_idx"].notna().all())
c("ASR da dedupe (khong 2 unit cung video_id+text_embed)",
  not units[units.channel == "asr"].duplicated(subset=["video_id", "text_embed"]).any())
c("caption dup co emb_row == -2",
  int((units.emb_row == -2).sum()) > 0, f"{int((units.emb_row==-2).sum()):,} dong dup")
c("dong dup deu co dup_of tro toi kf_id hop le",
  units.loc[units.emb_row == -2, "dup_of"].isin(set(kf.kf_id)).all())
c("summary units co unit video-level 'sum:<vid>' khong kem '#'",
  int(units.unit_id.str.match(r"^sum:[^#]+$").sum()) == 865,
  str(int(units.unit_id.str.match(r'^sum:[^#]+$').sum())))
c("tong text_units ~319K (+/-25%)", 0.75 * 319_000 <= len(units) <= 1.25 * 319_000,
  f"{len(units):,}")

print("\n  value_counts theo channel:")
for k, v in units.channel.value_counts().items():
    print(f"    {k:8s} {v:>8,}")

print("=" * 66)
print("SMOKE NB01:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
