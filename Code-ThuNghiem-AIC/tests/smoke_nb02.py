#!/usr/bin/env python3
"""Chay THAT cac cell sinh row + WRITER + VALIDATOR cua NB02 tren index TONG HOP.

Vi sao can: day la doan ma mot bug = 0 diem cau do (sai so cot, frame_idx khong ton
tai, TRAKE khong tang dan, answer > 100 ky tu). Khong the doi den luc co index that
tren Kaggle moi phat hien.

Index tong hop: 12 video x 40 keyframe, FAISS/embedding duoc stub bang vector ngau
nhien co seed. Query lay tu bo de THAT (../THUNGHIEM-bo-de-thi) neu co.

Chay:  python tests/smoke_nb02.py
"""
import os, re, sys, io, csv, json, glob, time, shutil, zipfile, random, unicodedata
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import faiss

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "src" / "nb02_pipeline_submit.py"
QDIR = HERE.parents[1] / "THUNGHIEM-bo-de-thi"
WORKDIR = HERE / "_smoke_nb02"
shutil.rmtree(WORKDIR, ignore_errors=True)
os.makedirs(WORKDIR, exist_ok=True)


def load_cells(path):
    out, label, buf = [], None, []
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r"^# %%(?:\s+\[(markdown)\])?\s*(.*)$", line)
        if m:
            if label is not None:
                out.append((label, "\n".join(buf)))
            label = None if m.group(1) else (m.group(2) or "?")
            buf = []
        elif label is not None:
            buf.append(line)
    if label is not None:
        out.append((label, "\n".join(buf)))
    return out

PAIRS = load_cells(SRC)
CELLS = dict(PAIRS)
ORDER = [l for l, _ in PAIRS]

# ============================================================ index TONG HOP
rng = np.random.default_rng(42)
VIDS = ([f"L21_V{i:03d}" for i in range(1, 5)] + [f"L26_V{i:03d}" for i in range(1, 7)]
        + [f"L30_V{i:03d}" for i in range(1, 3)])   # 12 video -> du de test diversity cap
NKF, DIM, VDIM = 40, 8, 8

rows = []
for vi, v in enumerate(VIDS):
    for n in range(1, NKF + 1):
        rows.append({"kf_id": f"{v}#{n:03d}", "video_id": v, "n": n,
                     "frame_idx": n * 25 + vi,          # frame_idx != n (bat loi nop nham n)
                     "pts_time": float(n) * 1.5, "fps": 25.0,
                     "kf_path": f"{v}/{n:03d}.jpg", "vis_row": len(rows)})
kf = pd.DataFrame(rows)
kf["n"] = kf["n"].astype("int32"); kf["frame_idx"] = kf["frame_idx"].astype("int32")
kf["pts_time"] = kf["pts_time"].astype("float32"); kf["vis_row"] = kf["vis_row"].astype("int32")

units = pd.DataFrame([{
    "unit_id": f"cap:{r.kf_id}", "channel": "caption", "video_id": r.video_id,
    "kf_id": r.kf_id, "frame_idx": r.frame_idx, "t_start": r.pts_time, "t_end": r.pts_time,
    "text_en": f"a scene in {r.video_id} at frame {r.n}", "text_vi": None,
    "text_embed": f"scene {r.n}", "lang_native": "en", "emb_row": i, "dup_of": None,
} for i, r in enumerate(kf.itertuples(index=False))] + [{
    "unit_id": f"ocr:{r.kf_id}", "channel": "ocr", "video_id": r.video_id,
    "kf_id": r.kf_id, "frame_idx": r.frame_idx, "t_start": r.pts_time, "t_end": r.pts_time,
    "text_en": None, "text_vi": f"Xã Vạn Ninh {r.n}", "text_embed": f"Xã Vạn Ninh {r.n}",
    "lang_native": "vi", "emb_row": i, "dup_of": None,
} for i, r in enumerate(kf.itertuples(index=False))])

videos = pd.DataFrame([{"video_id": v, "title": f"Title {v}", "summary_en": f"Summary {v}"}
                       for v in VIDS])

def mkindex(n, d):
    X = rng.standard_normal((n, d)).astype("float32")
    faiss.normalize_L2(X)
    ix = faiss.IndexFlatIP(d); ix.add(X)
    return ix

NS = {
    "__name__": "__smoke__", "np": np, "pd": pd, "faiss": faiss,
    "os": os, "re": re, "io": io, "csv": csv, "json": json, "glob": glob,
    "time": time, "shutil": shutil, "zipfile": zipfile, "random": random,
    "unicodedata": unicodedata, "Path": Path, "Counter": Counter, "defaultdict": defaultdict,
    "WORK": str(WORKDIR), "SUB_DIR": str(WORKDIR / "submission"),
    "DBG_DIR": str(WORKDIR / "debug"),
    "kf": kf, "units": units, "videos": videos,
    "MAX_ROWS_PER_CSV": 100, "RRF_K": 60, "TOPK_FUSED": 1000,
    "TRAKE_TOP_VIDEOS": 3, "TRAKE_BEAM": 5, "TRAKE_ABC": (0.5, 0.35, 0.15),
    "TEAM_NAME": "team_smoke", "ROUND": 1,
    "W": {"vision": 1.0, "caption": 0.9, "ocr": 0.7, "asr": 0.6,
          "summary_prior": 0.5, "meta_prior": 0.35, "bm25_ocr_vi": 0.8,
          "bm25_caption_en": 0.5, "bm25_asr_vi": 0.55, "bm25_asr_en": 0.45,
          "bm25_meta": 0.4, "bm25_summary_en": 0.4, "object_bonus": 0.3},
    "FAISS_IDX": {"vision": mkindex(len(kf), VDIM), "caption": mkindex(len(kf), DIM)},
    "OBJ_CLS": {"Person": 0, "Dragon": 1},
}
os.makedirs(NS["DBG_DIR"], exist_ok=True)

# lookup giong Cell 3
NS.update({
    "KF2ROW":  {k: i for i, k in enumerate(kf.kf_id)},
    "KF2FIDX": dict(zip(kf.kf_id, kf.frame_idx.astype(int))),
    "KF2VID":  dict(zip(kf.kf_id, kf.video_id.astype(str))),
    "KF2PTS":  dict(zip(kf.kf_id, kf.pts_time.astype(float))),
    "VID_FIDX_SET": {v: set(g.astype(int)) for v, g in
                     kf.groupby(kf.video_id.astype(str))["frame_idx"]},
    "KF_BY_VIDEO": {v: (g.kf_id.to_numpy(), g.pts_time.to_numpy("float32"),
                        g.vis_row.to_numpy("int64"))
                    for v, g in kf.groupby(kf.video_id.astype(str), sort=False)},
    "OCR_BY_KF": dict(zip(units[units.channel == "ocr"].kf_id,
                          units[units.channel == "ocr"].text_vi)),
    "CAP_BY_KF": dict(zip(units[units.channel == "caption"].kf_id,
                          units[units.channel == "caption"].text_en)),
})
# stub cac ham goi mang
NS["embed_texts"]  = lambda ts: np.tile(np.eye(1, DIM, dtype="float32"), (len(ts), 1))
NS["clip_encode"]  = lambda ts: np.tile(np.eye(1, VDIM, dtype="float32"), (len(ts), 1))
NS["tok_vi"]       = lambda s: [w for w in re.split(r"\W+", str(s).lower()) if w]
NS["tok_en"]       = lambda s: [w for w in re.split(r"\W+", str(s).lower()) if w]
NS["nfc"]          = lambda s: "" if s is None else unicodedata.normalize("NFC", str(s)).strip()
NS["translate_en"] = lambda s: "Van Ninh commune"        # khong goi API

def run(prefix):
    label = next(l for l in ORDER if l.startswith(prefix))
    exec(compile(CELLS[label], f"<{label}>", "exec"), NS)
    return label

# nap cac cell logic thuan
for pfx in ("Cell 12a",):        # dp_align + unit test co san trong cell
    print(f">>> {run(pfx)}")
for pfx in ("Cell 9c", "Cell 10", "Cell 11a", "Cell 12b"):
    run(pfx)
# Cell 11a tu dinh nghia translate_en (goi API) -> stub lai sau khi exec
NS["translate_en"] = lambda s: "Van Ninh commune"
print(">>> nap xong: final_order · build_kis_rows · build_qa_rows · build_trake_rows\n")

# ============================================================ query + candidate gia
if QDIR.is_dir():
    src_q = load_cells(SRC)
    EVENT_RE = re.compile(r"^\s*E\s*(\d+)\s*[:.]", flags=re.M)
    real = {}
    for p in sorted(glob.glob(f"{QDIR}/*.txt")):
        stem = Path(p).stem
        real[stem] = unicodedata.normalize("NFC", open(p, encoding="utf-8").read().strip())
    kis_q = next(k for k in real if k.endswith("-kis"))
    qa_q  = "query-p1-15-qa"
    tr_q  = "query-p1-18-trake"          # ca kho nhat: BTC danh may E1/E2/E2/E4
    n_ev  = len(EVENT_RE.findall(real[tr_q]))
    print(f"Dung query THAT: {kis_q} · {qa_q} · {tr_q} (n_events={n_ev})")
else:
    real = {"query-p1-1-kis": "q", "query-p1-15-qa": "q", "query-p1-18-trake": "q"}
    kis_q, qa_q, tr_q, n_ev = "query-p1-1-kis", "query-p1-15-qa", "query-p1-18-trake", 4
    print("[SKIP] khong thay bo de that - dung query gia")

PARSED = [
    {"query_id": kis_q, "query_file": kis_q + ".txt", "type": "kis", "q_vi": real[kis_q],
     "q_en": "a scene", "visual_desc_en": "a scene", "ocr_hints": [], "named_entities": [],
     "object_classes": [], "question_en": None, "n_events": 0, "events": []},
    {"query_id": qa_q, "query_file": qa_q + ".txt", "type": "qa", "q_vi": real[qa_q],
     "q_en": "which commune", "visual_desc_en": "a banner", "ocr_hints": ["FANA"],
     "named_entities": [], "object_classes": [], "question_en": "Which commune?",
     "n_events": 0, "events": []},
    {"query_id": tr_q, "query_file": tr_q + ".txt", "type": "trake", "q_vi": real[tr_q],
     "q_en": "moments", "visual_desc_en": "moments", "ocr_hints": [], "named_entities": [],
     "object_classes": [], "question_en": None, "n_events": n_ev,
     "events": [{"idx": i + 1, "desc_vi": f"su kien {i+1}", "desc_en": f"event {i+1}",
                 "visual_desc_en": f"event {i+1}"} for i in range(n_ev)]},
]
NS["PARSED"] = PARSED

# candidate gia: diem giam dan, tron nhieu video
def fake_cand(seed):
    r = np.random.default_rng(seed)
    d = kf.sample(frac=1.0, random_state=seed).reset_index(drop=True).copy()
    d["fused_score"] = np.sort(r.random(len(d)))[::-1]
    d["rerank_text_score"] = d["fused_score"] * 0.9
    d["rerank_vlm_score"] = np.nan
    d["vlm_reason"] = None
    d["vlm_answer"] = None
    return d.sort_values("fused_score", ascending=False).reset_index(drop=True)

CAND = {p["query_id"]: fake_cand(i) for i, p in enumerate(PARSED)}
# QA: gan answer cua VLM - co bien the de test vote clustering + truncate
qad = CAND[qa_q]
answers = ["Xã Vạn Ninh", "xã vạn ninh!", "Xã Vạn Ninh", "Xã Ninh Hòa",
           "Thà làm quỷ nước Nam còn hơn làm vương đất Bắc, " * 4]   # > 100 ky tu
for i, a in enumerate(answers):
    qad.loc[i, "vlm_answer"] = a
    qad.loc[i, "rerank_vlm_score"] = 10 - i
NS["CAND"] = CAND

# ============================================================ sinh rows + writer + validator
ROWS = {}
for p in PARSED:
    f = {"kis": NS["build_kis_rows"], "qa": NS["build_qa_rows"],
         "trake": NS["build_trake_rows"]}[p["type"]]
    ROWS[p["query_id"]] = f(p)
    print(f"  {p['query_id']:24s} {p['type']:5s} {len(ROWS[p['query_id']]):3d} dong")
NS["ROWS"] = ROWS

print(f"\n>>> {run('Cell 13')}")
print(f">>> {run('Cell 14')}")
print(f">>> {run('Cell 15')}")

# ============================================================ kiem tra bo sung
print("\n" + "=" * 66)
ok = True
def c(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))

sub = WORKDIR / "submission"
c("VALIDATOR khong bao van de nao", NS["problems"] == [], str(NS["problems"][:3]))
c("dung 3 file CSV", len(list(sub.glob("*.csv"))) == 3)

kis_rows = list(csv.reader(io.StringIO((sub / f"{kis_q}.csv").read_text(encoding="utf-8"))))
c("KIS: 100 dong (dung het quota)", len(kis_rows) == 100, str(len(kis_rows)))
c("KIS: dung 2 cot", all(len(r) == 2 for r in kis_rows))
c("KIS: khong video nao > 5 frame trong 30 dong dau",
  max(Counter(r[0] for r in kis_rows[:30]).values()) <= 5,
  str(dict(Counter(r[0] for r in kis_rows[:30]))))
c("KIS: frame_idx nop la frame_idx THAT, khong phai n",
  all(int(r[1]) in NS["VID_FIDX_SET"][r[0]] for r in kis_rows))
c("KIS: khong dong trung lap", len(set(map(tuple, kis_rows))) == len(kis_rows))

qa_rows = list(csv.reader(io.StringIO((sub / f"{qa_q}.csv").read_text(encoding="utf-8"))))
c("QA: dung 3 cot", all(len(r) == 3 for r in qa_rows))
c("QA: moi answer <= 100 ky tu", all(len(r[2]) <= 100 for r in qa_rows),
  str(max(len(r[2]) for r in qa_rows)))
c("QA: answer khong rong", all(r[2].strip() for r in qa_rows))
main_ans = Counter(r[2] for r in qa_rows).most_common(1)[0][0]
c("QA: answer chinh la tieng Viet verbatim (vote clustering gom 'Xã Vạn Ninh')",
  main_ans == "Xã Vạn Ninh", repr(main_ans))
c("QA: co dong hedge tieng Anh o cuoi",
  any(r[2] == "Van Ninh commune" for r in qa_rows),
  str(sorted({r[2][:24] for r in qa_rows})))

tr_rows = list(csv.reader(io.StringIO((sub / f"{tr_q}.csv").read_text(encoding="utf-8"))))
c(f"TRAKE: dung {1+n_ev} cot (1 video_id + {n_ev} frame)",
  all(len(r) == 1 + n_ev for r in tr_rows), str({len(r) for r in tr_rows}))
c("TRAKE: frame_idx TANG DAN NGHIEM NGAT moi dong",
  all(all(int(r[i]) < int(r[i + 1]) for i in range(1, len(r) - 1)) for r in tr_rows))
c("TRAKE: moi dong cung 1 video_id va frame ton tai that",
  all(all(int(x) in NS["VID_FIDX_SET"][r[0]] for x in r[1:]) for r in tr_rows))
c("TRAKE (p1-18): n_events = 4 chu khong phai 3 - khong dedupe theo con so sau E",
  (1 + n_ev) == 5 if tr_q == "query-p1-18-trake" else True, f"n_events={n_ev}")
c("TRAKE: sinh >= 5 hypothesis khac nhau", len(tr_rows) >= 5, str(len(tr_rows)))

raw = (sub / f"{qa_q}.csv").read_bytes()
c("khong co BOM", not raw.startswith(b"\xef\xbb\xbf"))
c("line ending \\n, khong co \\r", b"\r" not in raw)
c("khong co header (dong 1 la du lieu)", raw.split(b"\n")[0].split(b",")[0].startswith(b"L"))

zp = WORKDIR / "team_smoke_round1.zip"
names = zipfile.ZipFile(zp).namelist()
c("zip: MOI entry bat dau bang 'submission/'",
  all(n.startswith("submission/") for n in names), str(names[:3]))

print("=" * 66)
print("SMOKE NB02:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
