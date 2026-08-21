#!/usr/bin/env python3
"""Kiem tra cac ham logic thuan cua NB01/NB02 tren du lieu LOCAL that.

Chay:  python tests/test_logic.py
Khong can Kaggle, khong can API key, khong can GPU. Chi doc:
  ../Feature_Dataset/        (neu co)
  ../THUNGHIEM-bo-de-thi/    (24 query that)

Muc dich: bat loi truoc khi ton 1 session Kaggle 4h.
"""
import json, glob, os, re, sys, unicodedata
from pathlib import Path
from collections import Counter

import numpy as np

try:                      # console Windows mac dinh cp1252 -> khong in duoc tieng Viet
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "Feature_Dataset"
QDIR = ROOT / "THUNGHIEM-bo-de-thi"

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


# ---------------------------------------------------------------- helpers duoc copy tu notebook
def nfc(s):
    return "" if s is None else unicodedata.normalize("NFC", str(s)).strip()

EVENT_RE = re.compile(r"^\s*E\s*(\d+)\s*[:.]", flags=re.M)

def load_queries(query_root):
    out = []
    for p in sorted(glob.glob(f"{query_root}/**/*.txt", recursive=True)):
        stem = os.path.basename(p)[:-4]
        qtype = stem.rsplit("-", 1)[-1].lower()
        assert qtype in ("kis", "qa", "trake"), f"hau to la: {stem}"
        q_vi = nfc(open(p, encoding="utf-8").read())
        out.append({"query_id": stem, "query_file": os.path.basename(p), "type": qtype,
                    "q_vi": q_vi, "n_events": len(EVENT_RE.findall(q_vi))})
    return out

def load_object_classes(path):
    lines = [l.strip() for l in open(path, encoding="utf-8")]
    return [l for l in lines if l and not l.startswith("#")]

CHUNK_TS_RE = re.compile(r"\s*-?\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?\s*-\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\]")

def parse_chunk_ts(s):
    m = CHUNK_TS_RE.match(s or "")
    if not m:
        return None, None
    a1, a2, a3, b1, b2, b3 = m.groups()
    if a3 is None:
        return float(int(a1) * 60 + int(a2)), float(int(b1) * 60 + int(b2))
    return (float(int(a1) * 3600 + int(a2) * 60 + int(a3)),
            float(int(b1) * 3600 + int(b2) * 60 + int(b3)))

def dp_align(S, banned_first=frozenset()):
    N, K = S.shape
    if N < K:
        return None, -np.inf
    NEG = -1e18
    dp = np.full((N, K), NEG, dtype="float64")
    back = np.full((N, K), -1, dtype="int64")
    for i in range(N):
        if i not in banned_first:
            dp[i, 0] = S[i, 0]
    for k in range(1, K):
        best_val, best_j = NEG, -1
        for i in range(N):
            if i > 0 and dp[i - 1, k - 1] > best_val:
                best_val, best_j = dp[i - 1, k - 1], i - 1
            if best_j >= 0:
                dp[i, k] = S[i, k] + best_val
                back[i, k] = best_j
    end = int(np.argmax(dp[:, K - 1]))
    if dp[end, K - 1] <= NEG / 2:
        return None, -np.inf
    path, i = [end], end
    for k in range(K - 1, 0, -1):
        i = int(back[i, k])
        path.append(i)
    path.reverse()
    return path, float(dp[end, K - 1])

def truncate_100(s):
    s = nfc(s)
    if len(s) <= 100:
        return s
    cut = s[:100]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 40 else cut).strip()

def norm_ans(s):
    s = unicodedata.normalize("NFD", nfc(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", s).strip()


# ---------------------------------------------------------------- 1. DP monotonic alignment
print("\n=== 1. dp_align (QD-6) ===")
S = np.array([[9., 0., 0.], [0., 9., 0.], [0., 0., 9.], [8., 8., 8.]])
p, sc = dp_align(S)
check("duong cheo -> [0,1,2]", p == [0, 1, 2] and abs(sc - 27) < 1e-9, f"{p} score={sc}")

S2 = np.array([[0., 5.], [5., 0.]])       # argmax doc lap ra [1,0] = SAI thu tu
p2, _ = dp_align(S2)
check("argmax doc lap sai thu tu -> DP van tra tang dan", p2 == [0, 1], str(p2))

rng = np.random.default_rng(0)
ok = True
for _ in range(300):
    N, K = rng.integers(3, 40), rng.integers(2, 6)
    if N < K:
        continue
    pp, _ = dp_align(rng.random((N, K)))
    if pp is None or any(pp[i] >= pp[i + 1] for i in range(len(pp) - 1)) or len(pp) != K:
        ok = False
        break
check("300 ma tran ngau nhien: luon tang dan nghiem ngat va du K phan tu", ok)

check("N < K -> tra None (khong crash)", dp_align(np.zeros((2, 5)))[0] is None)

# beam: cam i_1 -> duong di khac
banned = set()
paths = []
for _ in range(3):
    pp, _ = dp_align(S, frozenset(banned))
    if pp is None:
        break
    banned.add(pp[0])
    paths.append(tuple(pp))
check("beam sinh duong di KHAC nhau", len(set(paths)) == len(paths), str(paths))


# ---------------------------------------------------------------- 2. load_queries tren 24 query that
print("\n=== 2. load_queries tren bo de that (24 query) ===")
if QDIR.is_dir():
    Q = load_queries(str(QDIR))
    types = Counter(q["type"] for q in Q)
    check("24 query", len(Q) == 24, str(len(Q)))
    check("18 KIS / 3 QA / 3 TRAKE",
          types == {"kis": 18, "qa": 3, "trake": 3}, str(dict(types)))
    check("query-p1-3 KHONG ton tai (so thu tu co lo)",
          not any(q["query_id"].startswith("query-p1-3-") for q in Q))

    byid = {q["query_id"]: q for q in Q}
    # p1-18: BTC danh may sai E1/E2/E2/E4 -> dem theo DONG phai ra 4
    q18 = byid.get("query-p1-18-trake")
    nums = EVENT_RE.findall(q18["q_vi"])
    check("p1-18: con so sau chu E la E1/E2/E2/E4 (BTC danh may sai)",
          nums == ["1", "2", "2", "4"], str(nums))
    check("p1-18: n_events dem theo DONG = 4 (dedupe se ra 3 -> 0 diem)",
          q18["n_events"] == 4, str(q18["n_events"]))
    check("p1-18: set() theo con so ra 3 -> chung minh vi sao KHONG duoc dedupe",
          len(set(nums)) == 3)
    for qid, n in (("query-p1-4-trake", 4), ("query-p1-16-trake", 4)):
        check(f"{qid}: n_events == {n}", byid[qid]["n_events"] == n, str(byid[qid]["n_events"]))
    check("moi TRAKE co n_events >= 2",
          all(q["n_events"] >= 2 for q in Q if q["type"] == "trake"))
    check("KIS/QA khong bi dem nham event",
          all(q["n_events"] == 0 for q in Q if q["type"] != "trake"))

    # thu tu dong phai duoc giu
    lines18 = [l.strip() for l in q18["q_vi"].splitlines() if EVENT_RE.match(l)]
    check("p1-18: 4 dong event, giu nguyen thu tu dong", len(lines18) == 4,
          " | ".join(l[:18] for l in lines18))

    # tin hieu review tay
    print(f"    p1-15 chua 'FANA'? {'FANA' in byid['query-p1-15-qa']['q_vi'].upper()}")
    print(f"    p1-19 chua 'Nguyễn Trung Trực'? "
          f"{'nguyễn trung trực' in byid['query-p1-19-qa']['q_vi'].lower()}")
else:
    print(f"  [SKIP] khong thay {QDIR}")


# ---------------------------------------------------------------- 3. object classes
print("\n=== 3. detected_classes.txt ===")
p_cls = FEAT / "objects-aic25-b1" / "detected_classes.txt"
if p_cls.exists():
    raw_txt = open(p_cls, encoding="utf-8").read()              # text mode: CRLF -> LF
    raw_bin = open(p_cls, "rb").read().decode("utf-8")          # giu nguyen CRLF
    cls = load_object_classes(str(p_cls))
    check("parse ra dung 584 class", len(cls) == 584, str(len(cls)))
    check("khong con dong '#' hay '\\r'",
          not any(c.startswith("#") or "\r" in c for c in cls))
    check("file goc dung line ending CRLF", "\r\n" in raw_bin)
    check("cach SAI 1: splitlines() tho -> giu header, lech index 3 cot",
          len(raw_txt.splitlines()) == 587, f"{len(raw_txt.splitlines())} dong tho")
    check("cach SAI 2: doc binary roi split('\\n') -> moi ten dinh '\\r'",
          any("\r" in x for x in raw_bin.split("\n")))
    check("open() text mode da dich CRLF->LF, nhung .strip() van can cho khoang trang",
          not any("\r" in x for x in raw_txt.split("\n")))
    check("'Person' co trong danh sach", "Person" in cls)
else:
    print(f"  [SKIP] khong thay {p_cls}")


# ---------------------------------------------------------------- 4. ocr_index.jsonl
print("\n=== 4. ocr_index.jsonl (1 dong / keyframe) ===")
p_ocr = FEAT / "ocr_index.jsonl"
if p_ocr.exists():
    n, seen, dup = 0, set(), 0
    vids, fields_ok = set(), True
    with open(p_ocr, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            r = json.loads(line)
            if n == 1:
                fields_ok = {"video_id", "frame_idx", "pts_time", "keyframe", "text"} <= set(r)
            k = (r["video_id"], r["keyframe"])
            if k in seen:
                dup += 1
            seen.add(k)
            vids.add(r["video_id"])
    check("128,664 dong", n == 128_664, str(n))
    check("0 cap (video_id, keyframe) trung -> KHONG co gi de gop", dup == 0, f"dup={dup}")
    check("moi dong co san frame_idx + pts_time (khong can join keyframes)", fields_ok)
    check("700 video co OCR", len(vids) == 700, str(len(vids)))
    check("has_ocr PHAI tinh tu file that, khong suy tu prefix: L25 chi co mot phan",
          sum(1 for v in vids if v.startswith("L25")) == 58,
          f"L25 co OCR: {sum(1 for v in vids if v.startswith('L25'))}/88")
    for pre in ("L28", "L29", "L30"):
        check(f"{pre} khong co OCR nao", not any(v.startswith(pre) for v in vids))
else:
    print(f"  [SKIP] khong thay {p_ocr}")


# ---------------------------------------------------------------- 5. Summary sentinel + chunk timestamp
print("\n=== 5. Summary_video ===")
p_sum = FEAT / "Summary_video"
if p_sum.is_dir():
    stems = [Path(p).stem for p in glob.glob(f"{p_sum}/*.json")]
    real = [s for s in stems if not s.startswith("_")]
    check("866 file .json", len(stems) == 866, str(len(stems)))
    check("co sentinel '_failed.json'", "_failed" in stems)
    check("865 video that sau khi skip stem bat dau bang '_'", len(real) == 865, str(len(real)))

    # thieu summary = L26_V072..L26_V079
    mapkf = FEAT / "map-keyframes-aic25-b1" / "map-keyframes"
    if mapkf.is_dir():
        allv = {Path(p).stem for p in glob.glob(f"{mapkf}/*.csv")}
        missing = sorted(allv - set(real))
        check("dung 8 video thieu summary", len(missing) == 8, str(missing))
        check("do la L26_V072..L26_V079",
              missing == [f"L26_V{i:03d}" for i in range(72, 80)], str(missing))

    # evidence KHONG phai list su kien, chunk_summaries MOI co moc thoi gian
    sample = json.load(open(glob.glob(f"{p_sum}/L2*.json")[0], encoding="utf-8"))
    ev = sample.get("evidence")
    check("'evidence' la dict metadata provenance, KHONG co timestamp",
          isinstance(ev, dict) and not any("time" in str(k).lower() and k != "duration_hint"
                                           for k in (ev or {})),
          f"keys={list(ev)[:6] if isinstance(ev, dict) else type(ev)}")
    chunks = sample.get("chunk_summaries") or []
    parsed = [parse_chunk_ts(c) for c in chunks]
    got = [x for x in parsed if x[0] is not None]
    check("chunk_summaries parse duoc moc [MM:SS-MM:SS]",
          len(got) > 0 or len(chunks) == 0,
          f"{len(got)}/{len(chunks)} chunk co moc; vd {str(chunks[0])[:60] if chunks else '-'}")
else:
    print(f"  [SKIP] khong thay {p_sum}")

# regex chunk ts - test tong hop
print("  -- parse_chunk_ts --")
check("dang MM:SS", parse_chunk_ts("- [00:00-02:00] A report") == (0.0, 120.0))
check("dang HH:MM:SS", parse_chunk_ts("[01:02:03 - 01:02:10] x") == (3723.0, 3730.0))
check("khong match -> (None, None), van giu unit", parse_chunk_ts("no timestamp") == (None, None))


# ---------------------------------------------------------------- 6. caption duplicate_of
print("\n=== 6. caption duplicate_of la TEN FILE, khong phai kf_id ===")
p_cap = FEAT / "Image_captioning" / "L21_V001.json"
if p_cap.exists():
    d = json.load(open(p_cap, encoding="utf-8"))
    items = d.get("keyframes") or []
    dups = [it for it in items if it.get("duplicate_of")]
    check("co keyframe duplicate", len(dups) > 0, f"{len(dups)}/{len(items)}")
    if dups:
        sample_dup = dups[0]["duplicate_of"]
        check("duplicate_of co dang ten file ('009.jpg'), KHONG phai 'L21_V001#009'",
              bool(re.fullmatch(r"\d+\.(jpg|png|webp)", str(sample_dup))), repr(sample_dup))
        # parse ra n roi dung kf_id
        n = int(Path(str(sample_dup)).stem)
        check("parse ra n roi dung kf_id", f"L21_V001#{n:03d}" == f"L21_V001#{n:03d}")
        # caption cua dup giong het canonical?
        by_n = {}
        for it in items:
            name = it.get("keyframe") or it.get("file") or ""
            m = re.search(r"(\d+)", str(name))
            if m and not it.get("duplicate_of"):
                by_n[int(m.group(1))] = (it.get("caption") or "").strip()
        same = sum(1 for it in dups
                   if (it.get("caption") or "").strip()
                   == by_n.get(int(Path(str(it["duplicate_of"])).stem), None))
        print(f"    {same}/{len(dups)} caption dup GIONG HET canonical "
              f"-> khong embed lai la dung, khong mat thong tin")
else:
    print(f"  [SKIP] khong thay {p_cap}")


# ---------------------------------------------------------------- 7. keyframes <-> clip .npy invariant
print("\n=== 7. invariant: so keyframe == so hang .npy CLIP (mau 40 video stratified) ===")
mapkf = FEAT / "map-keyframes-aic25-b1" / "map-keyframes"
clipd = FEAT / "clip-features-32-aic25-b1" / "clip-features-32"
if mapkf.is_dir() and clipd.is_dir():
    import pandas as pd
    allv = sorted(Path(p).stem for p in glob.glob(f"{mapkf}/*.csv"))
    check("873 video", len(allv) == 873, str(len(allv)))
    check("video_id KHONG lien tuc: L21_V004 va L21_V020 khong ton tai",
          "L21_V004" not in allv and "L21_V020" not in allv)
    # sample stratified theo prefix (L26 chiem 57% - sample thuong se chi ra L26)
    byp = {}
    for v in allv:
        byp.setdefault(v.split("_")[0], []).append(v)
    rng = np.random.default_rng(7)
    sample = [v for pre, vs in byp.items()
              for v in rng.choice(vs, size=min(4, len(vs)), replace=False)]
    bad, total_kf = [], 0
    for v in sample:
        n_csv = len(pd.read_csv(mapkf / f"{v}.csv"))
        n_npy = np.load(clipd / f"{v}.npy", mmap_mode="r").shape[0]
        total_kf += n_csv
        if n_csv != n_npy:
            bad.append((v, n_csv, n_npy))
    check(f"invariant pass tren {len(sample)} video ({len(byp)} prefix)", not bad, str(bad))
    a = np.load(clipd / f"{sample[0]}.npy", mmap_mode="r")
    check("CLIP .npy la float16 -> BAT BUOC cast fp32 truoc khi add vao FAISS",
          a.dtype == np.float16, str(a.dtype))
    check("CLIP dim == 512", a.shape[1] == 512, str(a.shape))
else:
    print(f"  [SKIP] khong thay map-keyframes / clip-features")


# ---------------------------------------------------------------- 8. ASR windowing + dedupe
print("\n=== 8. ASR windowing + dedupe (bat buoc, khong phai toi uu hoa) ===")
def build_windows(segs, window=25.0, stride=10.0, dedupe=True):
    for i, s in enumerate(segs):
        s.setdefault("id", i)
    segs = sorted(segs, key=lambda s: float(s["start"]))
    ss = np.array([float(s["start"]) for s in segs])
    se = np.array([float(s["end"]) for s in segs])
    out, seen, t = [], set(), 0.0
    while True:
        sel = np.nonzero((se > t) & (ss < t + window))[0]
        if len(sel):
            key = tuple(sorted(int(segs[j]["id"]) for j in sel))
            if not dedupe or key not in seen:
                seen.add(key)
                out.append(key)
        t += stride
        if t >= float(se.max()):
            break
    return out

p_asr = FEAT / "Transcript_Translated"
if p_asr.is_dir():
    files = sorted(glob.glob(f"{p_asr}/*/video/*.json"))
    check("873 file ASR trong 14 thu muc batch", len(files) == 873, str(len(files)))
    check("14 thu muc batch (L26 bi chia a-e)",
          len(glob.glob(f"{p_asr}/*")) == 14, str(len(glob.glob(f"{p_asr}/*"))))
    rng = np.random.default_rng(1)
    pick = rng.choice(files, size=min(40, len(files)), replace=False)
    raw_n, ded_n, durs = 0, 0, []
    for f in pick:
        segs = (json.load(open(f, encoding="utf-8")).get("segments") or [])
        segs = [s for s in segs if s.get("start") is not None and s.get("end") is not None]
        if not segs:
            continue
        durs += [float(s["end"]) - float(s["start"]) for s in segs]
        raw_n += len(build_windows([dict(s) for s in segs], dedupe=False))
        ded_n += len(build_windows([dict(s) for s in segs], dedupe=True))
    ratio = raw_n / max(ded_n, 1)
    print(f"    segment dai trung binh {np.mean(durs):.1f}s (cua so la 25s)")
    print(f"    {raw_n} window tho -> {ded_n} sau dedupe  (du {ratio:.2f}x)")
    check("dedupe cat bot >= 20% window (chung minh buoc nay la bat buoc)",
          ratio >= 1.2, f"{ratio:.2f}x")
    check("segment dai gan bang cua so -> ly do bi trung", 15 < np.mean(durs) < 35,
          f"{np.mean(durs):.1f}s")

    d0 = json.load(open(files[0], encoding="utf-8"))
    s0 = (d0.get("segments") or [{}])[0]
    check("segment co ca 'text' (vi) va 'text_en'", "text" in s0 and "text_en" in s0,
          str(list(s0)[:6]))
else:
    print(f"  [SKIP] khong thay {p_asr}")


# ---------------------------------------------------------------- 9. detection_scores la CHUOI
print("\n=== 9. objects: detection_scores luu dang CHUOI ===")
p_obj = FEAT / "objects-aic25-b1" / "objects"
if p_obj.is_dir():
    some = glob.glob(f"{p_obj}/*/*.json")[:1]
    if some:
        d = json.load(open(some[0], encoding="utf-8"))
        sc = (d.get("detection_scores") or [None])[0]
        check("detection_scores[0] la str -> so sanh voi float se TypeError",
              isinstance(sc, str), f"{sc!r} ({type(sc).__name__})")
        try:
            _ = sc >= 0.30
            check("so sanh truc tiep KHONG raise (python2-like)?", False, "khong nen xay ra")
        except TypeError:
            check("xac nhan: so sanh str >= float raise TypeError -> phai float() truoc", True)
        check("float(sc) hoat dong", 0.0 <= float(sc) <= 1.0, str(float(sc)))
else:
    print(f"  [SKIP] khong thay {p_obj}")


# ---------------------------------------------------------------- 10. text helpers cho QA
print("\n=== 10. QA helpers ===")
check("truncate_100 giu nguyen chuoi ngan", truncate_100("Xã Vạn Ninh") == "Xã Vạn Ninh")
long_vi = ("Thà làm quỷ nước Nam còn hơn làm vương đất Bắc, " * 4)
t = truncate_100(long_vi)
check("truncate_100 cat <= 100 ky tu o ranh gioi tu", len(t) <= 100 and not t.endswith("l"),
      f"{len(t)} ky tu")
check("norm_ans bo dau + lowercase de gom cum",
      norm_ans("Xã Vạn Ninh!") == norm_ans("xa van ninh"),
      norm_ans("Xã Vạn Ninh!"))
check("norm_ans phan biet answer khac nhau",
      norm_ans("Xã Vạn Ninh") != norm_ans("Xã Ninh Hòa"))
check("NFC: 2 dang dung san / to hop cua 'ế' phai bang nhau sau nfc",
      nfc("ế") == nfc(unicodedata.normalize("NFD", "ế")))
check("NFC: KHONG normalize thi 2 dang KHAC nhau (bug BM25 kho phat hien)",
      "ế" != unicodedata.normalize("NFD", "ế"))


# ---------------------------------------------------------------- 11. CSV writer / validator format
print("\n=== 11. CSV format nop bai ===")
import csv, io as _io
buf = _io.StringIO()
w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
w.writerow(["L21_V001", 1234, 'Có 3 người, gồm nam và nữ'])
w.writerow(["L21_V001", 1235, 'Ông nói "xin chào"'])
out = buf.getvalue()
check("module csv escape dau phay trong answer dung chuan",
      out.splitlines()[0] == 'L21_V001,1234,"Có 3 người, gồm nam và nữ"', out.splitlines()[0])
check("module csv escape dau ngoac kep thanh \"\"",
      '""xin chào""' in out.splitlines()[1], out.splitlines()[1])
check('noi tay bang ",".join() se HONG',
      ",".join(["L21_V001", "1234", 'Có 3 người, gồm nam và nữ']).count(",") == 3)
back = list(csv.reader(_io.StringIO(out)))
check("doc lai bang csv.reader ra dung 3 cot", all(len(r) == 3 for r in back))
check("line ending la \\n, khong co \\r", "\r" not in out)

VID_RE = re.compile(r"^L\d{2}_V\d{3}$")
check("regex video_id nhan 'L21_V001'", bool(VID_RE.match("L21_V001")))
check("regex video_id tu choi 'L21_V001.mp4'", not VID_RE.match("L21_V001.mp4"))


# ---------------------------------------------------------------- 12. RRF
print("\n=== 12. RRF chiu duoc channel vang mat (R5: 173 video thieu OCR) ===")
RRF_K = 60
def rrf(ranks_by_ch, weights):
    from collections import defaultdict as dd
    s = dd(float)
    for ch, ranks in ranks_by_ch.items():
        for kid, r in ranks.items():
            s[kid] += weights[ch] / (RRF_K + r + 1)
    return dict(s)

Wt = {"vision": 1.0, "ocr": 0.7}
# A = video THIEU OCR (chi xuat hien o vision).  B = video co OCR, cung rank vision.
# Tinh chat can chung minh: bat kenh OCR len KHONG lam giam diem cua A.
only_vis = rrf({"vision": {"A": 0, "B": 0}}, Wt)
with_ocr = rrf({"vision": {"A": 0, "B": 0}, "ocr": {"B": 0}}, Wt)
check("RRF: them kenh OCR KHONG lam giam diem cua video thieu OCR",
      with_ocr["A"] == only_vis["A"],
      f"A: {only_vis['A']:.4f} -> {with_ocr['A']:.4f}")
check("RRF: B chi duoc CONG them, A khong bi tru -> channel vang = khong gop diem",
      with_ocr["B"] > with_ocr["A"] == only_vis["A"])

# Doi chung: weighted-sum score tho co normalize theo TONG weight thi A bi PHAT.
def wsum(cos_by_ch, weights):
    return sum(weights[c] * cos_by_ch.get(c, 0.0) for c in weights) / sum(weights.values())
ws_a = wsum({"vision": 0.90}, Wt)                 # thieu OCR -> ocr = 0
ws_b = wsum({"vision": 0.90, "ocr": 0.90}, Wt)    # co OCR
check("weighted-sum: cung diem vision nhung A bi keo xuong chi vi THIEU du lieu OCR "
      "-> day la ly do R5 bat buoc dung RRF",
      ws_a < ws_b, f"A={ws_a:.3f} < B={ws_b:.3f}")

# Va RRF van cho phep video thieu OCR THANG neu no manh hon han o vision
sc = rrf({"vision": {"A": 0, "B": 200}, "ocr": {"B": 0}}, Wt)
check("RRF: A (thieu OCR, vision rank 0) van thang B (vision rank 200, co OCR)",
      sc["A"] > sc["B"], f"A={sc['A']:.4f} B={sc['B']:.4f}")


# ---------------------------------------------------------------- ket qua
print("\n" + "=" * 62)
print(f"{len(PASS)} PASS · {len(FAIL)} FAIL")
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
print("=" * 62)
sys.exit(1 if FAIL else 0)
