"""Audit raw BHSig+CEDAR data before building a manifest.
Usage: python -m preprocessing.audit_raw_dataset
"""
from __future__ import annotations
import collections, hashlib, json, re
from pathlib import Path
import cv2, numpy as np

DATA = Path('data'); ROOT = DATA / 'BHSig260'; REPORTS_DIR = Path('reports')
BHSIG_RE = re.compile(r'^([BH])-S-(\d+)-([GF])-\d+\.tif$', re.I)
CEDAR_ORG_RE = re.compile(r'^original_(\d+)_(\d+)\.png$')
CEDAR_FORG_RE = re.compile(r'^forgeries_(\d+)_(\d+)\.png$')

def _sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while c := f.read(8 << 20): h.update(c)
    return h.hexdigest()

def _is_img(p):
    return p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')

def _classify_bhsig(p):
    m = BHSIG_RE.match(p.name)
    if not m: return None
    lang, num, kind = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
    ds = 'bhsig_bn' if lang == 'B' else 'bhsig_hi'
    sid = f'{ds}_{num:03d}'; isf = kind == 'F'
    if p.parent.name.lower() != ('forged' if isf else 'genuine'):
        raise ValueError(f'Folder mismatch: {p}')
    return {'dataset': ds, 'signer_id': sid, 'is_forgery': isf,
            'source_path': str(p.relative_to(DATA).as_posix())}

def _classify_cedar(p):
    m = CEDAR_ORG_RE.match(p.name) or CEDAR_FORG_RE.match(p.name)
    if not m: return None
    sid = f'cedar_{int(m.group(1)):03d}'
    return {'dataset': 'cedar', 'signer_id': sid,
            'is_forgery': CEDAR_FORG_RE.match(p.name) is not None,
            'source_path': str(p.relative_to(DATA).as_posix())}

def _audit_one(p):
    try: img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    except Exception: return {'error': 'cv2_exception'}
    if img is None: return {'error': 'unreadable'}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    u = int(len(np.unique(gray)))
    return {'shape': list(gray.shape), 'unique': u,
            'mode': 'binary' if u <= 2 else 'grayscale',
            'density': float(np.mean(gray < 128))}

def main():
    if not ROOT.is_dir(): raise FileNotFoundError(str(ROOT.resolve()))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_files = [p for p in ROOT.rglob("*") if p.is_file()]
    cls, unr, junk, pfail, zbytes = [], [], [], [], []
    hashes = collections.defaultdict(list)

    for p in all_files:
        if p.stat().st_size == 0: zbytes.append(p)
        meta = _classify_bhsig(p) or _classify_cedar(p) if _is_img(p) else None
        if meta:
            info = _audit_one(p)
            if "error" in info:
                unr.append({"path": str(p), **meta})
            else:
                s = _sha256(p); hashes[s].append(p)
                cls.append({**meta, **info, "sha256": s, "size_bytes": p.stat().st_size})
        elif _is_img(p):
            pfail.append({"path": str(p), "note": "unrecognised filename"})
        else:
            junk.append({"path": str(p), "note": "non-image"})

    dups = [[str(q) for q in qs] for _, qs in hashes.items() if len(qs) > 1]
    bd, bs = {}, {}
    for r in cls:
        ds, si = r["dataset"], r["signer_id"]
        bd.setdefault(ds, {"g": 0, "f": 0, "n": 0, "ss": set()})
        bd[ds]["n"] += 1; bd[ds]["ss"].add(si)
        bs.setdefault(si, {"g": 0, "f": 0})
        k = "g" if not r["is_forgery"] else "f"
        bs[si][k] += 1; bd[ds][k] += 1

    dd, gd, ded = (collections.defaultdict(list) for _ in range(3))
    for r in cls:
        if "shape" in r:
            dd[r["dataset"]].append(r["shape"])
            gd[r["dataset"]].append(r["unique"])
            ded[r["dataset"]].append(r["density"])

    dsum = {}
    for ds in sorted(bd):
        d, g, de = dd[ds], gd[ds], ded[ds]
        dsum[ds] = {"n": bd[ds]["n"], "signers": len(bd[ds]["ss"]),
            "g": bd[ds]["g"], "f": bd[ds]["f"],
            "dh": [min(x[0] for x in d), max(x[0] for x in d)] if d else None,
            "dw": [min(x[1] for x in d), max(x[1] for x in d)] if d else None,
            "ug": round(float(np.mean(g)), 1) if g else None,
            "mode": "binary" if g and max(g) <= 2 else "grayscale" if g else None,
            "ink": round(100 * float(np.mean(de)), 2) if de else None}

    ssum = {si: {"g": v["g"], "f": v["f"], "t": v["g"] + v["f"]}
            for si, v in sorted(bs.items())}

    result = {"root": str(ROOT.resolve()), "total": len(all_files),
              "classified": len(cls), "unreadable": len(unr),
              "zero_byte": [str(p) for p in zbytes],
              "parse_failures": pfail, "junk": junk,
              "duplicates": dups, "datasets": dsum, "signers": ssum,
              "tot_signers": len(bs),
              "tot_genuine": sum(v["g"] for v in bd.values()),
              "tot_forged": sum(v["f"] for v in bd.values())}

    jp = REPORTS_DIR / "raw_audit.json"
    with jp.open("w") as f: json.dump(result, f, indent=2, default=str)
    print(f"Wrote {jp}")

    mp = REPORTS_DIR / "raw_audit.md"
    with mp.open("w") as f:
        f.write(f"# Raw Audit\n- **Root**: {result['root']}\n")
        f.write(f"- **Total**: {result['total']} files, {result['classified']} classified, "
                f"{len(unr)} unreadable, {len(zbytes)} zero, {len(junk)} junk\n\n")
        f.write("| Dataset | Files | Signers | G | F | Dims H | Dims W | Mode | Gray | Ink% |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for ds, s in dsum.items():
            h = f"{s['dh'][0]}-{s['dh'][1]}" if s['dh'] else "-"
            w = f"{s['dw'][0]}-{s['dw'][1]}" if s['dw'] else "-"
            gl = str(s['ug']) if s['ug'] is not None else "-"
            ink = f"{s['ink']}%" if s['ink'] is not None else "-"
            f.write(f"| {ds} | {s['n']} | {s['signers']} | {s['g']} | {s['f']} | {h} | {w} | {s['mode']} | {gl} | {ink} |\n")
        deficient = [si for si, v in ssum.items() if v["g"] < 2 or v["f"] < 1]
        if deficient:
            f.write(f"\n**{len(deficient)} deficient:** " + ", ".join(
                f"{s}({ssum[s]['g']}G/{ssum[s]['f']}F)" for s in deficient[:10]) + "\n")
        else:
            f.write("\nAll signers meet min (>=2G, >=1F).\n")
    print(f"Wrote {mp}")
    print(f"Done: {len(cls)} classified, {len(junk)} junk, {len(unr)} unreadable")

if __name__ == "__main__":
    main()
