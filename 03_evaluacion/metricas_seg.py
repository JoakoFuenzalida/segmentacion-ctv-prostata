"""Metricas por paciente (Dice, IoU, HD95 en mm, volumenes en cc) entre una carpeta de predicciones y
una de etiquetas. Reutiliza la logica de HD95/volumen de vigilante_resumen.py (mismo calculo).

Uso: python metricas_seg.py <carpeta_pred> <carpeta_labels> <salida.csv> [casos separados por coma]
El CSV incluye una fila por paciente y filas finales MEDIA, DE (desviacion estandar muestral) y MEDIANA.
"""
import csv
import glob
import os
import statistics as st
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

COLS = ["caso", "dice", "iou", "hd95_mm", "vol_gt_cc", "vol_pred_cc", "dif_vol_pct"]


def hd95(gt, pr, sp):
    # identico a vigilante_resumen.hd95: bordes = mascara menos su erosion; EDT con spacing; percentil 95 simetrico
    if not gt.any() or not pr.any():
        return float("nan")
    zz, yy, xx = np.where(gt | pr)
    m = 10
    sl = tuple(slice(max(a.min() - m, 0), a.max() + m + 1) for a in (zz, yy, xx))
    g, p = gt[sl], pr[sl]
    sg = g & ~ndi.binary_erosion(g); spp = p & ~ndi.binary_erosion(p)
    dg = ndi.distance_transform_edt(~sg, sampling=sp); dp = ndi.distance_transform_edt(~spp, sampling=sp)
    return float(max(np.percentile(dp[sg], 95), np.percentile(dg[spp], 95)))


def case_metrics(gt_path, pred_path):
    gi = sitk.ReadImage(gt_path)
    gt = sitk.GetArrayFromImage(gi) > 0
    pr = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)) > 0
    if gt.shape != pr.shape:
        raise ValueError(f"forma distinta GT {gt.shape} vs pred {pr.shape}")
    sp = gi.GetSpacing()[::-1]  # z, y, x
    tp = int((gt & pr).sum()); g = int(gt.sum()); p = int(pr.sum())
    dice = 2 * tp / (g + p) if (g + p) else float("nan")
    iou = tp / (g + p - tp) if (g + p - tp) else float("nan")
    vv = sp[0] * sp[1] * sp[2] / 1000.0
    vg, vp = g * vv, p * vv
    return {"dice": dice, "iou": iou, "hd95_mm": hd95(gt, pr, sp), "vol_gt_cc": vg, "vol_pred_cc": vp,
            "dif_vol_pct": 100 * (vp - vg) / vg if vg else float("nan")}


def evaluate(pred_dir, label_dir, cases=None):
    if cases is None:
        cases = sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    rows = []
    for c in cases:
        r = {"caso": c}
        r.update(case_metrics(os.path.join(label_dir, f"{c}.nii.gz"), os.path.join(pred_dir, f"{c}.nii.gz")))
        rows.append(r)
    return rows


def stats(rows, key):
    v = [r[key] for r in rows if r[key] == r[key]]
    if not v:
        return float("nan"), float("nan"), float("nan")
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), st.median(v)


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r["caso"]] + [f"{r[k]:.4f}" for k in COLS[1:]])
        for i, name in enumerate(("MEDIA", "DE", "MEDIANA")):
            w.writerow([name] + [f"{stats(rows, k)[i]:.4f}" for k in COLS[1:]])


def read_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["caso"] in ("MEDIA", "DE", "MEDIANA"):
                continue
            rows.append({"caso": r["caso"], **{k: float(r[k]) for k in COLS[1:]}})
    return rows


if __name__ == "__main__":
    pred_dir, label_dir, out = sys.argv[1:4]
    cases = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    rows = evaluate(pred_dir, label_dir, cases)
    write_csv(rows, out)
    for r in rows:
        print(f"{r['caso']}: Dice={r['dice']:.4f} IoU={r['iou']:.4f} HD95={r['hd95_mm']:.2f} mm "
              f"Vgt={r['vol_gt_cc']:.1f} Vpred={r['vol_pred_cc']:.1f} ({r['dif_vol_pct']:+.1f}%)")
    for k in COLS[1:]:
        m, s, md = stats(rows, k)
        print(f"{k}: media={m:.4f} DE={s:.4f} mediana={md:.4f}")
    print("CSV:", out)
