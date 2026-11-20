"""Informe comparativo: linea base (nnUNetTrainer_250epochs) vs sin aumento de datos (nnUNetTrainerNoDA_250epochs).
2d, fold 0, mismo plan y mismo split. Genera _nnunet_resumen/COMPARACION_baseline_vs_noDA.md y .html
(+ figuras en _nnunet_resumen/figuras_comparacion/). Rutas configurables por argumentos (para pruebas).
"""
import argparse
import base64
import datetime as dt
import glob
import html as H
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metricas_seg import read_csv  # noqa: E402

W = r"C:\Users\joaki\Documents\Tesis_segmentacion"
RES = os.path.join(W, "nnUNet_results", "Dataset001_ProstataCTV")
C_B, C_N = "#2a6fdb", "#d93025"  # azul = linea base, rojo = sin aumento
OUTLIER = "PROS_016"


def fmt(x, nd=3):
    if x is None or x != x:
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def parse_log(fold):
    epochs, cur = {}, None
    for path in sorted(glob.glob(os.path.join(fold, "training_log_*.txt"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r": Epoch (\d+)\s*$", line)
                if m:
                    cur = int(m.group(1)); epochs.setdefault(cur, {}); continue
                if cur is None:
                    continue
                for key, pat in (("train_loss", r"train_loss (-?[\d.]+)"), ("val_loss", r"val_loss (-?[\d.]+)"),
                                 ("time", r"Epoch time: ([\d.]+) s")):
                    m = re.search(pat, line)
                    if m:
                        epochs[cur][key] = float(m.group(1))
                m = re.search(r"Pseudo dice \[(.*)\]", line)
                if m:
                    vals = re.findall(r"(-?\d+\.\d+|-?\d+)\)", m.group(1)) or re.findall(r"-?\d+\.\d+", m.group(1))
                    if vals:
                        epochs[cur]["pseudo_dice"] = float(vals[0])
    return {k: v for k, v in epochs.items() if "time" in v}


def ema(vals):  # igual que nnU-Net: ema = 0,9*ema + 0,1*valor, partiendo del primer valor
    out, e = [], None
    for v in vals:
        if v is not None:
            e = v if e is None else 0.9 * e + 0.1 * v
        out.append(e)
    return out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
    return plt


def fig_loss(eb, en, path):
    plt = _plt(); fig, ax = plt.subplots(figsize=(10, 5.5))
    for e, c, name in ((eb, C_B, "Línea base (con aumento)"), (en, C_N, "Sin aumento")):
        if not e:
            continue
        ks = sorted(e)
        ax.plot(ks, [e[k].get("train_loss") for k in ks], color=c, lw=2, label=f"{name} – entrenamiento")
        ax.plot(ks, [e[k].get("val_loss") for k in ks], color=c, lw=2, ls="--", label=f"{name} – validación")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (Dice + CE; más abajo = mejor)")
    ax.set_title("Pérdida de entrenamiento (continua) y validación (discontinua)"); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_gap(eb, en, path):
    plt = _plt(); fig, ax = plt.subplots(figsize=(10, 4.5))
    for e, c, name in ((eb, C_B, "Línea base (con aumento)"), (en, C_N, "Sin aumento")):
        if not e:
            continue
        ks = [k for k in sorted(e) if "train_loss" in e[k] and "val_loss" in e[k]]
        ax.plot(ks, [e[k]["val_loss"] - e[k]["train_loss"] for k in ks], color=c, lw=1.8, label=name)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("Época"); ax.set_ylabel("val_loss − train_loss")
    ax.set_title("Brecha entre pérdida de validación y de entrenamiento (más alta = más sobreajuste)")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_ema(eb, en, path):
    plt = _plt(); fig, ax = plt.subplots(figsize=(10, 5.5))
    for e, c, name in ((eb, C_B, "Línea base (con aumento)"), (en, C_N, "Sin aumento")):
        if not e:
            continue
        ks = sorted(e); pd = [e[k].get("pseudo_dice") for k in ks]
        ax.plot(ks, pd, "o", ms=2.5, color=c, alpha=.25)
        ax.plot(ks, ema(pd), color=c, lw=2.5, label=f"{name} – EMA pseudo-Dice")
    ax.set_ylim(0, 1); ax.set_xlabel("Época"); ax.set_ylabel("Pseudo-Dice en validación")
    ax.set_title("Pseudo-Dice por época (puntos) y su EMA (línea)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_grouped(rb, rn, key, title, ylabel, path, nd=3):
    import numpy as np
    plt = _plt()
    cases = sorted({r["caso"] for r in rb} | {r["caso"] for r in rn})
    db = {r["caso"]: r[key] for r in rb}; dn = {r["caso"]: r[key] for r in rn}
    x = np.arange(len(cases)); w = .38
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(cases) + 2), 5.5))
    vb = [db.get(c, float("nan")) for c in cases]; vn = [dn.get(c, float("nan")) for c in cases]
    ax.bar(x - w / 2, vb, w, color=C_B, label="Línea base (con aumento)")
    ax.bar(x + w / 2, vn, w, color=C_N, label="Sin aumento")
    for i in range(len(cases)):
        for v, dx in ((vb[i], -w / 2), (vn[i], w / 2)):
            if v == v:
                ax.text(i + dx, v, fmt(v, nd), ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(cases, rotation=30)
    allv = [v for v in vb + vn if v == v]
    ax.set_ylim(0, 1.3 if key == "dice" else (max(allv) * 1.35 if allv else 1))
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(loc="upper center", ncol=2, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def stat3(rows, key, absval=False):
    v = [abs(r[key]) if absval else r[key] for r in rows if r[key] == r[key]]
    if not v:
        return float("nan"), float("nan"), float("nan"), 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), st.median(v), len(v)


def final_gap(e, last=10):
    ks = [k for k in sorted(e) if "train_loss" in e[k] and "val_loss" in e[k]]
    if not ks:
        return None
    k = ks[-1]; tail = ks[-last:]
    return {"epoch": k, "train": e[k]["train_loss"], "val": e[k]["val_loss"], "gap": e[k]["val_loss"] - e[k]["train_loss"],
            "gap_mean_last": st.mean(e[j]["val_loss"] - e[j]["train_loss"] for j in tail), "n_last": len(tail)}


def wilcoxon(rb, rn, key):
    try:
        from scipy.stats import wilcoxon as wx
        db = {r["caso"]: r[key] for r in rb}
        pares = [(db[r["caso"]], r[key]) for r in rn if r["caso"] in db and db[r["caso"]] == db[r["caso"]] and r[key] == r[key]]
        if len(pares) < 5 or all(a == b for a, b in pares):
            return None
        return float(wx([a for a, _ in pares], [b for _, b in pares]).pvalue)
    except Exception:  # noqa
        return None


# ----------------------------------------------------------------------------- informe
def build(args):
    figdir = os.path.join(args.out_dir, "figuras_comparacion"); os.makedirs(figdir, exist_ok=True)
    eb, en = parse_log(args.fold_b), parse_log(args.fold_n)
    load = lambda p: read_csv(p) if p and os.path.exists(p) else []  # noqa: E731
    vb, vn, tb, tn = load(args.val_b), load(args.val_n), load(args.test_b), load(args.test_n)
    final_n = os.path.exists(os.path.join(args.fold_n, "checkpoint_final.pth"))

    figs = {}
    def F(name, fn, *a):  # noqa: E306
        p = os.path.join(figdir, name)
        try:
            fn(*a, p); figs[name] = p
        except Exception as ex:  # noqa
            print("figura fallida", name, repr(ex))
    F("C1_perdida_superpuesta.png", fig_loss, eb, en)
    F("C2_brecha_sobreajuste.png", fig_gap, eb, en)
    F("C3_ema_pseudo_dice.png", fig_ema, eb, en)
    if vb or vn:
        F("C4_dice_validacion.png", lambda a, b, p: fig_grouped(a, b, "dice", "Dice por paciente – VALIDACIÓN (7, fold 0)", "Dice (1 = perfecto)", p), vb, vn)
        F("C6_hd95_validacion.png", lambda a, b, p: fig_grouped(a, b, "hd95_mm", "HD95 por paciente – VALIDACIÓN", "HD95 en mm (más bajo = mejor)", p, 1), vb, vn)
    if tb or tn:
        F("C5_dice_test.png", lambda a, b, p: fig_grouped(a, b, "dice", "Dice por paciente – TEST (8, nunca usados)", "Dice (1 = perfecto)", p), tb, tn)
        F("C7_hd95_test.png", lambda a, b, p: fig_grouped(a, b, "hd95_mm", "HD95 por paciente – TEST", "HD95 en mm (más bajo = mejor)", p, 1), tb, tn)

    md = []
    A = md.append
    A("# Comparación: línea base vs sin aumento de datos — nnU-Net v2, 2d, fold 0")
    A(f"_Generado automáticamente: {dt.datetime.now():%d/%m/%Y %H:%M}_")
    A("")
    if not final_n:
        A("> ⚠️ **El entrenamiento sin aumento NO terminó** (no existe `checkpoint_final.pth`). "
          "Las tablas del modelo nuevo pueden estar vacías o incompletas. Revisar `_nnunet_logs\\10_train_2d_fold0_noDA.log` y `11_post_noDA.log`.")
        A("")
    A("## Qué se comparó")
    A("| | Línea base | Sin aumento |")
    A("|---|---|---|")
    A("| Trainer | `nnUNetTrainer_250epochs` | `nnUNetTrainerNoDA_250epochs` |")
    A("| Configuración / plan | 2d · nnUNetPlans | 2d · nnUNetPlans (el mismo) |")
    A("| Split | fold 0 de `splits_final.json` (25 entr. / 7 val.) | el mismo fold 0 |")
    A("| Épocas | 250 | 250 |")
    A("| Aumento de datos en entrenamiento | Sí (rotación, escala, ruido, blur, brillo, contraste, gamma, espejo…) | **No** (solo los mismos parches sin transformar) |")
    A("| Espejo en inferencia (TTA) | Sí (ejes 0 y 1) | **No** |")
    tb_ = [e[k]["time"] for e in (eb,) for k in e]; tn_ = [e[k]["time"] for e in (en,) for k in e]
    A(f"| Épocas completadas | {len(eb)} | {len(en)} |")
    A(f"| Tiempo medio por época | {fmt(st.mean(tb_), 0) if tb_ else '—'} s | {fmt(st.mean(tn_), 0) if tn_ else '—'} s |")
    A(f"| Tiempo total de entrenamiento | {fmt(sum(tb_) / 3600, 1) if tb_ else '—'} h | {fmt(sum(tn_) / 3600, 1) if tn_ else '—'} h |")
    A("")

    A("## Nota metodológica (diferencia de protocolo)")
    A("`nnUNetTrainerNoDA` desactiva el aumento de datos **y además el espejo**: fija `inference_allowed_mirroring_axes = None`, "
      "que queda guardado en el checkpoint. Por eso, tanto la validación final como la predicción del test del modelo nuevo se hacen "
      "**sin test-time augmentation (TTA) por espejo**, mientras que la línea base promedia la predicción original con sus versiones "
      "espejadas (4 combinaciones en 2d). La comparación mide, por lo tanto, el efecto conjunto de *quitar el aumento en el entrenamiento* "
      "y *quitar el espejo en la inferencia*. Si se quiere aislar solo el efecto del aumento, habría que repredecir la línea base con "
      "`--disable_tta` (o, a la inversa, no es posible añadir TTA al modelo nuevo sin cambiar el checkpoint).")
    A("")

    def case_table(rb, rn, titulo):
        A(f"## Tabla por paciente — {titulo}")
        if not rb and not rn:
            A("_Sin datos._"); A(""); return
        db = {r["caso"]: r for r in rb}; dn = {r["caso"]: r for r in rn}
        A("| Paciente | Dice base | Dice sin aum. | Δ Dice | HD95 base (mm) | HD95 sin aum. (mm) | Vol. médico (cc) | Vol. base (cc) | Vol. sin aum. (cc) |")
        A("|---|---|---|---|---|---|---|---|---|")
        nan = float("nan")
        for c in sorted(set(db) | set(dn)):
            b, n = db.get(c, {}), dn.get(c, {})
            d = n.get("dice", nan) - b.get("dice", nan)
            vg = b.get("vol_gt_cc", n.get("vol_gt_cc", nan))
            A(f"| {c} | {fmt(b.get('dice', nan))} | {fmt(n.get('dice', nan))} | {('+' if d > 0 else '') + fmt(d) if d == d else '—'} | "
              f"{fmt(b.get('hd95_mm', nan), 1)} | {fmt(n.get('hd95_mm', nan), 1)} | {fmt(vg, 1)} | {fmt(b.get('vol_pred_cc', nan), 1)} | {fmt(n.get('vol_pred_cc', nan), 1)} |")
        A("")

    case_table(vb, vn, "VALIDACIÓN (7 pacientes del fold 0)")
    case_table(tb, tn, "TEST (8 pacientes, PROS_033–040)")

    A("## Resumen estadístico")
    A("Media ± desviación estándar (DE muestral) y mediana. Δvol absoluta = diferencia absoluta de volumen respecto del médico, en %.")
    A("")
    A("| Conjunto | Métrica | Línea base: media ± DE | Línea base: mediana | Sin aumento: media ± DE | Sin aumento: mediana | p (Wilcoxon pareado)* |")
    A("|---|---|---|---|---|---|---|")
    sets = [("Validación (7)", vb, vn), (f"Validación sin {OUTLIER} (6)", [r for r in vb if r["caso"] != OUTLIER], [r for r in vn if r["caso"] != OUTLIER]),
            ("Test (8)", tb, tn)]
    summary = {}
    for name, rb, rn in sets:
        for key, lab, nd, absval in (("dice", "Dice", 3, False), ("iou", "IoU", 3, False), ("hd95_mm", "HD95 (mm)", 1, False), ("dif_vol_pct", "Δvol absoluta (%)", 1, True)):
            mb, sb, medb, _ = stat3(rb, key, absval); mn, sn, medn, _ = stat3(rn, key, absval)
            p = wilcoxon(rb, rn, key) if not absval else None
            summary[(name, key)] = (mb, mn)
            A(f"| {name} | {lab} | {fmt(mb, nd)} ± {fmt(sb, nd)} | {fmt(medb, nd)} | {fmt(mn, nd)} ± {fmt(sn, nd)} | {fmt(medn, nd)} | {fmt(p, 3) if p is not None else '—'} |")
    A("")
    A("\\* Prueba de rangos con signo de Wilcoxon, pareada por paciente. Con 6–8 pacientes tiene muy poca potencia: un p alto **no** "
      "demuestra que los modelos sean iguales, y un p bajo debe tomarse como indicativo, no como prueba.")
    A("")

    A("## Sobreajuste: brecha final entre pérdida de validación y de entrenamiento")
    A("Las pérdidas de nnU-Net son negativas (Dice + CE, donde el Dice resta). La brecha `val_loss − train_loss` es positiva cuando el "
      "modelo rinde mejor en los pacientes que ya vio que en los nuevos; una brecha mayor indica más sobreajuste.")
    A("")
    A("| Modelo | Última época | train_loss | val_loss | Brecha final | Brecha media últimas 10 épocas |")
    A("|---|---|---|---|---|---|")
    gaps = {}
    for name, e in (("Línea base (con aumento)", eb), ("Sin aumento", en)):
        g = final_gap(e); gaps[name] = g
        if g:
            A(f"| {name} | {g['epoch']} | {fmt(g['train'], 4)} | {fmt(g['val'], 4)} | {fmt(g['gap'], 4)} | {fmt(g['gap_mean_last'], 4)} |")
        else:
            A(f"| {name} | — | — | — | — | — |")
    A("")

    A("## Figuras")
    captions = {
        "C1_perdida_superpuesta.png": "Pérdida por época de ambos modelos. Azul = línea base, rojo = sin aumento; continua = entrenamiento, discontinua = validación.",
        "C2_brecha_sobreajuste.png": "Brecha val_loss − train_loss por época. Si la curva roja queda por encima de la azul, el modelo sin aumento sobreajusta más.",
        "C3_ema_pseudo_dice.png": "Pseudo-Dice (aproximado, por parches) en validación y su media móvil exponencial (EMA), como la calcula nnU-Net. Es una estimación por parches sin TTA; la métrica válida es el Dice por paciente.",
        "C4_dice_validacion.png": "Dice por paciente en validación (barras agrupadas).",
        "C5_dice_test.png": "Dice por paciente en test (barras agrupadas).",
        "C6_hd95_validacion.png": "HD95 por paciente en validación (más bajo = mejor).",
        "C7_hd95_test.png": "HD95 por paciente en test (más bajo = mejor).",
    }
    for name in sorted(figs):
        A(f"### {name[:-4].replace('_', ' ')}")
        A(f"![{name}](figuras_comparacion/{name})")
        A(captions.get(name, ""))
        A("")

    A("## Conclusiones (prudentes)")
    A(conclusions(summary, gaps, bool(vn), bool(tn)))
    A("")
    A("## Archivos")
    A(f"- Línea base: `{args.fold_b}`")
    A(f"- Sin aumento: `{args.fold_n}`")
    A(f"- CSV: `{args.val_b}`, `{args.val_n}`, `{args.test_b}`, `{args.test_n}`")
    A("- Predicciones de test: `pred_test_baseline_2d\\` y `pred_test_noDA_2d\\`")

    md_path = os.path.join(args.out_dir, "COMPARACION_baseline_vs_noDA.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out_dir, "COMPARACION_baseline_vs_noDA.html"), "w", encoding="utf-8") as f:
        f.write(md_to_html(md, args.out_dir))
    try:
        md_to_docx(md, args.out_dir, os.path.join(args.out_dir, "COMPARACION_baseline_vs_noDA.docx"))
    except Exception as ex:  # noqa
        print("docx fallido", repr(ex))
    print("OK", md_path)


def md_to_docx(md, base_dir, path):
    """Mismo contenido en Word (para subir a Drive como Google Doc con las imágenes incluidas)."""
    from docx import Document
    from docx.shared import Inches, Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10.5)

    def runs(par, text):
        text = text.replace("\\*", "*")
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
            part = part.replace("`", "")
            if part:
                r = par.add_run(part); r.bold = (i % 2 == 1)
    i = 0
    while i < len(md):
        ln = md[i]
        if ln.startswith("|"):
            rows = []
            while i < len(md) and md[i].startswith("|"):
                rows.append([c.strip() for c in md[i].strip().strip("|").split("|")]); i += 1
            rows = [rows[0]] + rows[2:]
            t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
            for ri, r in enumerate(rows):
                for ci, c in enumerate(r[:len(rows[0])]):
                    cell = t.cell(ri, ci); cell.text = ""
                    runs(cell.paragraphs[0], c)
                    for rr in cell.paragraphs[0].runs:
                        rr.font.size = Pt(8.5)
                        if ri == 0:
                            rr.bold = True
            doc.add_paragraph(); continue
        m = re.match(r"!\[.*?\]\((.+?)\)", ln)
        if m:
            p = os.path.join(base_dir, m.group(1))
            if os.path.exists(p):
                doc.add_picture(p, width=Inches(6.3))
        elif ln.startswith("### "):
            doc.add_heading(ln[4:], level=3)
        elif ln.startswith("## "):
            doc.add_heading(ln[3:], level=2)
        elif ln.startswith("# "):
            doc.add_heading(ln[2:], level=1)
        elif ln.startswith("- "):
            runs(doc.add_paragraph(style="List Bullet"), ln[2:])
        elif ln.startswith("> "):
            runs(doc.add_paragraph(), ln[2:])
        elif ln.strip():
            runs(doc.add_paragraph(), ln.strip("_") if ln.startswith("_") and ln.endswith("_") else ln)
        i += 1
    doc.save(path)


def conclusions(s, gaps, has_v, has_t):
    if not has_v and not has_t:
        return "No hay resultados del modelo sin aumento, así que no se puede concluir nada todavía."
    parts = []
    def cmp(name, key, label, higher_better=True, nd=3):  # noqa: E306
        mb, mn = s.get((name, key), (float("nan"), float("nan")))
        if mb != mb or mn != mn:
            return None
        d = mn - mb
        better = (d > 0) == higher_better
        return f"{label} medio en {name[0].lower() + name[1:]}:{fmt(mb, nd)} (base) vs {fmt(mn, nd)} (sin aumento), " \
               f"diferencia {('+' if d > 0 else '') + fmt(d, nd)} ({'a favor del modelo sin aumento' if better and d != 0 else 'a favor de la línea base' if d != 0 else 'empate'})."
    for name in ("Validación (7)", f"Validación sin {OUTLIER} (6)", "Test (8)"):
        for key, lab, hb, nd in (("dice", "Dice", True, 3), ("hd95_mm", "HD95", False, 1)):
            t = cmp(name, key, lab, hb, nd)
            if t:
                parts.append("- " + t)
    gb, gn = gaps.get("Línea base (con aumento)"), gaps.get("Sin aumento")
    if gb and gn:
        parts.append(f"- Brecha de sobreajuste (media de las últimas épocas): {fmt(gb['gap_mean_last'], 3)} (base) vs {fmt(gn['gap_mean_last'], 3)} (sin aumento)"
                     + (": quitar el aumento **aumentó** la brecha, lo esperable si el aumento actúa como regularizador." if gn["gap_mean_last"] > gb["gap_mean_last"]
                        else ": quitar el aumento no aumentó la brecha."))
    parts.append("")
    parts.append("**Lectura prudente:** con solo 7 pacientes de validación y 8 de test, diferencias de unas pocas centésimas de Dice "
                 "(o de 1–2 mm de HD95) están dentro de lo que puede variar por azar, por la elección de pacientes o por un solo caso difícil "
                 f"(como {OUTLIER} en validación). Además, la comparación mezcla dos cambios (sin aumento en el entrenamiento y sin espejo en la "
                 "inferencia; ver nota metodológica). Se trata de un único fold y una única semilla. Para afirmar que el aumento de datos mejora "
                 "o no el resultado convendría repetir en más folds (validación cruzada de 5 folds) y, si es posible, con más pacientes; "
                 "lo que sí permite este experimento es describir la tendencia y el efecto sobre el sobreajuste.")
    return "\n".join(parts)


def md_to_html(md, base_dir):
    """Convierte el markdown simple generado arriba en HTML autocontenido (imágenes en base64)."""
    def inline(t):
        t = H.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        t = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<i>\1</i>", t)
        t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<i>\1</i>", t)
        return t.replace("\\*", "*")
    out, i, lines = [], 0, md
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for r in rows[2:]:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            out.append("</tbody></table>"); continue
        m = re.match(r"!\[.*?\]\((.+?)\)", ln)
        if m:
            p = os.path.join(base_dir, m.group(1))
            if os.path.exists(p):
                b64 = base64.b64encode(open(p, "rb").read()).decode()
                out.append(f'<img src="data:image/png;base64,{b64}" alt="{H.escape(m.group(1))}">')
        elif ln.startswith("### "):
            out.append(f"<h3>{inline(ln[4:])}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:])}</h2>")
        elif ln.startswith("# "):
            out.append(f"<h1>{inline(ln[2:])}</h1>")
        elif ln.startswith("> "):
            out.append(f"<div class='warn'>{inline(ln[2:])}</div>")
        elif ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
            out.append("</ul>"); continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;line-height:1.5}"
           "table{border-collapse:collapse;margin:10px 0;font-size:14px;display:block;overflow-x:auto}th,td{border:1px solid #ccc;padding:4px 8px;text-align:center}"
           "th{background:#f0f3f8}img{max-width:100%;border:1px solid #eee;margin:6px 0}code{background:#f4f4f4;padding:1px 4px}"
           ".warn{background:#fff3cd;border:1px solid #e0c060;padding:8px 12px;border-radius:6px}h2{border-bottom:2px solid #2a6fdb;padding-bottom:4px;margin-top:32px}")
    return ("<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Comparación baseline vs noDA</title><style>{css}</style></head><body>" + "\n".join(out) + "</body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_b", default=os.path.join(RES, "nnUNetTrainer_250epochs__nnUNetPlans__2d", "fold_0"))
    ap.add_argument("--fold_n", default=os.path.join(RES, "nnUNetTrainerNoDA_250epochs__nnUNetPlans__2d", "fold_0"))
    ap.add_argument("--val_b", default=os.path.join(W, "resultados_val_baseline_2d.csv"))
    ap.add_argument("--val_n", default=os.path.join(W, "resultados_val_noDA_2d.csv"))
    ap.add_argument("--test_b", default=os.path.join(W, "resultados_test_baseline_2d.csv"))
    ap.add_argument("--test_n", default=os.path.join(W, "resultados_test_noDA_2d.csv"))
    ap.add_argument("--out_dir", default=os.path.join(W, "_nnunet_resumen"))
    build(ap.parse_args())
