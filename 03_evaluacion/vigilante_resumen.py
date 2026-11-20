"""Vigilante del entrenamiento nnU-Net 2d fold 0 + generador del informe amigable.

- Cada 10 min genera un informe de avance en _nnunet_resumen\\ y lo copia a OneDrive\\Tesis_nnUNet_resultados\\.
- Al terminar genera el informe FINAL: un gráfico por imagen (carpeta figuras\\), cada uno explicado en
  lenguaje simple, tabla por caso, glosario y limitaciones. Formatos: .docx (para subir a Google Docs con
  imágenes), .html autocontenido y .md.
- Si el entrenamiento terminó (checkpoint_final.pth) pero falló la validación, re-ejecuta SOLO la validación (--val)
  con 2 procesos. Si el entrenamiento se cayó con error antes de terminar, lo reanuda UNA vez con --c.
No borra ni modifica nada existente fuera de _nnunet_resumen, _nnunet_logs y la carpeta de OneDrive.

Uso: python vigilante_resumen.py [--once]
"""
import base64
import datetime as dt
import glob
import html as htmlmod
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import time

W = r"C:\Users\joaki\Documents\Tesis_segmentacion"
TRAINER = "nnUNetTrainer_250epochs"
TOTAL_EPOCHS = 250
FOLD = os.path.join(W, "nnUNet_results", "Dataset001_ProstataCTV", f"{TRAINER}__nnUNetPlans__2d", "fold_0")
RAW = os.path.join(W, "nnUNet_raw", "Dataset001_ProstataCTV")
WRAPLOG = os.path.join(W, "_nnunet_logs", "04_train_2d_fold0.log")
MYLOG = os.path.join(W, "_nnunet_logs", "05_vigilante.log")
OUT = os.path.join(W, "_nnunet_resumen")
CLOUD = os.path.join(r"C:\Users\joaki\OneDrive", "Tesis_nnUNet_resultados")
PY = r"C:\Users\joaki\anaconda3\envs\nnunet\python.exe"
WRAPPER = os.path.join(W, "_nnunet_scripts", "run_train_keepawake.py")
DEADLINE = dt.datetime(2026, 11, 16, 16, 0)
INTERVAL_S = 600

C_TRAIN, C_VAL, C_EMA, C_GT, C_PRED = "#2a6fdb", "#e8710a", "#1a9e5a", "#1a9e5a", "#d93025"
state = {"resumes": 0, "val_reruns": 0}


# ----------------------------------------------------------------------------- utilidades
def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(MYLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def nnunet_env(extra=None):
    env = dict(os.environ)
    env.update({"nnUNet_raw": os.path.join(W, "nnUNet_raw"), "nnUNet_preprocessed": os.path.join(W, "nnUNet_preprocessed"),
                "nnUNet_results": os.path.join(W, "nnUNet_results"), "nnUNet_n_proc_DA": "3", "nnUNet_compile": "f",
                "PYTHONUNBUFFERED": "1"})
    env.update(extra or {})
    return env


def fmt(x, nd=2):
    """Número con coma decimal (estilo chileno)."""
    if x is None or x != x:
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def parse_training_logs():
    epochs, cur = {}, None
    for path in sorted(glob.glob(os.path.join(FOLD, "training_log_*.txt"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r": Epoch (\d+)\s*$", line)
                if m:
                    cur = int(m.group(1)); epochs.setdefault(cur, {}); continue
                if cur is None:
                    continue
                for key, pat in (("train_loss", r"train_loss (-?[\d.]+)"), ("val_loss", r"val_loss (-?[\d.]+)"),
                                 ("time", r"Epoch time: ([\d.]+) s"), ("ema_best", r"New best EMA pseudo Dice: ([\d.]+)"),
                                 ("lr", r"Current learning rate: ([\d.eE-]+)")):
                    m = re.search(pat, line)
                    if m:
                        epochs[cur][key] = float(m.group(1))
                m = re.search(r"Pseudo dice \[(.*)\]", line)
                if m:
                    vals = re.findall(r"(-?\d+\.\d+|-?\d+)\)", m.group(1)) or re.findall(r"-?\d+\.\d+", m.group(1))
                    if vals:
                        epochs[cur]["pseudo_dice"] = float(vals[0])
    return {k: v for k, v in epochs.items() if "time" in v}


def ema_series(vals):
    out, e = [], None
    for v in vals:
        if v is not None:
            e = v if e is None else 0.9 * e + 0.1 * v
        out.append(e)
    return out


def wrapper_status():
    if not os.path.exists(WRAPLOG):
        return 0, []
    txt = open(WRAPLOG, encoding="utf-8", errors="replace").read()
    return txt.count("] INICIO:"), [int(x) for x in re.findall(r"FIN: codigo de salida (-?\d+)", txt)]


def last_log_age_min():
    files = glob.glob(os.path.join(FOLD, "training_log_*.txt"))
    return (time.time() - max(os.path.getmtime(f) for f in files)) / 60 if files else 1e9


def disk_free_gb():
    return shutil.disk_usage("C:\\").free / 1024 ** 3


# ----------------------------------------------------------------------------- figuras
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
    return plt


def fig_loss(epochs, path):
    plt = _plt(); ks = sorted(epochs)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, [epochs[k].get("train_loss") for k in ks], color=C_TRAIN, lw=2, label="Entrenamiento (25 pacientes)")
    ax.plot(ks, [epochs[k].get("val_loss") for k in ks], color=C_VAL, lw=2, label="Validación (7 pacientes no vistos)")
    ax.set_xlabel("Época (1 época = 250 lotes de 12 cortes)"); ax.set_ylabel("Pérdida (más abajo = mejor)")
    ax.set_title("1. Pérdida durante el entrenamiento"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_pseudo(epochs, path):
    plt = _plt(); ks = sorted(epochs)
    pd = [epochs[k].get("pseudo_dice") for k in ks]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, pd, "o", ms=3, color=C_VAL, alpha=.45, label="Pseudo-Dice de cada época")
    ax.plot(ks, ema_series(pd), color=C_EMA, lw=2.5, label="Tendencia suavizada (EMA)")
    ax.set_ylim(0, 1); ax.set_xlabel("Época"); ax.set_ylabel("Pseudo-Dice (0 = nada, 1 = perfecto)")
    ax.set_title("2. Calidad estimada en validación durante el entrenamiento"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_lr(epochs, path):
    plt = _plt(); ks = [k for k in sorted(epochs) if "lr" in epochs[k]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ks, [epochs[k]["lr"] for k in ks], color=C_TRAIN, lw=2)
    ax.set_xlabel("Época"); ax.set_ylabel("Tasa de aprendizaje")
    ax.set_title("3. Tasa de aprendizaje (tamaño de los ajustes)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_time(epochs, path):
    plt = _plt(); ks = sorted(epochs)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ks, [epochs[k]["time"] for k in ks], color="#666", lw=1.5)
    ax.set_xlabel("Época"); ax.set_ylabel("Segundos por época"); ax.set_ylim(bottom=0)
    ax.set_title("4. Tiempo de cómputo por época (RTX 4060)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _band_color(d):
    return "#1a9e5a" if d >= 0.85 else ("#e8a10a" if d >= 0.75 else "#d93025")


def fig_dice_cases(rows, mean_dice, path):
    plt = _plt(); rs = sorted(rows, key=lambda r: r["dice"])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rs) + 2))
    ax.barh([r["case"] for r in rs], [r["dice"] for r in rs], color=[_band_color(r["dice"]) for r in rs])
    for i, r in enumerate(rs):
        ax.text(r["dice"] + 0.005, i, fmt(r["dice"], 3), va="center", fontsize=11)
    ax.axvline(mean_dice, color="k", ls="--", lw=1.2, label=f"Promedio = {fmt(mean_dice, 3)}")
    ax.set_xlim(max(0, min(r["dice"] for r in rs) - 0.15), 1.0)
    ax.set_xlabel("Dice (1 = coincidencia perfecta con el contorno del médico)")
    ax.set_title("5. Dice por paciente de validación"); ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_volumes(rows, path):
    import numpy as np
    plt = _plt(); x = np.arange(len(rows)); w = .38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [r["vol_gt"] for r in rows], w, color=C_GT, label="Médico (referencia)")
    ax.bar(x + w / 2, [r["vol_pred"] for r in rows], w, color=C_PRED, label="nnU-Net")
    for i, r in enumerate(rows):
        if r["vol_diff"] == r["vol_diff"]:
            ax.text(i + w / 2, r["vol_pred"], f"{r['vol_diff']:+.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([r["case"] for r in rows], rotation=0)
    ax.set_ylabel("Volumen del CTV (cc = cm³)"); ax.set_title("6. Volumen del CTV: médico vs modelo"); ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_hd95(rows, path):
    plt = _plt(); rs = sorted([r for r in rows if r["hd95"] == r["hd95"]], key=lambda r: -r["hd95"])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rs) + 2))
    ax.barh([r["case"] for r in rs], [r["hd95"] for r in rs], color="#5b6bbf")
    for i, r in enumerate(rs):
        ax.text(r["hd95"] + 0.1, i, f"{fmt(r['hd95'], 1)} mm", va="center", fontsize=11)
    ax.set_xlabel("HD95 en milímetros (más bajo = mejor)"); ax.set_title("7. Distancia máxima típica entre bordes (HD95)")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def load_case(c):
    import numpy as np
    import SimpleITK as sitk
    ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(RAW, "imagesTr", f"{c}_0000.nii.gz"))).astype(np.float32)
    gt_img = sitk.ReadImage(os.path.join(RAW, "labelsTr", f"{c}.nii.gz"))
    gt = sitk.GetArrayFromImage(gt_img) > 0
    pr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(FOLD, "validation", f"{c}.nii.gz"))) > 0
    return ct, gt, pr, gt_img.GetSpacing()[::-1]  # spacing en orden z, y, x


def hd95(gt, pr, sp):
    import numpy as np
    from scipy import ndimage as ndi
    if not gt.any() or not pr.any():
        return float("nan")
    zz, yy, xx = np.where(gt | pr)
    m = 10
    sl = tuple(slice(max(a.min() - m, 0), a.max() + m + 1) for a in (zz, yy, xx))
    g, p = gt[sl], pr[sl]
    sg = g & ~ndi.binary_erosion(g); spp = p & ~ndi.binary_erosion(p)
    dg = ndi.distance_transform_edt(~sg, sampling=sp); dp = ndi.distance_transform_edt(~spp, sampling=sp)
    return float(max(np.percentile(dp[sg], 95), np.percentile(dg[spp], 95)))


def fig_case(c, ct, gt, pr, dice, path):
    import numpy as np
    from matplotlib.lines import Line2D
    plt = _plt()
    zs = np.where(gt.any(axis=(1, 2)))[0]
    ys, xs = np.where(gt.any(axis=0) | pr.any(axis=0))
    y0, y1 = max(ys.min() - 35, 0), min(ys.max() + 35, gt.shape[1])
    x0, x1 = max(xs.min() - 35, 0), min(xs.max() + 35, gt.shape[2])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for a, (name, q) in zip(axes, (("zona inferior", .15), ("zona media", .5), ("zona superior", .85))):
        z = int(zs[0] + q * (zs[-1] - zs[0]))
        a.imshow(np.clip(ct[z, y0:y1, x0:x1], -160, 240), cmap="gray")
        if gt[z, y0:y1, x0:x1].any():
            a.contour(gt[z, y0:y1, x0:x1], [0.5], colors=C_GT, linewidths=2)
        if pr[z, y0:y1, x0:x1].any():
            a.contour(pr[z, y0:y1, x0:x1], [0.5], colors=C_PRED, linewidths=1.6, linestyles="--")
        a.set_title(f"{name} del CTV (corte {z})", fontsize=12, fontweight="normal"); a.axis("off")
    fig.legend(handles=[Line2D([], [], color=C_GT, lw=2, label="Contorno del médico"),
                        Line2D([], [], color=C_PRED, lw=2, ls="--", label="Contorno de nnU-Net")],
               loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.suptitle(f"Paciente {c} — Dice {fmt(dice, 3)}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(path, dpi=100); plt.close(fig)


# ----------------------------------------------------------------------------- contenido del informe
GLOSARIO = [
    ("nnU-Net", "Programa de inteligencia artificial que se configura solo para segmentar imágenes médicas. Es la línea base de la tesis."),
    ("Segmentar", "Dibujar automáticamente el contorno de una estructura (aquí, el CTV prostático) corte a corte en la TC."),
    ("CTV", "Volumen blanco clínico: la zona que el médico contornea para irradiar (próstata ± vesículas según el caso)."),
    ("GT (ground truth) / referencia", "El contorno hecho por el médico. Es lo que consideramos «la respuesta correcta»."),
    ("2d", "El modelo mira un corte axial a la vez (no el volumen 3D completo)."),
    ("Fold 0", "Una de 5 particiones posibles de los 32 pacientes: 25 para aprender y 7 para evaluar. Solo entrenamos esta."),
    ("Entrenamiento / validación", "Entrenamiento: pacientes con los que el modelo aprende. Validación: pacientes que el modelo NUNCA vio, usados para medir qué tan bien funciona."),
    ("Test", "8 pacientes adicionales guardados aparte; todavía no se usaron (se usan al final de la tesis)."),
    ("Época", "Una ronda de aprendizaje: 250 lotes de 12 cortes cada uno. Entrenamos 250 épocas."),
    ("Pérdida (loss)", "Número que mide cuánto se equivoca el modelo; el modelo aprende tratando de bajarla. En nnU-Net es negativa porque le resta el Dice: más abajo = mejor."),
    ("Dice", "Superposición entre el contorno del modelo y el del médico. 0 = no se tocan, 1 = idénticos. 0,85 significa ~85 % de coincidencia."),
    ("Pseudo-Dice", "Dice aproximado que nnU-Net calcula durante el entrenamiento con trozos de imagen. Sirve para ver la tendencia; el número oficial es el Dice final por paciente."),
    ("EMA", "Promedio móvil exponencial: una línea suavizada que muestra la tendencia sin el ruido de cada época."),
    ("IoU (Jaccard)", "Otra medida de superposición, más estricta que el Dice (siempre da un número menor). 1 = perfecto."),
    ("HD95", "Distancia de Hausdorff al 95 %: qué tan lejos (en mm) puede quedar el borde del modelo respecto del borde del médico, ignorando el 5 % de puntos más extremos. Más bajo = mejor."),
    ("Volumen (cc)", "Tamaño del CTV en centímetros cúbicos. Si el modelo da más volumen que el médico, sobresegmenta; si da menos, subsegmenta."),
    ("Tasa de aprendizaje (learning rate)", "Tamaño de los ajustes que hace el modelo en cada paso. Empieza en 0,01 y baja hasta casi 0 para afinar al final."),
    ("Checkpoint", "Copia guardada del modelo entrenado (archivo .pth)."),
    ("HU", "Unidades Hounsfield: escala de densidad de la TC (agua = 0, aire = −1000, hueso > 400)."),
]


def explain_loss(epochs):
    ks = sorted(epochs); last = ks[-5:]
    tl = st.mean(epochs[k]["train_loss"] for k in last); vl = st.mean(epochs[k]["val_loss"] for k in last)
    gap = vl - tl
    if gap > 0.1:
        obs = (f"Al final, la pérdida de entrenamiento ({fmt(tl)}) quedó claramente más baja que la de validación ({fmt(vl)}). "
               "Esto sugiere algo de sobreajuste: el modelo aprende mejor a los pacientes que ya vio que a los nuevos.")
    else:
        obs = (f"Al final, ambas curvas quedaron cerca (entrenamiento {fmt(tl)}, validación {fmt(vl)}). "
               "Esto indica que el modelo funciona casi igual de bien en pacientes nuevos que en los que usó para aprender: no hay señales fuertes de sobreajuste.")
    return ("Muestra cuánto se equivoca el modelo a lo largo del entrenamiento. La línea azul es con los pacientes de entrenamiento "
            "y la naranja con los de validación (no vistos).",
            "Ambas deben bajar y luego aplanarse. Si la azul sigue bajando pero la naranja sube o se separa mucho, el modelo está "
            "«memorizando» (sobreajuste). Los valores son negativos porque nnU-Net combina dos medidas y una se resta; lo importante es la forma.",
            obs)


def explain_pseudo(epochs):
    ks = sorted(epochs); pd = [epochs[k].get("pseudo_dice") for k in ks]; ema = ema_series(pd)
    n = len(ks)
    reach = next((ks[i] for i, e in enumerate(ema) if e is not None and e >= 0.95 * ema[-1]), ks[-1])
    obs = (f"La tendencia (línea verde) llegó al 95 % de su valor final alrededor de la época {reach} y terminó en {fmt(ema[-1], 3)}. "
           f"Después de eso las mejoras fueron pequeñas: {'buena parte de las épocas restantes sirvieron para afinar' if reach < 0.5 * n else 'el modelo siguió mejorando hasta tarde, más épocas podrían ayudar'}.")
    return ("Cada punto es una estimación rápida de la calidad (pseudo-Dice) sobre los 7 pacientes de validación al final de cada época. "
            "La línea verde suaviza esos puntos para ver la tendencia.",
            "Mientras más cerca de 1, mejor. La línea verde parte en 0 por cómo se calcula el suavizado (no porque el modelo empiece en 0). "
            "Es una estimación con trozos de imagen; el resultado oficial es el Dice por paciente del gráfico 5.",
            obs)


def explain_lr():
    return ("Cuánto ajusta el modelo sus parámetros en cada paso.",
            "Empieza alto (0,01) para aprender rápido y baja gradualmente hasta casi 0 para hacer ajustes finos al final. "
            "Es el comportamiento estándar de nnU-Net (decaimiento «poly»); no hay que interpretarlo, solo confirma que el entrenamiento siguió el plan.",
            "Bajó de forma suave y continua, como se esperaba.")


def explain_time(epochs):
    ts = [epochs[k]["time"] for k in sorted(epochs)]
    return ("Cuántos segundos tardó cada época en la tarjeta gráfica RTX 4060.",
            "Sirve para estimar el costo computacional y planificar los próximos experimentos (3d, más folds).",
            f"Cada época tardó en promedio {fmt(st.mean(ts), 0)} s (≈{fmt(st.mean(ts) / 60, 1)} min). "
            f"El entrenamiento completo sumó {fmt(sum(ts) / 3600, 1)} horas.")


def explain_dice(rows, mean):
    best = max(rows, key=lambda r: r["dice"]); worst = min(rows, key=lambda r: r["dice"])
    n_g = sum(r["dice"] >= .85 for r in rows); n_a = sum(.75 <= r["dice"] < .85 for r in rows); n_r = sum(r["dice"] < .75 for r in rows)
    return ("El resultado principal: qué tanto se superpone el contorno del modelo con el del médico en cada uno de los 7 pacientes de validación, "
            "usando el volumen completo.",
            "Barras más largas = mejor. Colores orientativos: verde ≥ 0,85 (muy buena coincidencia), amarillo 0,75–0,85 (buena, con diferencias visibles), "
            "rojo < 0,75 (revisar). Como referencia general, en la literatura de próstata en TC suelen reportarse Dice entre ~0,80 y ~0,90, "
            "y entre dos médicos distintos tampoco se obtiene 1,0.",
            f"Dice promedio {fmt(mean, 3)}. Mejor paciente: {best['case']} ({fmt(best['dice'], 3)}); más difícil: {worst['case']} ({fmt(worst['dice'], 3)}). "
            f"{n_g} en verde, {n_a} en amarillo y {n_r} en rojo.")


def explain_vol(rows):
    ok = [r["vol_diff"] for r in rows if r["vol_diff"] == r["vol_diff"]]
    md = st.mean(ok); mad = st.mean(abs(x) for x in ok)
    tend = "a dibujar un CTV más grande que el médico (sobresegmentación)" if md > 3 else (
        "a dibujar un CTV más chico que el médico (subsegmentación)" if md < -3 else "a no tener un sesgo claro de tamaño")
    return ("Compara el tamaño del CTV dibujado por el médico (verde) con el del modelo (rojo) en cada paciente. El número sobre la barra roja es la diferencia en %.",
            "Barras parecidas = el modelo acierta el tamaño. Ojo: dos volúmenes iguales no garantizan la misma forma (eso lo mide el Dice).",
            f"En promedio el modelo tiende {tend}: diferencia media {'+' if md >= 0 else ''}{fmt(md, 1)} % (diferencia absoluta media {fmt(mad, 1)} %).")


def explain_hd(rows):
    v = [r["hd95"] for r in rows if r["hd95"] == r["hd95"]]
    worst = max(rows, key=lambda r: r["hd95"] if r["hd95"] == r["hd95"] else -1)
    return ("Para cada paciente, qué tan lejos puede quedar el borde del modelo respecto del borde del médico (en milímetros), ignorando el 5 % de puntos más extremos.",
            "Más bajo = mejor. Complementa al Dice: el Dice mide superposición general y el HD95 detecta «escapes» locales (por ejemplo, el modelo se extiende "
            "hacia la vejiga o el recto en algunos cortes). Para dar escala: un corte de esta TC mide 3 mm de grosor.",
            f"HD95 promedio {fmt(st.mean(v), 1)} mm (mediana {fmt(st.median(v), 1)} mm). El mayor fue {worst['case']} con {fmt(worst['hd95'], 1)} mm.")


def explain_cases():
    return ("Para cada paciente de validación se muestran 3 cortes de la TC (zona inferior, media y superior del CTV) con el contorno del médico (verde continuo) "
            "y el de nnU-Net (rojo discontinuo). La imagen está ampliada alrededor de la próstata, con ventana de tejido blando.",
            "Donde las dos líneas coinciden, el modelo acertó. Las diferencias suelen aparecer en los extremos (ápex y base/vesículas), que también son las zonas "
            "donde los médicos más discrepan entre sí.")


# ----------------------------------------------------------------------------- ensamblado
def collect_final(figdir):
    summary_path = os.path.join(FOLD, "validation", "summary.json")
    s = json.load(open(summary_path, encoding="utf-8"))
    rows, case_figs = [], []
    for m in s["metric_per_case"]:
        c = os.path.basename(m["reference_file"]).replace(".nii.gz", "")
        mm = m["metrics"].get("1", next(iter(m["metrics"].values())))
        row = {"case": c, "dice": mm["Dice"], "iou": mm["IoU"], "hd95": float("nan"), "vol_gt": float("nan"), "vol_pred": float("nan"), "vol_diff": float("nan")}
        try:
            ct, gt, pr, sp = load_case(c)
            vv = sp[0] * sp[1] * sp[2] / 1000.0
            row.update(vol_gt=gt.sum() * vv, vol_pred=pr.sum() * vv)
            row["vol_diff"] = 100 * (row["vol_pred"] - row["vol_gt"]) / row["vol_gt"]
            row["hd95"] = hd95(gt, pr, sp)
            p = os.path.join(figdir, f"08_contornos_{c}.png"); fig_case(c, ct, gt, pr, row["dice"], p); case_figs.append(p)
            del ct, gt, pr
        except Exception as e:  # noqa
            log(f"Error procesando {c}: {e!r}")
        rows.append(row)
    return s, rows, case_figs


def build_report(epochs, final, note=""):
    figdir = os.path.join(OUT, "figuras"); os.makedirs(figdir, exist_ok=True)
    now = dt.datetime.now(); ks = sorted(epochs); last = ks[-1] if ks else None
    times = [epochs[k]["time"] for k in ks[-10:]]
    mean_t = st.mean(times) if times else 0
    eta = now + dt.timedelta(seconds=(TOTAL_EPOCHS - 1 - last) * mean_t) if (last is not None and mean_t) else None
    best_ema = max((v.get("ema_best", 0) for v in epochs.values()), default=0)
    total_h = sum(v["time"] for v in epochs.values()) / 3600 if epochs else 0

    sections = []  # (titulo, [(tipo, contenido)])
    figs = []
    if epochs:
        for fn, f, ex in (("01_perdida.png", fig_loss, explain_loss), ("02_pseudo_dice.png", fig_pseudo, explain_pseudo),
                          ("03_tasa_aprendizaje.png", fig_lr, lambda e: explain_lr()), ("04_tiempo_por_epoca.png", fig_time, explain_time)):
            p = os.path.join(figdir, fn)
            try:
                f(epochs, p); figs.append((p, ex(epochs)))
            except Exception as e:  # noqa
                log(f"Error figura {fn}: {e!r}")

    rows, s, case_figs, mean_dice = [], None, [], None
    summary_ok = os.path.exists(os.path.join(FOLD, "validation", "summary.json"))
    if final and summary_ok:
        s, rows, case_figs = collect_final(figdir)
        mean_dice = s.get("foreground_mean", {}).get("Dice", st.mean(r["dice"] for r in rows))
        for fn, f, ex in (("05_dice_por_paciente.png", lambda p: fig_dice_cases(rows, mean_dice, p), lambda: explain_dice(rows, mean_dice)),
                          ("06_volumenes.png", lambda p: fig_volumes(rows, p), lambda: explain_vol(rows)),
                          ("07_hd95.png", lambda p: fig_hd95(rows, p), lambda: explain_hd(rows))):
            p = os.path.join(figdir, fn)
            try:
                f(p); figs.append((p, ex()))
            except Exception as e:  # noqa
                log(f"Error figura {fn}: {e!r}")

    # ---- texto
    title = "Resultados nnU-Net — Segmentación del CTV prostático en TC"
    sub = f"Línea base 2d, fold 0, 250 épocas · {'INFORME FINAL' if final else 'AVANCE (entrenamiento en curso)'} · generado {now:%d/%m/%Y %H:%M}"
    key = []
    if note:
        key.append(f"Nota: {note}")
    if final and rows:
        dices = [r["dice"] for r in rows]; hds = [r["hd95"] for r in rows if r["hd95"] == r["hd95"]]
        key += [f"Dice promedio en los 7 pacientes de validación: {fmt(mean_dice, 3)} (rango {fmt(min(dices), 3)} a {fmt(max(dices), 3)}). "
                f"En palabras simples: el contorno del modelo coincide en ~{round(100 * mean_dice)} % con el del médico.",
                f"HD95 promedio: {fmt(st.mean(hds), 1)} mm (qué tan lejos pueden quedar los bordes)." if hds else "",
                f"Entrenamiento: {len(ks)} épocas en {fmt(total_h, 1)} h en una RTX 4060 de 8 GB, desde cero (sin modelos preentrenados).",
                "Es una línea base: un primer resultado de referencia para comparar mejoras futuras."]
    elif last is not None:
        key += [f"Épocas completadas: {last + 1} de {TOTAL_EPOCHS} ({fmt(total_h, 1)} h, ≈{fmt(mean_t, 0)} s por época).",
                f"Último pseudo-Dice: {fmt(epochs[last].get('pseudo_dice'), 3)} · mejor tendencia (EMA): {fmt(best_ema, 3)}.",
                f"Fin estimado del entrenamiento: {eta:%H:%M} (+ ~10 min de validación final)." if eta else ""]
        if final and not summary_ok:
            key.append("La validación final no terminó: solo están las curvas de entrenamiento.")
    key = [k for k in key if k]

    que_hicimos = ("Entrenamos nnU-Net para que dibuje automáticamente el CTV prostático en tomografías de planificación. "
                   "De los 32 pacientes con contorno, el modelo aprendió con 25 y lo evaluamos con 7 que nunca vio (validación). "
                   "Otros 8 pacientes (test) quedaron guardados sin usar para la evaluación final de la tesis. "
                   "Usamos la configuración 2d (el modelo mira un corte a la vez), entrenada desde cero, que es la línea base estándar.")
    limit = ["Solo 7 pacientes de validación: los promedios pueden cambiar bastante con otros pacientes.",
             "Solo se entrenó 1 de los 5 folds (particiones). Lo usual en nnU-Net es entrenar los 5 y promediar.",
             "250 épocas en vez de las 1000 por defecto de nnU-Net (por tiempo de cómputo).",
             "El conjunto de test (8 pacientes) todavía no se evaluó.",
             "3 pacientes (PROS_030–032) tienen cortes de 2,5 mm y el resto de 3 mm; no afecta al modelo 2d pero hay que mencionarlo."]
    pasos = ["Evaluar en los 8 pacientes de test.", "Entrenar la configuración 3d_fullres y comparar con esta 2d.",
             "Entrenar los 5 folds para tener una estimación más robusta.", "Revisar visualmente los pacientes con menor Dice para entender los errores."]

    files = [p for p, _ in figs] + case_figs
    write_md(title, sub, key, que_hicimos, figs, rows, case_figs, limit, pasos, final)
    write_html(title, sub, key, que_hicimos, figs, rows, case_figs, limit, pasos)
    try:
        write_docx(title, sub, key, que_hicimos, figs, rows, case_figs, limit, pasos)
        files.append(os.path.join(OUT, "RESUMEN_resultados.docx"))
    except Exception as e:  # noqa
        log(f"Error generando docx: {e!r}")
    files += [os.path.join(OUT, n) for n in ("RESUMEN_resultados.md", "RESUMEN_resultados.html")]
    if s is not None:
        shutil.copy2(os.path.join(FOLD, "validation", "summary.json"), os.path.join(OUT, "summary_validacion.json"))
        files.append(os.path.join(OUT, "summary_validacion.json"))
    status = {"generado": now.isoformat(timespec="seconds"), "final": bool(final), "validacion_ok": bool(final and summary_ok),
              "epocas_completadas": (last + 1) if last is not None else 0, "mejor_ema_pseudo_dice": best_ema,
              "dice_medio_validacion": mean_dice, "eta": eta.isoformat(timespec="minutes") if (eta and not final) else None, "nota": note,
              "figuras": [os.path.basename(f) for f in [p for p, _ in figs] + case_figs]}
    json.dump(status, open(os.path.join(OUT, "estado.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    files.append(os.path.join(OUT, "estado.json"))
    copy_to_cloud(files)
    log(f"Informe {'FINAL' if final else 'de avance'} generado (época {last}, {len(files)} archivos).")


def case_table(rows):
    head = ["Paciente", "Dice", "IoU", "HD95 (mm)", "Vol. médico (cc)", "Vol. modelo (cc)", "Diferencia vol."]
    body = [[r["case"], fmt(r["dice"], 3), fmt(r["iou"], 3), fmt(r["hd95"], 1), fmt(r["vol_gt"], 1), fmt(r["vol_pred"], 1),
             (f"{r['vol_diff']:+.1f} %".replace(".", ",") if r["vol_diff"] == r["vol_diff"] else "—")] for r in rows]
    if rows:
        d = [r["dice"] for r in rows]; h = [r["hd95"] for r in rows if r["hd95"] == r["hd95"]]
        body.append(["Promedio", fmt(st.mean(d), 3), fmt(st.mean(r["iou"] for r in rows), 3), fmt(st.mean(h), 1) if h else "—",
                     fmt(st.mean(r["vol_gt"] for r in rows), 1), fmt(st.mean(r["vol_pred"] for r in rows), 1), ""])
    return head, body


def write_md(title, sub, key, qh, figs, rows, case_figs, limit, pasos, final):
    L = [f"# {title}", f"_{sub}_", "", "## Resumen en 1 minuto"] + [f"- {k}" for k in key] + ["", "## ¿Qué hicimos?", qh, ""]
    for p, (que, como, obs) in figs:
        n = os.path.basename(p)
        L += [f"## {int(n[:2])}. {FIG_TITLES.get(n, n)}", f"![{n}](figuras/{n})", f"**Qué muestra:** {que}", "", f"**Cómo leerlo:** {como}", "", f"**Qué vemos aquí:** {obs}", ""]
    if rows:
        h, b = case_table(rows)
        L += ["## Tabla por paciente", "| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(r) + " |" for r in b] + [""]
    if case_figs:
        que, como = explain_cases()
        L += ["## 8. Ejemplos visuales por paciente", f"**Qué muestra:** {que}", "", f"**Cómo leerlo:** {como}", ""]
        L += [f"![{os.path.basename(p)}](figuras/{os.path.basename(p)})" for p in case_figs] + [""]
    L += ["## Glosario (qué significa cada término)"] + [f"- **{t}:** {d}" for t, d in GLOSARIO]
    L += ["", "## Limitaciones a tener en cuenta"] + [f"- {x}" for x in limit] + ["", "## Próximos pasos sugeridos"] + [f"- {x}" for x in pasos]
    open(os.path.join(OUT, "RESUMEN_resultados.md"), "w", encoding="utf-8").write("\n".join(L))


FIG_TITLES = {"01_perdida.png": "Pérdida durante el entrenamiento", "02_pseudo_dice.png": "Calidad estimada durante el entrenamiento (pseudo-Dice)",
              "03_tasa_aprendizaje.png": "Tasa de aprendizaje", "04_tiempo_por_epoca.png": "Tiempo por época",
              "05_dice_por_paciente.png": "Dice por paciente (resultado principal)", "06_volumenes.png": "Volumen: médico vs modelo",
              "07_hd95.png": "Distancia entre bordes (HD95)"}


def write_html(title, sub, key, qh, figs, rows, case_figs, limit, pasos):
    e = htmlmod.escape
    def img(p):
        return f"<img alt='{e(os.path.basename(p))}' src='data:image/png;base64,{base64.b64encode(open(p, 'rb').read()).decode()}'>"
    H = [f"<h1>{e(title)}</h1><p class='sub'>{e(sub)}</p>", "<div class='key'><h2>Resumen en 1 minuto</h2><ul>"
         + "".join(f"<li>{e(k)}</li>" for k in key) + "</ul></div>", f"<h2>¿Qué hicimos?</h2><p>{e(qh)}</p>"]
    for p, (que, como, obs) in figs:
        n = os.path.basename(p)
        H.append(f"<section><h2>{int(n[:2])}. {e(FIG_TITLES.get(n, n))}</h2>{img(p)}<p><b>Qué muestra:</b> {e(que)}</p>"
                 f"<p><b>Cómo leerlo:</b> {e(como)}</p><p class='obs'><b>Qué vemos aquí:</b> {e(obs)}</p></section>")
    if rows:
        h, b = case_table(rows)
        H.append("<h2>Tabla por paciente</h2><div class='tw'><table><tr>" + "".join(f"<th>{e(x)}</th>" for x in h) + "</tr>"
                 + "".join("<tr>" + "".join(f"<td>{e(x)}</td>" for x in r) + "</tr>" for r in b) + "</table></div>")
    if case_figs:
        que, como = explain_cases()
        H.append(f"<section><h2>8. Ejemplos visuales por paciente</h2><p><b>Qué muestra:</b> {e(que)}</p><p><b>Cómo leerlo:</b> {e(como)}</p>"
                 + "".join(img(p) for p in case_figs) + "</section>")
    H.append("<h2>Glosario</h2><dl>" + "".join(f"<dt>{e(t)}</dt><dd>{e(d)}</dd>" for t, d in GLOSARIO) + "</dl>")
    H.append("<h2>Limitaciones</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in limit) + "</ul>")
    H.append("<h2>Próximos pasos sugeridos</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in pasos) + "</ul>")
    css = ("body{font-family:system-ui,Segoe UI,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.55;color:#222;background:#fff}"
           "h1{margin-bottom:0}.sub{color:#666;margin-top:4px}.key{background:#eef5ff;border-left:5px solid #2a6fdb;padding:8px 16px;border-radius:6px}"
           "section{border-top:1px solid #ddd;margin-top:28px}img{max-width:100%;display:block;margin:12px 0}.obs{background:#f1faf4;padding:8px 12px;border-radius:6px}"
           ".tw{overflow-x:auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px 9px;text-align:right}th{background:#f4f4f4}"
           "td:first-child,th:first-child{text-align:left}dt{font-weight:bold;margin-top:8px}dd{margin-left:16px}")
    doc = (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Resultados nnU-Net</title><style>{css}</style></head><body>{''.join(H)}</body></html>")
    open(os.path.join(OUT, "RESUMEN_resultados.html"), "w", encoding="utf-8").write(doc)


def write_docx(title, sub, key, qh, figs, rows, case_figs, limit, pasos):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    d = Document()
    d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(11)
    d.add_heading(title, 0)
    p = d.add_paragraph(sub); p.runs[0].italic = True; p.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    d.add_heading("Resumen en 1 minuto", 1)
    for k in key:
        d.add_paragraph(k, style="List Bullet")
    d.add_heading("¿Qué hicimos?", 1); d.add_paragraph(qh)

    def labeled(lbl, txt):
        q = d.add_paragraph(); r = q.add_run(lbl + " "); r.bold = True; q.add_run(txt)

    for pth, (que, como, obs) in figs:
        n = os.path.basename(pth)
        d.add_heading(f"{int(n[:2])}. {FIG_TITLES.get(n, n)}", 1)
        d.add_picture(pth, width=Inches(6.3))
        labeled("Qué muestra:", que); labeled("Cómo leerlo:", como); labeled("Qué vemos aquí:", obs)
    if rows:
        d.add_heading("Tabla por paciente", 1)
        h, b = case_table(rows)
        t = d.add_table(rows=1, cols=len(h)); t.style = "Light Grid Accent 1"
        for i, x in enumerate(h):
            t.rows[0].cells[i].text = x
        for r in b:
            cells = t.add_row().cells
            for i, x in enumerate(r):
                cells[i].text = x
    if case_figs:
        que, como = explain_cases()
        d.add_heading("8. Ejemplos visuales por paciente", 1)
        labeled("Qué muestra:", que); labeled("Cómo leerlo:", como)
        for pth in case_figs:
            d.add_picture(pth, width=Inches(6.5))
    d.add_heading("Glosario (qué significa cada término)", 1)
    for t_, de in GLOSARIO:
        q = d.add_paragraph(style="List Bullet"); r = q.add_run(t_ + ": "); r.bold = True; q.add_run(de)
    d.add_heading("Limitaciones a tener en cuenta", 1)
    for x in limit:
        d.add_paragraph(x, style="List Bullet")
    d.add_heading("Próximos pasos sugeridos", 1)
    for x in pasos:
        d.add_paragraph(x, style="List Bullet")
    d.save(os.path.join(OUT, "RESUMEN_resultados.docx"))


def copy_to_cloud(files):
    try:
        os.makedirs(os.path.join(CLOUD, "figuras"), exist_ok=True)
        for f in files:
            if os.path.exists(f):
                sub = "figuras" if os.path.dirname(f).endswith("figuras") else ""
                shutil.copy2(f, os.path.join(CLOUD, sub, os.path.basename(f)))
    except Exception as e:  # noqa
        log(f"No se pudo copiar a OneDrive: {e}")


# ----------------------------------------------------------------------------- control
def run_validation_only():
    state["val_reruns"] += 1
    log("checkpoint_final existe pero falta validation/summary.json: re-ejecutando SOLO la validación con 2 procesos.")
    rc = subprocess.call([PY, WRAPPER, os.path.join(W, "_nnunet_logs", "06_validacion_reintento.log"), "1", "2d", "0", "-tr", TRAINER, "--val"],
                         env=nnunet_env({"nnUNet_def_n_proc": "2"}))
    log(f"Validación re-ejecutada, código {rc}.")


def resume_training():
    state["resumes"] += 1
    log("El entrenamiento terminó con error antes de checkpoint_final: reanudando UNA vez con --c (nnUNet_def_n_proc=2).")
    subprocess.Popen([PY, WRAPPER, WRAPLOG, "1", "2d", "0", "-tr", TRAINER, "--c"], env=nnunet_env({"nnUNet_def_n_proc": "2"}),
                     creationflags=0x08000000)


def main():
    once = "--once" in sys.argv
    log(f"Vigilante iniciado (once={once}).")
    while True:
        try:
            epochs = parse_training_logs()
            starts, ends = wrapper_status()
            running = starts > len(ends) and last_log_age_min() < 30
            ckpt_final = os.path.exists(os.path.join(FOLD, "checkpoint_final.pth"))
            summary = os.path.exists(os.path.join(FOLD, "validation", "summary.json"))
            if running:
                build_report(epochs, final=False)
            elif ckpt_final and summary:
                build_report(epochs, final=True)
                if once:
                    return
                log("Listo. Informe final copiado a OneDrive. Fin del vigilante.")
                return
            elif once:
                build_report(epochs, final=False)
            elif ckpt_final and not summary:
                if state["val_reruns"] < 1:
                    build_report(epochs, final=False, note="Entrenamiento terminado; la validación final falló y se está re-ejecutando.")
                    run_validation_only()
                    continue
                build_report(epochs, final=True, note="La validación final falló dos veces; solo hay curvas de entrenamiento.")
                return
            elif ends and ends[-1] != 0 and state["resumes"] < 1:
                build_report(epochs, final=False, note="El entrenamiento se interrumpió con error; se reanudó automáticamente desde el último checkpoint.")
                resume_training()
            else:
                build_report(epochs, final=False, note="El entrenamiento no parece estar corriendo (sin actividad en el log). Revisar el PC.")
            if once:
                return
            if dt.datetime.now() > DEADLINE:
                log("Se alcanzó la hora límite del vigilante. Fin.")
                return
        except Exception as e:  # noqa
            log(f"Error en el ciclo: {e!r}")
            if once:
                raise
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
