"""Pasos encadenados tras el entrenamiento nnUNetTrainerNoDA_250epochs (lo llama lanzar_entrenamiento_2d_fold0_noDA.bat).

1. Si no hay checkpoint_final.pth: reanuda UNA vez con --c (via run_train_keepawake.py).
2. Si hay checkpoint_final pero falta validation/summary.json: re-ejecuta solo la validacion (--val) con 2 procesos.
3. Predice el test (imagesTs) en pred_test_noDA_2d (sin sobrescribir nada existente).
4. Metricas (Dice, IoU, HD95, volumenes) en validacion (7) y test (8) del modelo nuevo -> CSV nuevos.
5. Informe comparativo _nnunet_resumen/COMPARACION_baseline_vs_noDA.md/.html.
Mantiene el equipo despierto mientras corre (SetThreadExecutionState). No borra ni sobrescribe nada.
"""
import ctypes
import datetime as dt
import glob
import os
import shutil
import subprocess
import sys
import traceback

W = r"C:\Users\joaki\Documents\Tesis_segmentacion"
PY = r"C:\Users\joaki\anaconda3\envs\nnunet\python.exe"
SCRIPTS = os.path.join(W, "_nnunet_scripts")
EXE = r"C:\Users\joaki\anaconda3\envs\nnunet\Scripts"
RAW = os.path.join(W, "nnUNet_raw", "Dataset001_ProstataCTV")
TR = "nnUNetTrainerNoDA_250epochs"
FOLD = os.path.join(W, "nnUNet_results", "Dataset001_ProstataCTV", f"{TR}__nnUNetPlans__2d", "fold_0")
TRAINLOG = os.path.join(W, "_nnunet_logs", "10_train_2d_fold0_noDA.log")
PRED = os.path.join(W, "pred_test_noDA_2d")
VAL_CASES = ["PROS_001", "PROS_004", "PROS_009", "PROS_013", "PROS_016", "PROS_019", "PROS_028"]


def log(msg):
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def env(extra=None):
    e = dict(os.environ)
    e.update({"nnUNet_raw": os.path.join(W, "nnUNet_raw"), "nnUNet_preprocessed": os.path.join(W, "nnUNet_preprocessed"),
              "nnUNet_results": os.path.join(W, "nnUNet_results"), "nnUNet_n_proc_DA": "3", "nnUNet_compile": "f",
              "PYTHONUNBUFFERED": "1", "nnUNet_def_n_proc": "3"})
    e.update(extra or {})
    return e


def run(cmd, logfile, extra_env=None):
    log("Ejecutando: " + " ".join(cmd) + f"  (log: {logfile})")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {' '.join(cmd)}\n"); f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env(extra_env), cwd=W)
    log(f"codigo de salida {rc}")
    return rc


def free_gb():
    return shutil.disk_usage("C:\\").free / 1024 ** 3


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log(f"=== post_noDA inicio. Disco libre C: {free_gb():.1f} GB")
    final = os.path.join(FOLD, "checkpoint_final.pth")
    summary = os.path.join(FOLD, "validation", "summary.json")

    # 1. reanudar una vez si el entrenamiento no termino
    if not os.path.exists(final):
        log("No existe checkpoint_final.pth: el entrenamiento no termino. Reanudo UNA vez con --c.")
        run([PY, os.path.join(SCRIPTS, "run_train_keepawake.py"), TRAINLOG, "1", "2d", "0", "-tr", TR, "--c"],
            os.path.join(W, "_nnunet_logs", "12_reanudacion_noDA.log"))
    # 2. validacion sola si falto
    if os.path.exists(final) and not os.path.exists(summary):
        log("Hay checkpoint_final pero falta validation/summary.json: re-ejecuto solo la validacion con 2 procesos.")
        run([os.path.join(EXE, "nnUNetv2_train.exe"), "1", "2d", "0", "-tr", TR, "--val"],
            os.path.join(W, "_nnunet_logs", "13_validacion_reintento_noDA.log"), {"nnUNet_def_n_proc": "2"})

    # 3. prediccion del test
    if os.path.exists(final):
        n_in = len(glob.glob(os.path.join(RAW, "imagesTs", "*_0000.nii.gz")))
        n_out = len(glob.glob(os.path.join(PRED, "*.nii.gz")))
        if n_out < n_in:
            run([os.path.join(EXE, "nnUNetv2_predict.exe"), "-i", os.path.join(RAW, "imagesTs"), "-o", PRED, "-d", "1",
                 "-c", "2d", "-f", "0", "-tr", TR, "-npp", "1", "-nps", "1", "--continue_prediction"],
                os.path.join(W, "_nnunet_logs", "14_pred_test_noDA_2d.log"))
        else:
            log("pred_test_noDA_2d ya esta completo; no se repite.")
    else:
        log("Sigue sin existir checkpoint_final.pth: no se predice el test. El informe se genera con lo disponible.")

    # 4. metricas (CSV nuevos; no se sobrescribe ninguno existente)
    sys.path.insert(0, SCRIPTS)
    import metricas_seg as M
    jobs = [(os.path.join(FOLD, "validation"), os.path.join(RAW, "labelsTr"), os.path.join(W, "resultados_val_noDA_2d.csv"), VAL_CASES),
            (PRED, os.path.join(RAW, "labelsTs"), os.path.join(W, "resultados_test_noDA_2d.csv"), None),
            (os.path.join(W, "nnUNet_results", "Dataset001_ProstataCTV", "nnUNetTrainer_250epochs__nnUNetPlans__2d", "fold_0", "validation"),
             os.path.join(RAW, "labelsTr"), os.path.join(W, "resultados_val_baseline_2d.csv"), VAL_CASES),
            (os.path.join(W, "pred_test_baseline_2d"), os.path.join(RAW, "labelsTs"), os.path.join(W, "resultados_test_baseline_2d.csv"), None)]
    for pred_dir, lab, out, cases in jobs:
        if os.path.exists(out):
            log(f"{os.path.basename(out)} ya existe; no se sobrescribe.")
            continue
        if cases is not None and not all(os.path.exists(os.path.join(pred_dir, f"{c}.nii.gz")) for c in cases):
            log(f"Faltan predicciones en {pred_dir}; no se calcula {os.path.basename(out)}.")
            continue
        if cases is None and not glob.glob(os.path.join(pred_dir, "*.nii.gz")):
            log(f"No hay predicciones en {pred_dir}; no se calcula {os.path.basename(out)}.")
            continue
        try:
            rows = M.evaluate(pred_dir, lab, cases); M.write_csv(rows, out)
            log(f"{os.path.basename(out)}: Dice medio {sum(r['dice'] for r in rows) / len(rows):.4f} (n={len(rows)})")
        except Exception:  # noqa
            log("Error calculando " + out + "\n" + traceback.format_exc())

    # 5. informe comparativo
    rc = subprocess.call([PY, os.path.join(SCRIPTS, "comparar_baseline_vs_noDA.py")], env=env(), cwd=W)
    log(f"Informe comparativo: codigo {rc}. Disco libre C: {free_gb():.1f} GB")
    log("=== post_noDA FIN")
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa
        log("ERROR NO CONTROLADO\n" + traceback.format_exc())
        raise
