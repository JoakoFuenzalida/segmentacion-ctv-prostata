"""Lanza nnUNetv2_train y evita que Windows entre en suspension mientras corre.

Uso: python run_train_keepawake.py <log_path> <args de nnUNetv2_train...>
No modifica la configuracion de energia: solo pide a Windows (SetThreadExecutionState)
que no suspenda el equipo mientras este proceso este vivo, como hace un reproductor de video.
"""
import ctypes
import datetime
import os
import subprocess
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

log_path = sys.argv[1]
train_args = sys.argv[2:]
scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
exe = os.path.join(scripts, "nnUNetv2_train.exe")

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
with open(log_path, "a", encoding="utf-8", buffering=1) as log:
    log.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] INICIO: {exe} {' '.join(train_args)}\n")
    log.write("ENV: " + " ".join(f"{k}={os.environ.get(k)}" for k in
              ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results", "nnUNet_n_proc_DA", "nnUNet_compile")) + "\n")
    rc = subprocess.call([exe] + train_args, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] FIN: codigo de salida {rc}\n")
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
sys.exit(rc)
