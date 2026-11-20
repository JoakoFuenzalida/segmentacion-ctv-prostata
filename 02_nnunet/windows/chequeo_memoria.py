"""Chequeo previo al entrenamiento (solo lectura). Sale con codigo 1 si probablemente NO alcanza.

Medido el 19/11/2026: el entrenamiento 2d (proceso principal + 3 workers) necesita ~17 GB de memoria
comprometida (RAM + archivo de paginacion). Lo que falte, Windows lo saca agrandando C:\\pagefile.sys,
es decir, del espacio libre del disco. Ademas los checkpoints ocupan ~1,1 GB de disco.
"""
import ctypes
import shutil
import sys

NEED_GB = 19.0      # memoria comprometida que usa el entrenamiento (19/11: el pagefile crecio ~13 GB con 6,3 GB libres)
DISK_MIN_GB = 3.0   # disco que debe quedar libre al final (checkpoints ~1,1 GB + margen)


class MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


m = MEMSTAT(); m.dwLength = ctypes.sizeof(MEMSTAT)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
commit_free = m.ullAvailPageFile / 1024 ** 3
disk = shutil.disk_usage("C:\\").free / 1024 ** 3
falta = max(0.0, NEED_GB - commit_free)          # lo que Windows tendria que sacar del disco
disk_after = disk - falta - 1.1
print(f"Memoria comprometida libre: {commit_free:.1f} GB | Disco C: libre: {disk:.1f} GB")
print(f"Estimado: el archivo de paginacion crecera ~{falta:.1f} GB -> disco libre al final ~{disk_after:.1f} GB")
if disk_after < DISK_MIN_GB:
    print("NO ALCANZA con seguridad. Cierra Docker Desktop (y en una terminal: wsl --shutdown), Chrome, Discord")
    print("y la app de ChatGPT, o libera espacio en C:, y vuelve a ejecutar este .bat.")
    sys.exit(1)
print("OK: deberia alcanzar.")
