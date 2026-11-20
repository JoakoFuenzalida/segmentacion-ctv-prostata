"""
ver_hu.py

Muestra los valores y un corte de un volumen guardado por extraer_hu.py.

Uso:
    python ver_hu.py salida/PACIENTE.npz          # corte del medio
    python ver_hu.py salida/PACIENTE.npz 75       # corte 75
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

ruta = sys.argv[1] if len(sys.argv) > 1 else "salida/PACIENTE.npz"

hu = np.load(ruta)["hu"]                     # forma [z, fila, columna]
z = int(sys.argv[2]) if len(sys.argv) > 2 else hu.shape[0] // 2

print("Forma del volumen:", hu.shape, "| tipo:", hu.dtype)
print("HU mínimo y máximo:", hu.min(), hu.max())
print(f"Bloque de 5 x 5 valores del corte {z}:")
print(hu[z, 250:255, 250:255])

plt.imshow(hu[z], cmap="gray", vmin=-160, vmax=240)   # ventana de tejido blando
plt.colorbar(label="HU")
plt.title(f"Corte {z} de {hu.shape[0]}")
plt.show()
