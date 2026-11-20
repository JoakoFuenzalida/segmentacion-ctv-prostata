"""
ver_hu.py

Muestra los valores y un corte de un volumen guardado por extraer_hu.py y,
opcionalmente, superpone el contorno de una estructura generada por
generar_mascaras.py. Con --comparar, dibuja además el contorno original del
RTSTRUCT (tal como lo delineó el equipo clínico) y mide la diferencia.

Uso:
    python ver_hu.py salida/PACIENTE.npz                    # corte del medio
    python ver_hu.py salida/PACIENTE.npz 75                 # corte 75
    python ver_hu.py salida/PACIENTE.npz --listar           # estructuras disponibles
    python ver_hu.py salida/PACIENTE.npz --roi Bladder      # corte central del órgano
    python ver_hu.py salida/PACIENTE.npz 75 --roi FEMUR DER # nombres con espacios
    python ver_hu.py salida/PACIENTE.npz --roi FEMUR DER --comparar "C:/ruta/a/PACIENTE"
    python ver_hu.py salida/PACIENTE.npz --roi FEMUR DER --ventana hueso

(si indicas un número de corte, escríbelo antes de --roi; --comparar recibe la
carpeta DICOM del paciente, la misma que se usó en generar_mascaras.py)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

VENTANAS = {"blando": (-160, 240), "hueso": (-400, 1500), "amplia": (-1000, 1000)}

p = argparse.ArgumentParser(description="Visualiza un corte en HU y, opcionalmente, una máscara.")
p.add_argument("npz", type=Path, help="archivo <paciente>.npz de extraer_hu.py")
p.add_argument("z", nargs="?", type=int, help="número de corte (por defecto, el del medio)")
p.add_argument("--roi", nargs="+",
               help="estructura a superponer; acepta espacios (FEMUR DER) o la clave (FEMUR_DER)")
p.add_argument("--listar", action="store_true", help="lista las estructuras y termina")
p.add_argument("--comparar", type=Path, metavar="CARPETA_DICOM",
               help="superpone el contorno original del RTSTRUCT y mide la diferencia")
p.add_argument("--ventana", choices=VENTANAS, default="blando", help="ventana de visualización")
args = p.parse_args()
if args.comparar and not args.roi:
    sys.exit("--comparar necesita --roi.")

hu = np.load(args.npz)["hu"]                          # forma [z, fila, columna]
ruta_masc = args.npz.with_name(f"{args.npz.stem}_mascaras.npz")
ruta_info = args.npz.with_name(f"{args.npz.stem}_mascaras.json")
ruta_meta = args.npz.with_name(f"{args.npz.stem}_meta.json")

mascara, nombre_roi = None, None
if args.listar or args.roi:
    if not ruta_info.exists():
        sys.exit(f"No existe {ruta_masc}; ejecuta primero generar_mascaras.py.")
    info = json.loads(ruta_info.read_text(encoding="utf-8"))
    if args.listar:
        for e in info["estructuras"]:
            print(f"{e['nombre']:<28} clave={e['clave']:<24} {e['volumen_cc']:>9.1f} cm³")
        sys.exit(0)

    def normalizar(texto):  # ignora mayúsculas, espacios, guiones y guiones bajos
        return re.sub(r"[^0-9a-z]", "", texto.lower())

    buscada = normalizar(" ".join(args.roi))
    coincide = [e for e in info["estructuras"]
                if buscada in (normalizar(e["nombre"]), normalizar(e["clave"]))]
    if not coincide:
        sys.exit("No se encontró la estructura. Disponibles: "
                 + ", ".join(e["nombre"] for e in info["estructuras"]))
    nombre_roi = coincide[0]["nombre"]
    mascara = np.load(ruta_masc)[coincide[0]["clave"]]

if args.z is not None:
    z = args.z
elif mascara is not None and mascara.any():
    cortes = np.flatnonzero(mascara.any(axis=(1, 2)))
    z = int(cortes[len(cortes) // 2])                 # corte central del órgano
else:
    z = hu.shape[0] // 2

print("Forma del volumen:", hu.shape, "| tipo:", hu.dtype)
print("HU mínimo y máximo:", hu.min(), hu.max())
print(f"Bloque de 5 x 5 valores del corte {z}:")
print(hu[z, 250:255, 250:255])
if mascara is not None:
    dentro = hu[z][mascara[z]]
    if dentro.size:
        print(f"'{nombre_roi}' en el corte {z}: {dentro.size} vóxeles, "
              f"HU media {dentro.mean():.1f}, mediana {np.median(dentro):.1f}")
    else:
        print(f"'{nombre_roi}' no tiene contorno en el corte {z}.")

# ---------------------------------------------------------------------------
# Comparación con el contorno original del RTSTRUCT
# ---------------------------------------------------------------------------
originales = {}
if args.comparar:
    from generar_mascaras import area_poligono, contornos_originales

    _, originales = contornos_originales(args.comparar, ruta_meta, nombre_roi)
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    sr, sc = meta["pixel_spacing_mm"]
    dz = meta.get("espaciado_entre_cortes_mm") or meta.get("slice_thickness_mm") or 1.0
    a_cc = sr * sc * dz / 1000                        # cm³ por píxel² de un corte

    cortes_ref = sorted(originales)
    area_real = np.array([sum(area_poligono(c, f) for c, f in originales[k]) for k in cortes_ref])
    area_masc = np.array([int(mascara[k].sum()) for k in cortes_ref])
    vol_real, vol_masc = area_real.sum() * a_cc, area_masc.sum() * a_cc
    print(f"\nComparación de '{nombre_roi}' con el contorno original del RTSTRUCT")
    print(f"  Cortes con contorno: {len(cortes_ref)}")
    print(f"  Volumen según el contorno original: {vol_real:9.2f} cm³")
    print(f"  Volumen según la máscara:           {vol_masc:9.2f} cm³")
    if vol_real > 0:
        print(f"  Diferencia: {100 * (vol_masc - vol_real) / vol_real:+.2f} %")
        rel = np.where(area_real > 0, np.abs(area_masc - area_real) / np.maximum(area_real, 1e-9), 0)
        peor = int(np.argmax(rel))
        print(f"  Mayor diferencia por corte: {100 * rel[peor]:.1f} % en el corte {cortes_ref[peor]} "
              f"({area_real[peor]:.0f} px² del contorno vs {area_masc[peor]} píxeles de la máscara)")
    if z in originales:
        a_z = sum(area_poligono(c, f) for c, f in originales[z])
        print(f"  Corte {z}: contorno {a_z:.1f} px², máscara {int(mascara[z].sum())} píxeles")

# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402  (se importa después de validar argumentos)
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

vmin, vmax = VENTANAS[args.ventana]
plt.imshow(hu[z], cmap="gray", vmin=vmin, vmax=vmax)
plt.colorbar(label="HU")
leyenda = []
if mascara is not None and mascara[z].any():
    if args.comparar:
        # con --comparar se pintan los píxeles de la máscara, para ver exactamente
        # cuáles quedaron dentro respecto del contorno original
        plt.imshow(np.ma.masked_where(~mascara[z], mascara[z]),
                   cmap=ListedColormap(["red"]), alpha=0.4, interpolation="nearest")
        leyenda.append(Patch(color="red", alpha=0.4, label="Píxeles de la máscara"))
    else:
        plt.contour(mascara[z], levels=[0.5], colors="red", linewidths=1.5)
        leyenda.append(Line2D([], [], color="red", lw=1.5, label="Máscara"))
for cols, fils in originales.get(z, []):
    plt.plot(np.append(cols, cols[0]), np.append(fils, fils[0]), color="cyan", lw=1.2, ls="--")
if originales.get(z):
    leyenda.append(Line2D([], [], color="cyan", lw=1.2, ls="--", label="Contorno original (RTSTRUCT)"))
    # acercamiento a la estructura para que la diferencia sea visible
    filas_m, cols_m = np.nonzero(mascara[z]) if mascara[z].any() else (np.array([0]), np.array([0]))
    todas_c = np.concatenate([c for c, _ in originales[z]] + [cols_m])
    todas_f = np.concatenate([f for _, f in originales[z]] + [filas_m])
    margen = 15
    plt.xlim(todas_c.min() - margen, todas_c.max() + margen)
    plt.ylim(todas_f.max() + margen, todas_f.min() - margen)
if leyenda:
    plt.legend(handles=leyenda, loc="lower right", fontsize=8)
titulo = f"Corte {z} de {hu.shape[0]}"
plt.title(f"{titulo} – {nombre_roi}" if nombre_roi else titulo)
plt.show()
