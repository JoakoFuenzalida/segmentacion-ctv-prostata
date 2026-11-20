#!/usr/bin/env python3
"""
generar_mascaras.py

Genera, a partir del RTSTRUCT de cada paciente, una máscara binaria por
estructura (vejiga, recto, cabezas femorales, CTV, PTV, etc.), alineada vóxel
a vóxel con el volumen en HU que produce extraer_hu.py.

Requiere haber ejecutado antes extraer_hu.py sobre el mismo paciente, porque
usa la geometría guardada en <paciente>_meta.json (posición de cada corte,
orientación y espaciado).

Salida por paciente (en la misma carpeta de salida de extraer_hu.py):
    <paciente>_mascaras.npz    un arreglo bool [z, fila, columna] por estructura
    <paciente>_mascaras.json   nombre original de cada estructura, clave en el
                               .npz, volumen en cm³, cortes con contorno y avisos

Uso:
    python generar_mascaras.py datos/PACIENTE salida
    python generar_mascaras.py datos/ salida --lote
    python generar_mascaras.py datos/PACIENTE salida --roi Bladder Rectum
    python generar_mascaras.py datos/PACIENTE salida --roi "FEMUR DER" FEMUR_IZQ
    (un nombre con espacios va entre comillas o con guion bajo en vez de espacio)

Criterio de rasterización: un vóxel pertenece a la estructura si su centro
queda dentro del contorno. Si una estructura tiene varios contornos en el
mismo corte, se combinan con la regla par-impar (XOR), que es la convención
de DICOM para representar huecos (por ejemplo, un anillo).

Requisitos: numpy y pydicom (pip install numpy pydicom).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pydicom


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def _geometria(meta: dict) -> dict:
    """Prepara, desde el _meta.json, lo necesario para pasar de mm a índices."""
    iop = np.asarray(meta["image_orientation_patient"], dtype=float)
    pos = np.asarray(meta["posiciones_sobre_normal_mm"], dtype=float)
    dz = meta.get("espaciado_entre_cortes_mm") or meta.get("slice_thickness_mm") or 1.0
    return {
        "dir_columnas": iop[:3],          # hacia donde aumenta el índice de columna
        "dir_filas": iop[3:],             # hacia donde aumenta el índice de fila
        "normal": np.asarray(meta["normal_al_plano"], dtype=float),
        "ipp": np.asarray(meta["image_position_patient"], dtype=float),
        "pos": pos,
        "esp_fila": float(meta["pixel_spacing_mm"][0]),     # distancia entre filas
        "esp_columna": float(meta["pixel_spacing_mm"][1]),  # distancia entre columnas
        "dz": float(dz),
        "tolerancia": float(dz) / 2,
        "forma": tuple(meta["forma_zyx"]),
    }


def _contorno_a_indices(puntos: np.ndarray, geo: dict):
    """Convierte puntos (x, y, z) en mm a (corte, columnas, filas) del volumen.

    Devuelve None si el contorno no cae sobre ningún corte de la serie.
    """
    s = puntos @ geo["normal"]
    k = int(np.argmin(np.abs(geo["pos"] - s.mean())))
    if abs(geo["pos"][k] - s.mean()) > geo["tolerancia"]:
        return None
    d = puntos - geo["ipp"][k]
    columnas = d @ geo["dir_columnas"] / geo["esp_columna"]
    filas = d @ geo["dir_filas"] / geo["esp_fila"]
    return k, columnas, filas


def _rellenar_poligono(x: np.ndarray, y: np.ndarray, n_filas: int, n_columnas: int) -> np.ndarray:
    """Rellena un polígono cerrado dado en coordenadas de índice (x = columna,
    y = fila). Un píxel pertenece si su centro está dentro (regla par-impar)."""
    mascara = np.zeros((n_filas, n_columnas), dtype=bool)
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    no_horizontal = y != y2
    x1, y1, x2, y2 = x[no_horizontal], y[no_horizontal], x2[no_horizontal], y2[no_horizontal]
    if x1.size == 0:
        return mascara

    f0 = max(int(np.ceil(y.min())), 0)
    f1 = min(int(np.floor(y.max())), n_filas - 1)
    if f0 > f1:
        return mascara

    filas = np.arange(f0, f1 + 1, dtype=float)[:, None]                 # R x 1
    cruza = ((y1 <= filas) & (filas < y2)) | ((y2 <= filas) & (filas < y1))  # R x E
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = x1 + (filas - y1) * (x2 - x1) / (y2 - y1)
    xs = np.sort(np.where(cruza, xs, np.inf), axis=1)
    n_cruces = cruza.sum(axis=1)

    columnas = np.arange(n_columnas, dtype=float)
    for i, fila in enumerate(range(f0, f1 + 1)):
        if n_cruces[i] >= 2:
            cruces = xs[i, : n_cruces[i]]
            mascara[fila] = np.searchsorted(cruces, columnas, side="right") % 2 == 1
    return mascara


# ---------------------------------------------------------------------------
# RTSTRUCT
# ---------------------------------------------------------------------------
def _frames_referenciados(ds) -> set:
    return {str(getattr(item, "FrameOfReferenceUID", ""))
            for item in (getattr(ds, "ReferencedFrameOfReferenceSequence", None) or [])}


def _elegir_rtstruct(carpeta: Path, meta: dict, avisos: list):
    """Elige el RTSTRUCT que referencia el mismo sistema de coordenadas que el TC."""
    if not meta.get("archivos_rtstruct"):
        raise FileNotFoundError("El paciente no tiene RTSTRUCT (según el _meta.json).")
    compatibles = []
    for rel in meta["archivos_rtstruct"]:
        ds = pydicom.dcmread(str(carpeta / rel))
        frames = _frames_referenciados(ds)
        if not frames or not meta.get("frame_of_reference_uid") or meta["frame_of_reference_uid"] in frames:
            compatibles.append((rel, ds))
    if not compatibles:
        raise ValueError(
            "Ningún RTSTRUCT referencia el FrameOfReferenceUID del TC extraído; "
            "probablemente se seleccionó otra serie de TC. Revisa los avisos de extraer_hu.py.")
    if len(compatibles) > 1:
        avisos.append(f"Hay {len(compatibles)} RTSTRUCT compatibles; se usa {compatibles[0][0]}.")
    return compatibles[0]


def _normalizar(texto: str) -> str:
    """Ignora mayúsculas, espacios, guiones y guiones bajos al comparar nombres."""
    return re.sub(r"[^0-9a-z]", "", texto.lower())


def _clave(nombre: str, numero: int, usadas: set) -> str:
    clave = re.sub(r"[^0-9A-Za-z_-]+", "_", nombre).strip("_") or f"roi_{numero}"
    if clave in usadas:
        clave = f"{clave}_{numero}"
    usadas.add(clave)
    return clave


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------
def generar_mascaras(carpeta, salida, rois=None):
    """Devuelve (mascaras, resumen).

    mascaras: dict {clave: np.ndarray bool [z, fila, columna]}
    resumen:  dict con la descripción de cada estructura y los avisos.
    """
    carpeta, salida = Path(carpeta), Path(salida)
    ruta_meta = salida / f"{carpeta.name}_meta.json"
    if not ruta_meta.exists():
        raise FileNotFoundError(f"No existe {ruta_meta}; ejecuta primero extraer_hu.py.")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    geo = _geometria(meta)
    n_z, n_filas, n_columnas = geo["forma"]
    sop_serie = set(meta.get("sop_instance_uids", []))

    avisos = []
    rel_rt, rt = _elegir_rtstruct(carpeta, meta, avisos)

    nombres = {int(r.ROINumber): str(r.ROIName)
               for r in (getattr(rt, "StructureSetROISequence", None) or [])}
    filtro = {_normalizar(r) for r in rois} if rois else None

    mascaras, estructuras, usadas = {}, [], set()
    for roi in getattr(rt, "ROIContourSequence", None) or []:
        numero = int(roi.ReferencedROINumber)
        nombre = nombres.get(numero, f"ROI_{numero}")
        if filtro and _normalizar(nombre) not in filtro:
            continue

        mascara = np.zeros((n_z, n_filas, n_columnas), dtype=bool)
        n_contornos = fuera_de_serie = otra_imagen = 0
        for c in getattr(roi, "ContourSequence", None) or []:
            if str(getattr(c, "ContourGeometricType", "CLOSED_PLANAR")).upper() != "CLOSED_PLANAR":
                continue
            puntos = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            if len(puntos) < 3:
                continue
            for img in getattr(c, "ContourImageSequence", None) or []:
                if sop_serie and str(img.ReferencedSOPInstanceUID) not in sop_serie:
                    otra_imagen += 1
            indices = _contorno_a_indices(puntos, geo)
            if indices is None:
                fuera_de_serie += 1
                continue
            k, cols, fils = indices
            mascara[k] ^= _rellenar_poligono(cols, fils, n_filas, n_columnas)
            n_contornos += 1

        clave = _clave(nombre, numero, usadas)
        cortes = np.flatnonzero(mascara.any(axis=(1, 2)))
        voxeles = int(mascara.sum())
        if voxeles == 0:
            avisos.append(f"'{nombre}' quedó vacía (sin contornos utilizables en esta serie).")
        if fuera_de_serie:
            avisos.append(f"'{nombre}': {fuera_de_serie} contorno(s) no coinciden con ningún corte.")
        if otra_imagen:
            avisos.append(f"'{nombre}': {otra_imagen} contorno(s) referencian imágenes de otra serie.")

        color = getattr(roi, "ROIDisplayColor", None)
        mascaras[clave] = mascara
        estructuras.append({
            "nombre": nombre,
            "clave": clave,
            "roi_number": numero,
            "contornos": n_contornos,
            "voxeles": voxeles,
            "volumen_cc": round(voxeles * geo["esp_fila"] * geo["esp_columna"] * geo["dz"] / 1000, 2),
            "cortes": [int(cortes[0]), int(cortes[-1])] if cortes.size else None,
            "color_rgb": [int(v) for v in color] if color is not None else None,
        })

    if filtro:
        faltantes = [r for r in rois if _normalizar(r) not in
                     {_normalizar(e["nombre"]) for e in estructuras}]
        if faltantes:
            avisos.append("No se encontraron: " + ", ".join(faltantes)
                          + ". Nombres disponibles: " + ", ".join(sorted(nombres.values())))

    resumen = {
        "paciente": carpeta.name,
        "rtstruct": rel_rt,
        "forma_zyx": [n_z, n_filas, n_columnas],
        "criterio": "centro del vóxel dentro del contorno; contornos del mismo corte combinados por XOR",
        "estructuras": estructuras,
        "avisos": avisos,
    }
    return mascaras, resumen


def contornos_originales(carpeta, ruta_meta, roi: str):
    """Devuelve los contornos originales del RTSTRUCT de una estructura, en
    coordenadas de índice del volumen, sin rasterizar.

    Retorna (nombre, {corte: [(columnas, filas), ...]}). Sirve para comparar
    la máscara con el contorno clínico tal como fue dibujado.
    """
    carpeta = Path(carpeta)
    meta = json.loads(Path(ruta_meta).read_text(encoding="utf-8"))
    geo = _geometria(meta)
    _, rt = _elegir_rtstruct(carpeta, meta, [])
    nombres = {int(r.ROINumber): str(r.ROIName)
               for r in (getattr(rt, "StructureSetROISequence", None) or [])}
    for item in getattr(rt, "ROIContourSequence", None) or []:
        numero = int(item.ReferencedROINumber)
        nombre = nombres.get(numero, f"ROI_{numero}")
        if _normalizar(nombre) != _normalizar(roi):
            continue
        por_corte = {}
        for c in getattr(item, "ContourSequence", None) or []:
            if str(getattr(c, "ContourGeometricType", "CLOSED_PLANAR")).upper() != "CLOSED_PLANAR":
                continue
            puntos = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            indices = _contorno_a_indices(puntos, geo) if len(puntos) >= 3 else None
            if indices is not None:
                k, cols, fils = indices
                por_corte.setdefault(k, []).append((cols, fils))
        return nombre, por_corte
    raise KeyError(f"La estructura '{roi}' no está en el RTSTRUCT.")


def area_poligono(columnas: np.ndarray, filas: np.ndarray) -> float:
    """Área exacta de un polígono en píxeles² (fórmula del área de Gauss)."""
    return 0.5 * abs(np.dot(columnas, np.roll(filas, -1)) - np.dot(np.roll(columnas, -1), filas))


def guardar(mascaras: dict, resumen: dict, salida, nombre: str) -> Path:
    salida = Path(salida)
    final = salida / f"{nombre}_mascaras.npz"
    temporal = salida / f".{nombre}_mascaras.npz.tmp"
    with open(temporal, "wb") as f:
        np.savez_compressed(f, **mascaras)
    (salida / f"{nombre}_mascaras.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(final)
    return final


def cargar_mascaras(ruta_npz):
    """Carga (mascaras, resumen) guardados por este script."""
    ruta = Path(ruta_npz)
    with np.load(ruta) as datos:
        mascaras = {k: datos[k] for k in datos.files}
    resumen = json.loads(ruta.with_suffix(".json").read_text(encoding="utf-8"))
    return mascaras, resumen


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Genera máscaras por estructura desde el RTSTRUCT.")
    p.add_argument("entrada", type=Path, help="carpeta DICOM del paciente, o de pacientes con --lote")
    p.add_argument("salida", type=Path, help="carpeta de salida de extraer_hu.py")
    p.add_argument("--lote", action="store_true", help="procesa cada subcarpeta como un paciente")
    p.add_argument("--roi", nargs="+",
                   help='solo estas estructuras; un nombre con espacios va entre comillas ("FEMUR DER") '
                        "o con guion bajo (FEMUR_DER)")
    p.add_argument("--sobrescribir", action="store_true", help="regenera máscaras ya existentes")
    args = p.parse_args(argv)

    pacientes = (sorted(d for d in args.entrada.iterdir() if d.is_dir())
                 if args.lote else [args.entrada])
    errores = {}
    for i, carpeta in enumerate(pacientes, 1):
        print(f"[{i}/{len(pacientes)}] {carpeta.name}")
        if (args.salida / f"{carpeta.name}_mascaras.npz").exists() and not args.sobrescribir:
            print("    omitido (ya generado)")
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mascaras, resumen = generar_mascaras(carpeta, args.salida, args.roi)
            guardar(mascaras, resumen, args.salida, carpeta.name)
            for e in resumen["estructuras"]:
                rango = f"cortes {e['cortes'][0]}-{e['cortes'][1]}" if e["cortes"] else "vacía"
                print(f"    {e['nombre']:<28} {e['volumen_cc']:>9.1f} cm³  {rango}")
            for a in resumen["avisos"]:
                print(f"    aviso: {a}")
        except Exception as exc:
            errores[carpeta.name] = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR: {errores[carpeta.name]}")

    if errores:
        (args.salida / "errores_mascaras.json").write_text(
            json.dumps(errores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{len(errores)} paciente(s) con error; detalle en errores_mascaras.json")
    return 1 if errores and not args.lote else 0


if __name__ == "__main__":
    sys.exit(main())
