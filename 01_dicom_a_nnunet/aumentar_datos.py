#!/usr/bin/env python3
"""
aumentar_datos.py

Genera variantes aumentadas de cada paciente a partir de los archivos de
extraer_hu.py y generar_mascaras.py. A cada variante se le aplica UNA misma
transformación aleatoria en el plano axial, idéntica para todos los cortes del
volumen y para todas sus máscaras, de modo que imagen y etiqueta siguen
alineadas y la anatomía 3D se mantiene coherente.

Transformaciones (todas en el plano axial, en milímetros reales):
    - rotación aleatoria en [-rot, +rot] grados
    - escala aleatoria en [1 - escala, 1 + escala]
    - desplazamiento aleatorio en [-desp, +desp] mm en cada eje
    - ruido gaussiano opcional (desviación en HU)
No se aplica volteo izquierda-derecha: invierte la lateralidad de estructuras
pares (por ejemplo, las cabezas femorales).

Criterios:
    - Volumen en HU: interpolación bilineal; lo que queda fuera de la imagen
      original se rellena con -1024 HU (aire). Rellenar con 0 equivaldría a agua.
    - Máscaras: vecino más cercano, para que sigan siendo binarias.

Salida (en la carpeta de destino), por paciente y variante NN:
    <paciente>_augNN.npz             volumen aumentado, int16 [z, fila, columna]
    <paciente>_augNN_mascaras.npz    máscaras aumentadas (mismo formato)
    <paciente>_augNN_mascaras.json   resumen compatible con ver_hu.py
    <paciente>_augNN_aumento.json    parámetros exactos de la transformación
Con --nrrd, además, en la subcarpeta nrrd/ (formato que 3D Slicer abre directo):
    <paciente>_orig.nrrd, <paciente>_augNN.nrrd        volúmenes
    <paciente>_orig_<roi>.nrrd, <paciente>_augNN_<roi>.nrrd   máscaras

Uso:
    python aumentar_datos.py salida/PACIENTE.npz aumentados --n 5 --nrrd
    python aumentar_datos.py salida aumentados --lote --n 5 --roi Bladder "FEMUR DER"
    python aumentar_datos.py salida/PACIENTE.npz aumentados --n 3 --rot 15 --ruido 5

Importante: aumenta solo los pacientes del conjunto de entrenamiento. La
división en entrenamiento, validación y prueba debe hacerse por paciente y
antes de aumentar.

Requisitos: numpy (pip install numpy).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HU_AIRE = -1024


# ---------------------------------------------------------------------------
# Transformación
# ---------------------------------------------------------------------------
def sortear_parametros(rng: np.random.Generator, rot: float, escala: float,
                       desp: float, ruido: float) -> dict:
    return {
        "rotacion_grados": float(rng.uniform(-rot, rot)),
        "escala": float(rng.uniform(1 - escala, 1 + escala)),
        "desplazamiento_mm": [float(rng.uniform(-desp, desp)), float(rng.uniform(-desp, desp))],
        "ruido_hu": float(ruido),
    }


def coordenadas_origen(forma_plano, espaciado, parametros):
    """Para cada píxel de salida (fila, columna), su posición en la imagen
    original (transformación inversa), trabajando en mm alrededor del centro."""
    n_filas, n_cols = forma_plano
    esp_fila, esp_col = espaciado
    cf, cc = (n_filas - 1) / 2, (n_cols - 1) / 2
    filas, cols = np.meshgrid(np.arange(n_filas), np.arange(n_cols), indexing="ij")
    x = (cols - cc) * esp_col - parametros["desplazamiento_mm"][0]
    y = (filas - cf) * esp_fila - parametros["desplazamiento_mm"][1]
    t = np.deg2rad(parametros["rotacion_grados"])
    s = parametros["escala"]
    xo = (np.cos(t) * x + np.sin(t) * y) / s       # rotación inversa y escala inversa
    yo = (-np.sin(t) * x + np.cos(t) * y) / s
    return yo / esp_fila + cf, xo / esp_col + cc


def _bilineal(corte, fo, co, relleno):
    m = 2
    p = np.pad(corte.astype(np.float32), m, constant_values=relleno)
    f0 = np.floor(fo).astype(np.int64)
    c0 = np.floor(co).astype(np.int64)
    df = (fo - f0).astype(np.float32)
    dc = (co - c0).astype(np.float32)
    # índices fuera de rango caen en el margen de relleno
    f0 = np.clip(f0, -m, corte.shape[0]) + m
    c0 = np.clip(c0, -m, corte.shape[1]) + m
    return ((1 - df) * (1 - dc) * p[f0, c0] + (1 - df) * dc * p[f0, c0 + 1]
            + df * (1 - dc) * p[f0 + 1, c0] + df * dc * p[f0 + 1, c0 + 1])


def _vecino(corte, fo, co):
    fi, ci = np.rint(fo).astype(np.int64), np.rint(co).astype(np.int64)
    valido = (fi >= 0) & (fi < corte.shape[0]) & (ci >= 0) & (ci < corte.shape[1])
    salida = corte[np.clip(fi, 0, corte.shape[0] - 1), np.clip(ci, 0, corte.shape[1] - 1)]
    return salida & valido


def transformar(hu, mascaras: dict, espaciado, parametros, rng):
    """Aplica la misma transformación a todo el volumen y a todas las máscaras."""
    fo, co = coordenadas_origen(hu.shape[1:], espaciado, parametros)
    hu_t = np.empty(hu.shape, dtype=np.int16)
    for k in range(hu.shape[0]):
        corte = _bilineal(hu[k], fo, co, HU_AIRE)
        if parametros["ruido_hu"] > 0:
            corte += rng.normal(0, parametros["ruido_hu"], corte.shape).astype(np.float32)
        hu_t[k] = np.clip(np.rint(corte), -32768, 32767).astype(np.int16)
    mascaras_t = {}
    for clave, m in mascaras.items():
        mt = np.zeros(m.shape, dtype=bool)
        for k in np.flatnonzero(m.any(axis=(1, 2))):   # solo cortes con estructura
            mt[k] = _vecino(m[k], fo, co)
        mascaras_t[clave] = mt
    return hu_t, mascaras_t


# ---------------------------------------------------------------------------
# Exportación NRRD (3D Slicer)
# ---------------------------------------------------------------------------
def geometria_nrrd(meta: dict) -> dict:
    """Origen y direcciones en el sistema del paciente DICOM (LPS), en mm."""
    iop = np.asarray(meta["image_orientation_patient"], dtype=float)
    ipp = np.asarray(meta["image_position_patient"], dtype=float)
    esp_fila, esp_col = meta["pixel_spacing_mm"]
    if len(ipp) > 1:
        paso_z = (ipp[-1] - ipp[0]) / (len(ipp) - 1)
    else:
        paso_z = np.asarray(meta["normal_al_plano"], float) * float(meta.get("slice_thickness_mm") or 1.0)
    return {"origen": ipp[0], "dirs": [iop[:3] * esp_col, iop[3:] * esp_fila, paso_z]}


def escribir_nrrd(ruta: Path, volumen: np.ndarray, geo: dict):
    """Escribe un .nrrd comprimido. El arreglo [z, fila, columna] en orden C
    corresponde exactamente al orden de NRRD (columna es el eje más rápido)."""
    tipo = {np.dtype("int16"): "int16", np.dtype("uint8"): "uint8"}[volumen.dtype]
    vec = lambda v: "(" + ",".join(f"{x:.10g}" for x in v) + ")"  # noqa: E731
    encabezado = (
        "NRRD0004\n"
        f"type: {tipo}\n"
        "dimension: 3\n"
        "space: left-posterior-superior\n"
        f"sizes: {volumen.shape[2]} {volumen.shape[1]} {volumen.shape[0]}\n"
        f"space directions: {' '.join(vec(d) for d in geo['dirs'])}\n"
        "kinds: domain domain domain\n"
        "endian: little\n"
        "encoding: gzip\n"
        f"space origin: {vec(geo['origen'])}\n\n"
    )
    with open(ruta, "wb") as f:
        f.write(encabezado.encode("ascii"))
        f.write(gzip.compress(np.ascontiguousarray(volumen).astype(volumen.dtype.newbyteorder("<")).tobytes(),
                              compresslevel=6))


# ---------------------------------------------------------------------------
# Proceso por paciente
# ---------------------------------------------------------------------------
def _normalizar(texto: str) -> str:
    return re.sub(r"[^0-9a-z]", "", texto.lower())


def _guardar_npz(ruta: Path, **arreglos):
    temporal = ruta.with_name(f".{ruta.name}.tmp")
    with open(temporal, "wb") as f:
        np.savez_compressed(f, **arreglos)
    temporal.replace(ruta)


def aumentar_paciente(ruta_npz: Path, destino: Path, args) -> list:
    nombre = ruta_npz.stem
    meta = json.loads(ruta_npz.with_name(f"{nombre}_meta.json").read_text(encoding="utf-8"))
    ruta_info = ruta_npz.with_name(f"{nombre}_mascaras.json")
    if not ruta_info.exists():
        raise FileNotFoundError(f"Falta {ruta_info.name}; ejecuta primero generar_mascaras.py.")
    info = json.loads(ruta_info.read_text(encoding="utf-8"))

    estructuras = info["estructuras"]
    if args.roi:
        buscadas = {_normalizar(r) for r in args.roi}
        estructuras = [e for e in estructuras
                       if _normalizar(e["nombre"]) in buscadas or _normalizar(e["clave"]) in buscadas]
        if not estructuras:
            raise ValueError("Ninguna de las estructuras pedidas existe. Disponibles: "
                             + ", ".join(e["nombre"] for e in info["estructuras"]))

    hu = np.load(ruta_npz)["hu"]
    with np.load(ruta_npz.with_name(f"{nombre}_mascaras.npz")) as datos:
        mascaras = {e["clave"]: datos[e["clave"]] for e in estructuras}

    espaciado = tuple(float(v) for v in meta["pixel_spacing_mm"])
    dz = meta.get("espaciado_entre_cortes_mm") or meta.get("slice_thickness_mm") or 1.0
    cc_por_voxel = espaciado[0] * espaciado[1] * float(dz) / 1000
    destino.mkdir(parents=True, exist_ok=True)

    geo = None
    if args.nrrd:
        geo = geometria_nrrd(meta)
        (destino / "nrrd").mkdir(exist_ok=True)
        escribir_nrrd(destino / "nrrd" / f"{nombre}_orig.nrrd", hu.astype(np.int16), geo)
        for e in estructuras:
            escribir_nrrd(destino / "nrrd" / f"{nombre}_orig_{e['clave']}.nrrd",
                          mascaras[e["clave"]].astype(np.uint8), geo)

    # Semilla reproducible por paciente: misma semilla y mismos datos -> mismas variantes.
    # Se usa un hash del nombre y no la suma de sus bytes: con identificadores
    # numericos la suma colisiona con facilidad (dos identificadores distintos pueden sumar lo mismo),
    # y dos pacientes distintos recibirian las mismas transformaciones.
    sem_paciente = int(hashlib.sha256(nombre.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng([args.semilla, sem_paciente])
    lineas = []
    for v in range(1, args.n + 1):
        etiqueta = f"{nombre}_aug{v:02d}"
        param = sortear_parametros(rng, args.rot, args.escala, args.desp, args.ruido)
        hu_t, masc_t = transformar(hu, mascaras, espaciado, param, rng)

        _guardar_npz(destino / f"{etiqueta}.npz", hu=hu_t)
        _guardar_npz(destino / f"{etiqueta}_mascaras.npz", **masc_t)
        resumen = {"paciente": etiqueta, "origen": nombre, "forma_zyx": list(hu_t.shape),
                   "estructuras": [], "avisos": []}
        for e in estructuras:
            m = masc_t[e["clave"]]
            cortes = np.flatnonzero(m.any(axis=(1, 2)))
            voxeles = int(m.sum())
            resumen["estructuras"].append({
                **{k: e[k] for k in ("nombre", "clave", "roi_number", "color_rgb") if k in e},
                "voxeles": voxeles,
                "volumen_cc": round(voxeles * cc_por_voxel, 2),
                "volumen_original_cc": e["volumen_cc"],
                "cortes": [int(cortes[0]), int(cortes[-1])] if cortes.size else None,
            })
            # una estructura que se sale de la imagen pierde volumen: se avisa
            perdida = 1 - voxeles / max(e["voxeles"] * param["escala"] ** 2, 1)
            if e["voxeles"] and perdida > 0.02:
                resumen["avisos"].append(
                    f"'{e['nombre']}' perdió cerca del {100 * perdida:.0f} % al salir del borde de la imagen.")
        (destino / f"{etiqueta}_mascaras.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        (destino / f"{etiqueta}_aumento.json").write_text(json.dumps({
            "origen": ruta_npz.name, "semilla": args.semilla, "variante": v,
            "parametros": param,
            "criterios": {"hu": "bilineal, relleno -1024 HU, redondeo a int16",
                          "mascaras": "vecino más cercano", "volteo": "no se aplica"},
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        if geo is not None:
            escribir_nrrd(destino / "nrrd" / f"{etiqueta}.nrrd", hu_t, geo)
            for e in estructuras:
                escribir_nrrd(destino / "nrrd" / f"{etiqueta}_{e['clave']}.nrrd",
                              masc_t[e["clave"]].astype(np.uint8), geo)

        lineas.append(f"aug{v:02d}: rot {param['rotacion_grados']:+.1f}°, escala {param['escala']:.3f}, "
                      f"desp ({param['desplazamiento_mm'][0]:+.1f}, {param['desplazamiento_mm'][1]:+.1f}) mm"
                      + "".join(f"\n        aviso: {a}" for a in resumen["avisos"]))
    return lineas


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aumento de datos de volúmenes en HU y sus máscaras.")
    p.add_argument("entrada", type=Path, help="<paciente>.npz, o carpeta de salida con --lote")
    p.add_argument("destino", type=Path, help="carpeta donde se guardan las variantes")
    p.add_argument("--lote", action="store_true", help="procesa todos los pacientes de la carpeta")
    p.add_argument("--n", type=int, default=5, help="variantes por paciente (por defecto 5)")
    p.add_argument("--rot", type=float, default=10.0, help="rotación máxima en grados (10)")
    p.add_argument("--escala", type=float, default=0.10, help="variación máxima de escala (0.10 = ±10 %%)")
    p.add_argument("--desp", type=float, default=10.0, help="desplazamiento máximo en mm (10)")
    p.add_argument("--ruido", type=float, default=0.0, help="ruido gaussiano en HU (0 = sin ruido)")
    p.add_argument("--roi", nargs="+", help='solo estas estructuras ("FEMUR DER" o FEMUR_DER)')
    p.add_argument("--semilla", type=int, default=0, help="semilla para reproducir las variantes")
    p.add_argument("--nrrd", action="store_true", help="exporta original y variantes a NRRD para 3D Slicer")
    args = p.parse_args(argv)

    if args.lote:
        pacientes = sorted(r for r in args.entrada.glob("*.npz")
                           if not r.stem.endswith("_mascaras")
                           and r.with_name(f"{r.stem}_mascaras.npz").exists())
    else:
        pacientes = [args.entrada]

    errores = {}
    for i, ruta in enumerate(pacientes, 1):
        print(f"[{i}/{len(pacientes)}] {ruta.stem}")
        try:
            for linea in aumentar_paciente(ruta, args.destino, args):
                print(f"    {linea}")
        except Exception as exc:
            errores[ruta.stem] = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR: {errores[ruta.stem]}")
    if errores:
        print(f"{len(errores)} paciente(s) con error.")
    return 1 if errores and not args.lote else 0


if __name__ == "__main__":
    sys.exit(main())
