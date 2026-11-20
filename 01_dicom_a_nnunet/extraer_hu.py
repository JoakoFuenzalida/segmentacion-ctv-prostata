#!/usr/bin/env python3
"""
extraer_hu.py

Extrae los volúmenes de tomografía computarizada (TC) en unidades Hounsfield
(HU) desde las carpetas DICOM de cada paciente, sin pérdida de información, y
guarda junto a cada volumen los metadatos necesarios para, más adelante,
generar máscaras desde el RTSTRUCT y volver a escribir en DICOM los contornos
que produzca la red.

Conversión aplicada (DICOM PS3.3, Modality LUT), corte por corte:
    HU = valor_almacenado * RescaleSlope + RescaleIntercept

Salida por paciente (en la carpeta de salida):
    <paciente>.npz        arreglo "hu" con forma [z, fila, columna]
                          (int16 si la conversión es entera, como es habitual
                          en TC; float32 en caso contrario)
    <paciente>_meta.json  geometría, UIDs y archivos de origen

Uso:
    # Un paciente (carpeta con los CT_Image_Storage_XXX y el RTSTRUCT)
    python extraer_hu.py datos/PACIENTE salida/

    # Todos los pacientes: una subcarpeta por paciente dentro de "datos/"
    python extraer_hu.py datos/ salida/ --lote

    # Opciones útiles
    --padding conservar     no reemplaza los píxeles de relleno
    --sobrescribir          vuelve a procesar pacientes ya extraídos
    --min-libre-gb 10       detiene el lote si queda menos espacio en disco

Para leer un resultado desde otro script:
    from extraer_hu import cargar_hu
    hu, meta = cargar_hu("salida/PACIENTE.npz")

Requisitos: Python 3.9 o superior, numpy y pydicom
    pip install numpy pydicom
Si las imágenes vienen comprimidas (JPEG o JPEG 2000), instala además
    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

# Valor de aire con el que se reemplaza el relleno fuera del campo de
# reconstrucción (PixelPaddingValue). Es el mínimo habitual de la escala HU.
HU_AIRE = -1024


# ---------------------------------------------------------------------------
# Lectura e inventario
# ---------------------------------------------------------------------------
def _leer_encabezado(ruta: Path):
    """Lee solo el encabezado DICOM. Devuelve None si el archivo no es DICOM."""
    try:
        return pydicom.dcmread(str(ruta), stop_before_pixels=True)
    except InvalidDicomError:
        # Algunos exportadores omiten el preámbulo de 128 bytes.
        try:
            ds = pydicom.dcmread(str(ruta), stop_before_pixels=True, force=True)
        except Exception:
            return None
        return ds if getattr(ds, "SOPClassUID", None) else None
    except Exception:
        return None


def _es_corte_ct(ds) -> bool:
    """Corte axial de TC utilizable: excluye localizadores (scout) y multi-frame."""
    if str(getattr(ds, "Modality", "")).upper() != "CT":
        return False
    tipo = [str(t).upper() for t in (getattr(ds, "ImageType", None) or [])]
    if "LOCALIZER" in tipo:
        return False
    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        return False
    return (getattr(ds, "ImagePositionPatient", None) is not None
            and getattr(ds, "ImageOrientationPatient", None) is not None)


def _inventariar(carpeta: Path):
    """Clasifica, de forma recursiva, los archivos DICOM de la carpeta."""
    cortes, rtstruct, otros = [], [], {}
    for ruta in sorted(p for p in carpeta.rglob("*") if p.is_file()):
        ds = _leer_encabezado(ruta)
        if ds is None:
            continue
        modalidad = str(getattr(ds, "Modality", "")).upper() or "DESCONOCIDA"
        if _es_corte_ct(ds):
            cortes.append((ruta, ds))
        elif modalidad == "RTSTRUCT":
            rtstruct.append(ruta)
        else:
            if modalidad == "CT":
                modalidad = "CT_no_utilizable (localizador o multi-frame)"
            otros[modalidad] = otros.get(modalidad, 0) + 1
    return cortes, rtstruct, otros


def _seleccionar_serie(cortes):
    """Si hay varias series de TC, conserva la de mayor número de cortes."""
    por_serie = {}
    for ruta, ds in cortes:
        por_serie.setdefault(str(ds.SeriesInstanceUID), []).append((ruta, ds))
    if len(por_serie) > 1:
        uid, elegida = max(por_serie.items(), key=lambda kv: len(kv[1]))
        warnings.warn(
            f"Se encontraron {len(por_serie)} series de TC; se usa {uid} "
            f"({len(elegida)} cortes). Revisa si corresponde a la serie de "
            "planificación referenciada por el RTSTRUCT.")
        return elegida
    return next(iter(por_serie.values()))


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def _ordenar_espacialmente(cortes):
    """Ordena por la proyección de ImagePositionPatient sobre la normal al plano.

    Es más confiable que InstanceNumber o que el nombre del archivo, que no
    siempre siguen la posición física del corte.
    """
    iop = np.asarray(cortes[0][1].ImageOrientationPatient, dtype=float)
    for ruta, ds in cortes[1:]:
        if not np.allclose(np.asarray(ds.ImageOrientationPatient, float), iop, atol=1e-4):
            raise ValueError(f"Orientación distinta entre cortes (p. ej. {ruta.name}).")
    normal = np.cross(iop[:3], iop[3:])
    pos = np.array([np.dot(np.asarray(ds.ImagePositionPatient, float), normal)
                    for _, ds in cortes])
    orden = np.argsort(pos, kind="stable")
    return [cortes[i] for i in orden], pos[orden], iop, normal


def _espaciado_cortes(pos: np.ndarray):
    if pos.size < 2:
        return None
    d = np.diff(pos)
    if np.any(np.abs(d) < 1e-4):
        warnings.warn("Hay cortes con la misma posición: la serie podría tener duplicados.")
    if d.max() - d.min() > 1e-2:
        warnings.warn(f"Espaciado entre cortes no uniforme ({d.min():.3f} a {d.max():.3f} mm).")
    return float(np.median(d))


# ---------------------------------------------------------------------------
# Conversión a HU
# ---------------------------------------------------------------------------
def _valor_padding(ds):
    """PixelPaddingValue ajustado al signo de los píxeles (o None)."""
    pad = getattr(ds, "PixelPaddingValue", None)
    if pad is None:
        return None
    pad = int(pad)
    # Con VR implícito, el valor puede leerse sin signo aunque los píxeles
    # sean con signo (por ejemplo, -2000 leído como 63536).
    if int(getattr(ds, "PixelRepresentation", 0)) == 1 and pad > 32767:
        pad -= 65536
    return pad


def _corte_a_hu(ruta: Path, reemplazar_padding: bool):
    ds = pydicom.dcmread(str(ruta))
    try:
        crudo = ds.pixel_array
    except Exception as exc:
        raise RuntimeError(
            f"No se pudieron decodificar los píxeles de {ruta.name} "
            f"(Transfer Syntax {getattr(ds.file_meta, 'TransferSyntaxUID', '?')}). "
            "Si están comprimidos, instala pylibjpeg, pylibjpeg-libjpeg y "
            "pylibjpeg-openjpeg.") from exc

    pendiente = getattr(ds, "RescaleSlope", None)
    intercepto = getattr(ds, "RescaleIntercept", None)
    pendiente = 1.0 if pendiente is None else float(pendiente)
    intercepto = 0.0 if intercepto is None else float(intercepto)

    hu = crudo.astype(np.float64) * pendiente + intercepto

    n_padding = 0
    pad = _valor_padding(ds)
    if reemplazar_padding and pad is not None:
        mascara = crudo.astype(np.int64) == pad
        n_padding = int(mascara.sum())
        hu[mascara] = HU_AIRE
    return hu.astype(np.float32), (pendiente, intercepto), pad, n_padding


def _tipo_sin_perdida(hu: np.ndarray) -> np.ndarray:
    """Usa int16 si todos los valores son enteros dentro de su rango."""
    if (np.all(np.isfinite(hu)) and np.array_equal(hu, np.round(hu))
            and hu.min() >= -32768 and hu.max() <= 32767):
        return hu.astype(np.int16)
    return hu.astype(np.float32)


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------
def extraer_hu(carpeta, reemplazar_padding: bool = True):
    """Devuelve (hu, meta) para la serie de TC contenida en `carpeta`.

    hu:   np.ndarray [z, fila, columna], z ascendente a lo largo de la normal
          al plano (en cortes axiales, de inferior a superior del paciente).
    meta: dict con geometría, UIDs y archivos de origen.
    """
    carpeta = Path(carpeta)
    cortes, rtstruct, otros = _inventariar(carpeta)
    if not cortes:
        raise FileNotFoundError(f"No se encontraron cortes de TC en {carpeta}.")

    cortes = _seleccionar_serie(cortes)
    cortes, pos, iop, normal = _ordenar_espacialmente(cortes)

    filas, columnas = int(cortes[0][1].Rows), int(cortes[0][1].Columns)
    if any((int(ds.Rows), int(ds.Columns)) != (filas, columnas) for _, ds in cortes):
        raise ValueError("Los cortes de la serie no tienen todos el mismo tamaño.")

    volumen = np.empty((len(cortes), filas, columnas), dtype=np.float32)
    rescales, pads, n_padding = set(), set(), 0
    for k, (ruta, _) in enumerate(cortes):
        volumen[k], rescale, pad, n = _corte_a_hu(ruta, reemplazar_padding)
        rescales.add(rescale)
        pads.add(pad)
        n_padding += n
    hu = _tipo_sin_perdida(volumen)
    del volumen

    ds0 = cortes[0][1]
    if str(getattr(ds0, "PatientName", "") or "").strip():
        warnings.warn("El campo PatientName no está vacío: verifica que los datos "
                      "estén anonimizados (el nombre no se copia a la salida).")

    def _num(valor):
        return None if valor is None else float(valor)

    meta = {
        "carpeta_origen": carpeta.name,
        "forma_zyx": list(hu.shape),
        "dtype": str(hu.dtype),
        "orden_ejes": "z (ascendente sobre la normal al plano), fila, columna",
        "pixel_spacing_mm": [float(v) for v in ds0.PixelSpacing],  # [fila, columna]
        "espaciado_entre_cortes_mm": _espaciado_cortes(pos),
        "slice_thickness_mm": _num(getattr(ds0, "SliceThickness", None)),
        "image_orientation_patient": iop.tolist(),
        "normal_al_plano": normal.tolist(),
        "posicion_paciente": str(getattr(ds0, "PatientPosition", "") or ""),
        "image_position_patient": [[float(v) for v in ds.ImagePositionPatient]
                                   for _, ds in cortes],
        "posiciones_sobre_normal_mm": pos.tolist(),
        "study_instance_uid": str(getattr(ds0, "StudyInstanceUID", "")),
        "series_instance_uid": str(ds0.SeriesInstanceUID),
        "frame_of_reference_uid": str(getattr(ds0, "FrameOfReferenceUID", "")),
        "sop_class_uid": str(ds0.SOPClassUID),
        "sop_instance_uids": [str(ds.SOPInstanceUID) for _, ds in cortes],
        "rescale_slope_intercept": sorted(list(r) for r in rescales),
        "pixel_padding_value": sorted(p for p in pads if p is not None) or None,
        "pixeles_padding_reemplazados": n_padding if reemplazar_padding else 0,
        "hu_min": float(hu.min()),
        "hu_max": float(hu.max()),
        "fabricante": str(getattr(ds0, "Manufacturer", "") or ""),
        "archivos_ct": [str(r.relative_to(carpeta)) for r, _ in cortes],
        "archivos_rtstruct": [str(r.relative_to(carpeta)) for r in rtstruct],
        "otras_modalidades": otros,
        "alerta_anonimizacion": bool(str(getattr(ds0, "PatientName", "") or "").strip()),
    }
    return hu, meta


def guardar(hu: np.ndarray, meta: dict, salida, nombre: str) -> Path:
    """Guarda <nombre>.npz y <nombre>_meta.json de forma atómica."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    final = salida / f"{nombre}.npz"
    temporal = salida / f".{nombre}.npz.tmp"
    with open(temporal, "wb") as f:
        np.savez_compressed(f, hu=hu)
    (salida / f"{nombre}_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(final)  # el .npz aparece solo si todo terminó bien
    return final


def cargar_hu(ruta_npz):
    """Carga un volumen guardado por este script: devuelve (hu, meta)."""
    ruta = Path(ruta_npz)
    with np.load(ruta) as datos:
        hu = datos["hu"]
    meta = json.loads(ruta.with_name(f"{ruta.stem}_meta.json").read_text(encoding="utf-8"))
    return hu, meta


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------
def _procesar(carpeta: Path, salida: Path, args) -> str:
    destino = salida / f"{carpeta.name}.npz"
    if destino.exists() and not args.sobrescribir:
        return "omitido (ya extraído)"
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        hu, meta = extraer_hu(carpeta, reemplazar_padding=(args.padding == "aire"))
        meta["avisos"] = [str(a.message) for a in avisos]
    guardar(hu, meta, salida, carpeta.name)
    for a in meta["avisos"]:
        print(f"    aviso: {a}")
    return (f"{hu.shape[0]} cortes, {hu.dtype}, HU [{meta['hu_min']:.0f}, "
            f"{meta['hu_max']:.0f}], {len(meta['archivos_rtstruct'])} RTSTRUCT")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Extrae volúmenes de TC en HU desde DICOM.")
    p.add_argument("entrada", type=Path, help="carpeta del paciente, o carpeta de pacientes con --lote")
    p.add_argument("salida", type=Path, help="carpeta donde se guardan .npz y _meta.json")
    p.add_argument("--lote", action="store_true", help="procesa cada subcarpeta como un paciente")
    p.add_argument("--padding", choices=["aire", "conservar"], default="aire",
                   help=f"reemplazar PixelPaddingValue por {HU_AIRE} HU (aire) o conservarlo")
    p.add_argument("--sobrescribir", action="store_true", help="reprocesa pacientes ya extraídos")
    p.add_argument("--min-libre-gb", type=float, default=5.0,
                   help="detiene el lote si el disco de salida baja de este espacio libre")
    args = p.parse_args(argv)

    if not args.entrada.is_dir():
        print(f"No existe la carpeta {args.entrada}", file=sys.stderr)
        return 1
    args.salida.mkdir(parents=True, exist_ok=True)

    pacientes = (sorted(d for d in args.entrada.iterdir() if d.is_dir())
                 if args.lote else [args.entrada])
    errores = {}
    for i, carpeta in enumerate(pacientes, 1):
        libre_gb = shutil.disk_usage(args.salida).free / 1e9
        if libre_gb < args.min_libre_gb:
            print(f"Detenido: quedan {libre_gb:.1f} GB libres (mínimo {args.min_libre_gb} GB).")
            break
        print(f"[{i}/{len(pacientes)}] {carpeta.name}")
        try:
            print(f"    {_procesar(carpeta, args.salida, args)}")
        except Exception as exc:  # un paciente con problemas no detiene el lote
            errores[carpeta.name] = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR: {errores[carpeta.name]}")
            if not args.lote:
                traceback.print_exc()

    if errores:
        (args.salida / "errores.json").write_text(
            json.dumps(errores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{len(errores)} paciente(s) con error; detalle en errores.json")
    return 1 if errores and not args.lote else 0


if __name__ == "__main__":
    sys.exit(main())
