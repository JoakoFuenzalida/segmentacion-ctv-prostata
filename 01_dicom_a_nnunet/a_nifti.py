"""Convierte los volumenes .npz + mascaras a NIfTI en la estructura que espera nnU-Net v2.

La geometria se reconstruye desde el _meta.json: DICOM trabaja en el sistema LPS
y NIfTI en RAS, de modo que se niegan los dos primeros ejes. Imagen y etiqueta
comparten exactamente el mismo affine, que es lo que nnU-Net exige para emparejarlas.
"""
import json, sys, os
from pathlib import Path
import numpy as np, nibabel as nib

def affine_desde_meta(m):
    ps = m['pixel_spacing_mm']            # [dy_fila, dx_columna]
    dz = float(m['espaciado_entre_cortes_mm'])
    iop = np.asarray(m['image_orientation_patient'], float)
    fila_dir, col_dir = iop[0:3], iop[3:6]   # DICOM: primeros 3 = direccion de la columna
    normal = np.cross(fila_dir, col_dir)
    origen = np.asarray(m['image_position_patient'][0], float)
    M = np.eye(4)
    M[:3, 0] = fila_dir * ps[1]     # eje i del array = columna
    M[:3, 1] = col_dir  * ps[0]     # eje j del array = fila
    M[:3, 2] = normal   * dz        # eje k del array = corte
    M[:3, 3] = origen
    M[0, :] *= -1; M[1, :] *= -1    # LPS -> RAS
    return M

def convertir(pac, carpeta_npz, destino_img, destino_lbl, nombre):
    m = json.loads((carpeta_npz/f'{pac}_meta.json').read_text(encoding='utf-8'))
    hu = np.load(carpeta_npz/f'{pac}.npz')['hu']                 # [z, fila, col]
    with np.load(carpeta_npz/f'{pac}_mascaras.npz') as d:
        claves = list(d.files)
        if not claves: raise RuntimeError(f'{pac}: sin mascara')
        msk = d[claves[0]]
    if hu.shape != msk.shape: raise RuntimeError(f'{pac}: formas distintas')
    aff = affine_desde_meta(m)
    # [z, fila, col] -> [col, fila, z] para que coincida con el affine
    img = np.ascontiguousarray(np.transpose(hu,  (2,1,0)).astype(np.int16))
    lbl = np.ascontiguousarray(np.transpose(msk, (2,1,0)).astype(np.uint8))
    ni = nib.Nifti1Image(img, aff); ni.header.set_xyzt_units('mm')
    nl = nib.Nifti1Image(lbl, aff); nl.header.set_xyzt_units('mm')
    nib.save(ni, destino_img/f'{nombre}_0000.nii.gz')
    nib.save(nl, destino_lbl/f'{nombre}.nii.gz')
    # verificacion: volumen en cm3 calculado desde el NIfTI
    vox = abs(np.linalg.det(aff[:3,:3]))
    return {'clave_mascara':claves[0], 'cortes_con_organo':int((msk.any(axis=(1,2))).sum()),
            'volumen_cm3':round(float(lbl.sum())*vox/1000, 1), 'forma':list(img.shape),
            'voxel_mm3':round(vox,4)}

if __name__ == '__main__':
    pac, npz, img, lbl, nombre = sys.argv[1:6]
    r = convertir(pac, Path(npz), Path(img), Path(lbl), nombre)
    print(json.dumps(r, ensure_ascii=False))
