"""Mide el volumen y la extension de cada ROI de tipo CTV prostatico.
Excluye los CTV ganglionares (CTVgg, CTVgg+), que no son prostata."""
import os, re, sys, json
from pathlib import Path
import numpy as np, pydicom

# Carpeta con una subcarpeta DICOM por paciente. Se puede cambiar con la variable de entorno DATOS_DICOM.
RAIZ = Path(os.environ.get('DATOS_DICOM', 'PACIENTES_COMPLETOS'))
# nombres aceptados como CTV prostatico, y los que se excluyen explicitamente
ACEPTA = re.compile(r'^(ctv|ctvp|ctvprostata|ctvp\b)$|^ctv\s*p$|^ctv\s*prostata$', re.I)
EXCLUYE = re.compile(r'gg|gang|nod|vesic|vs\b', re.I)

def norm(s): return re.sub(r'[^a-z0-9]','',s.lower())
def area(p):
    x,y=p[:,0],p[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(np.roll(x,-1),y))

ini,fin=int(sys.argv[1]),int(sys.argv[2])
out=[]
for d in sorted(x for x in RAIZ.iterdir() if x.is_dir())[ini:fin]:
    rt=[f for f in d.rglob('*') if f.name.startswith('RT_Structure_Set_Storage')]
    ct=[f for f in d.rglob('CT_Image_Storage-*')]
    if not rt or not ct: continue
    try: gr=float(getattr(pydicom.dcmread(str(ct[0]),stop_before_pixels=True,
                 specific_tags=['SliceThickness']),'SliceThickness',3.0) or 3.0)
    except Exception: gr=3.0
    ds=pydicom.dcmread(str(rt[0]))
    nm={int(x.ROINumber):str(x.ROIName) for x in (getattr(ds,'StructureSetROISequence',None) or [])}
    cands=[]
    for c in (getattr(ds,'ROIContourSequence',None) or []):
        nombre=nm.get(int(c.ReferencedROINumber),'?')
        if EXCLUYE.search(nombre): continue
        if not ACEPTA.match(norm(nombre)) and not ACEPTA.match(nombre.strip()): continue
        seq=getattr(c,'ContourSequence',None) or []
        vol=0.0; zz=[]
        for it in seq:
            p=np.asarray(it.ContourData,dtype=float).reshape(-1,3)
            if len(p)<3: continue
            vol+=area(p)*gr; zz.append(p[0,2])
        if not zz: continue
        cands.append({'roi':nombre,'cm3':round(vol/1000,1),'cortes':len({round(q,1) for q in zz}),
                      'ext_z_mm':round(max(zz)-min(zz),1)})
    if cands:
        best=max(cands,key=lambda x:x['cortes'])
        best['paciente']=d.name; best['n_candidatos']=len(cands)
        out.append(best)
        print(f"{d.name:<24}{best['roi']:<16}{best['cm3']:>7.1f} cm3{best['cortes']:>5} cortes{best['ext_z_mm']:>7.0f} mm", flush=True)
    else:
        print(f"{d.name:<24}-- sin CTV prostatico --", flush=True)
json.dump(out, open(f'ctv_{ini}_{fin}.json','w'), indent=1, ensure_ascii=False)
