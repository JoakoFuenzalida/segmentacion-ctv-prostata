# Segmentación automática del CTV prostático en TC con nnU-Net

Código del seminario de título de Ingeniería Civil Informática (PUCV, 2026) de **Joaquín Fuenzalida** y **Pedro Nordenflycht**, con la guía del profesor **Guillermo Cabrera** y el profesor correferente **Mauricio Moyano**.

El objetivo del proyecto es segmentar de forma automática el volumen blanco clínico (CTV) prostático en tomografías computarizadas de planificación de radioterapia y, más adelante, devolver esa segmentación al formato DICOM (RTSTRUCT) para que pueda leerla un sistema de planificación de tratamiento. Este repositorio reúne lo hecho hasta ahora: la extracción de los datos desde DICOM, la conversión al formato de nnU-Net, una línea base con nnU-Net 2D entrenada desde cero y un primer experimento sobre el efecto del aumento de datos.

> **Importante:** este repositorio **no contiene datos de pacientes**. Las imágenes y contornos provienen del Instituto Oncológico Viña del Mar y su acceso se solicita al profesor guía.

## Estado actual

| Etapa | Estado |
|---|---|
| Lectura de DICOM en unidades Hounsfield y rasterización del RTSTRUCT | Lista |
| Conversión a NIfTI (formato nnU-Net) | Lista, 40 pacientes |
| Línea base nnU-Net 2D, fold 0, 250 épocas | Lista |
| Experimento sin aumento de datos | Listo |
| Configuración 3D y validación cruzada completa | Pendiente |
| Reescritura de la predicción en RTSTRUCT | Pendiente |

## Estructura

```
01_dicom_a_nnunet/      Extracción desde DICOM y conversión a NIfTI
    extraer_hu.py           CT en unidades Hounsfield + geometría (.npz y _meta.json)
    generar_mascaras.py     Contornos del RTSTRUCT -> máscaras binarias (regla par-impar)
    medir_ctv.py            Mide volumen y extensión del CTV prostático de cada paciente (selección)
    a_nifti.py              .npz -> NIfTI con la afín correcta (LPS -> RAS) en formato nnU-Net
    aumentar_datos.py       Aumento fuera de línea (rotación, escala, desplazamiento, sin volteo).
                            Se desarrolló y validó, pero en los experimentos se usó el aumento en línea de nnU-Net
    ver_hu.py, ver_imagen.py  Visualización rápida de cortes y máscaras
02_nnunet/
    dataset.json            Descripción del Dataset001_ProstataCTV
    trainers/               nnUNetTrainerNoDA_250epochs (entrenador propio del experimento 2)
    windows/                Lanzadores .bat, envoltorio que evita la suspensión de Windows,
                            chequeo previo de memoria/disco y post-proceso del experimento 2
03_evaluacion/
    metricas_seg.py         Dice, IoU, HD95 (mm) y volúmenes por paciente -> CSV
    comparar_baseline_vs_noDA.py, vigilante_resumen.py   Figuras e informes
04_monitoreo/           Monitores en vivo del entrenamiento (PowerShell)
05_historico_matlab/    Primera exploración del DICOM en MATLAB (referencia)
resultados/             CSV de métricas y figuras de la comparación (sin imágenes de pacientes)
```

## Requisitos

- Python 3.11 en un entorno aislado (conda o venv).
- GPU NVIDIA con CUDA. Los experimentos se hicieron en una RTX 4060 de 8 GB, con 16 GB de RAM y Windows 11.
- PyTorch con CUDA (instalar primero desde [pytorch.org](https://pytorch.org)) y luego:

```bash
pip install -r requirements.txt
```

## Uso paso a paso

### 1. De DICOM a NIfTI

```bash
# Volumen en HU y geometría de cada paciente
python 01_dicom_a_nnunet/extraer_hu.py <carpeta_datos>/ <salida>/ --lote
# Máscaras de todas las estructuras del RTSTRUCT
python 01_dicom_a_nnunet/generar_mascaras.py <carpeta_datos>/ <salida>/ --lote
# Un paciente a NIfTI con la convención de nombres de nnU-Net
python 01_dicom_a_nnunet/a_nifti.py <paciente> <salida> <imagesTr> <labelsTr> PROS_001
```

Se aceptan como CTV prostático las estructuras `CTV` y `CTVp` (con sus variantes de escritura) y se excluyen las ganglionares (`CTVgg`, `CTVgg+`). Criterios de selección: volumen entre 20 y 130 cm³, extensión craneocaudal entre 24 y 70 mm y al menos 10 cortes con contorno. Partición: 32 pacientes para entrenamiento y validación y 8 para prueba, con división por paciente.

### 2. nnU-Net

```bash
set nnUNet_raw=...\nnUNet_raw
set nnUNet_preprocessed=...\nnUNet_preprocessed
set nnUNet_results=...\nnUNet_results
set nnUNet_n_proc_DA=3      # con más procesos Windows falla (WinError 1455)
set nnUNet_compile=f        # torch.compile no funciona en Windows

nnUNetv2_extract_fingerprint -d 1 -np 1
nnUNetv2_plan_experiment -d 1
nnUNetv2_preprocess -d 1 -c 2d -np 2

# Experimento 1: línea base (con el aumento de datos de nnU-Net)
nnUNetv2_train 1 2d 0 -tr nnUNetTrainer_250epochs
# Experimento 2: sin aumento de datos (copiar antes 02_nnunet/trainers/ al paquete)
nnUNetv2_train 1 2d 0 -tr nnUNetTrainerNoDA_250epochs

# Predicción sobre el conjunto de prueba
nnUNetv2_predict -i <imagesTs> -o <salida> -d 1 -c 2d -f 0 -tr nnUNetTrainer_250epochs
```

Los `.bat` de `02_nnunet/windows/` hacen lo mismo y además evitan que el equipo se suspenda durante la noche. Tienen rutas de la máquina en que se ejecutaron, que deben ajustarse.

### 3. Métricas

```bash
python 03_evaluacion/metricas_seg.py <predicciones> <labelsTs> resultados.csv
```

El HD95 se calcula como el máximo entre el percentil 95 de las distancias de superficie en cada dirección, en milímetros y con el espaciado real del vóxel.

## Resultados preliminares (nnU-Net 2D, fold 0, 250 épocas)

| Conjunto | Modelo | Dice (media ± DE) | HD95 en mm (media ± DE) |
|---|---|---|---|
| Validación (7 pacientes) | Con aumento de datos | 0,776 ± 0,251 | 9,2 ± 10,1 |
| Validación (7 pacientes) | Sin aumento de datos | 0,752 ± 0,275 | 10,2 ± 8,7 |
| Prueba (8 pacientes) | Con aumento de datos | 0,817 ± 0,088 | 7,8 ± 3,9 |
| Prueba (8 pacientes) | Sin aumento de datos | 0,806 ± 0,078 | 8,3 ± 4,8 |

Las diferencias entre ambos modelos no son estadísticamente significativas (prueba de Wilcoxon pareada, n = 6 a 8). El efecto más claro del aumento de datos es un menor sobreajuste. En validación, un paciente (PROS_016) concentra la mayor parte del error. Su CTV parece incluir el segmento proximal de las vesículas seminales, lo que todavía debe confirmarse con el equipo clínico. El detalle está en el informe del seminario.

## Limitaciones

Un solo fold y una sola semilla, pocos pacientes de validación y de prueba, 250 épocas en lugar de las 1000 por defecto de nnU-Net y solo la configuración 2D. Además, la comparación del experimento 2 incluye el efecto de desactivar el espejado en la inferencia.

## Uso de herramientas de IA

Parte del código y de la automatización se desarrolló con apoyo de Claude (Anthropic). Las decisiones de diseño y la verificación de los resultados son de los autores.
