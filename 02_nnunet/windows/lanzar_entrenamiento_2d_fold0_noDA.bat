@echo off
REM Experimento 2: nnU-Net v2 - Dataset001_ProstataCTV - 2d - fold 0 - 250 epocas, SIN aumento de datos
REM (nnUNetTrainerNoDA_250epochs). Mismo plan, mismo split (splits_final.json, fold 0) que la linea base.
REM Al terminar el entrenamiento, post_noDA.py encadena: reanudacion/validacion si hiciera falta,
REM prediccion del test (pred_test_noDA_2d), metricas en validacion y test, e informe comparativo.
REM Para reanudar si se interrumpe: ejecutar este mismo .bat con el argumento --c
set nnUNet_raw=C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_raw
set nnUNet_preprocessed=C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_preprocessed
set nnUNet_results=C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_results
REM 3 workers de aumentacion: con mas, Windows falla por archivo de paginacion insuficiente (WinError 1455)
set nnUNet_n_proc_DA=3
set nnUNet_compile=f
set PYTHONUNBUFFERED=1
REM 3 procesos para exportar la validacion final (en vez de 8): menos riesgo de quedarse sin memoria
set nnUNet_def_n_proc=3
cd /d C:\Users\joaki\Documents\Tesis_segmentacion
REM Chequeo previo de memoria/disco (solo lectura). Si no alcanza, pregunta antes de seguir.
"C:\Users\joaki\anaconda3\envs\nnunet\python.exe" "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_scripts\chequeo_memoria.py"
if errorlevel 1 (
  choice /M "La memoria o el disco probablemente NO alcanzan. Lanzar de todos modos"
  if errorlevel 2 exit /b 1
)
"C:\Users\joaki\anaconda3\envs\nnunet\python.exe" "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_scripts\run_train_keepawake.py" "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_logs\10_train_2d_fold0_noDA.log" 1 2d 0 -tr nnUNetTrainerNoDA_250epochs %*
"C:\Users\joaki\anaconda3\envs\nnunet\python.exe" "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_scripts\post_noDA.py" >> "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_logs\11_post_noDA.log" 2>&1
