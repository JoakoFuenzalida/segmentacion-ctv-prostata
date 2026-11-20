@echo off
REM Entrenamiento nnU-Net v2 - Dataset001_ProstataCTV - 2d - fold 0 - 250 epocas, desde cero.
REM Para REANUDAR desde el ultimo checkpoint si se interrumpe, ejecutar este .bat con el argumento: --c
set nnUNet_raw=C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_raw
set nnUNet_preprocessed=C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_preprocessed
set nnUNet_results=C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_results
REM 3 workers de aumentacion: con 6 Windows falla por archivo de paginacion insuficiente (WinError 1455)
set nnUNet_n_proc_DA=3
set nnUNet_compile=f
set PYTHONUNBUFFERED=1
"C:\Users\joaki\anaconda3\envs\nnunet\python.exe" "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_scripts\run_train_keepawake.py" "C:\Users\joaki\Documents\Tesis_segmentacion\_nnunet_logs\04_train_2d_fold0.log" 1 2d 0 -tr nnUNetTrainer_250epochs %*
