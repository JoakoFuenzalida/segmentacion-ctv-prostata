# Monitor en vivo del entrenamiento nnU-Net (solo lectura: no toca el entrenamiento).
# Cerrar esta ventana o Ctrl+C NO detiene el entrenamiento.
$fold = "C:\Users\joaki\Documents\Tesis_segmentacion\nnUNet_results\Dataset001_ProstataCTV\nnUNetTrainer_250epochs__nnUNetPlans__2d\fold_0"
$total = 250
while ($true) {
    $logs = Get-ChildItem "$fold\training_log_*.txt" -ErrorAction SilentlyContinue | Sort-Object Name
    $lines = @()
    foreach ($l in $logs) { $lines += Get-Content $l.FullName }
    $epochLines = $lines | Select-String ": Epoch (\d+)\s*$"
    $cur = 0
    if ($epochLines) { $cur = [int]$epochLines[-1].Matches[0].Groups[1].Value }
    $times = @($lines | Select-String "Epoch time: ([\d.]+) s" | ForEach-Object { [double]$_.Matches[0].Groups[1].Value })
    $dices = @($lines | Select-String "Pseudo dice \[np.float32\(([\d.]+)\)" | ForEach-Object { [double]$_.Matches[0].Groups[1].Value })
    $emas  = @($lines | Select-String "New best EMA pseudo Dice: ([\d.]+)" | ForEach-Object { [double]$_.Matches[0].Groups[1].Value })
    $tl = $lines | Select-String "train_loss (-?[\d.]+)" | Select-Object -Last 1
    $vl = $lines | Select-String "val_loss (-?[\d.]+)" | Select-Object -Last 1
    $done = $times.Count
    $recent = $times | Select-Object -Last 10
    $mean = 0
    if ($recent) { $mean = ($recent | Measure-Object -Average).Average }
    $eta = (Get-Date).AddSeconds(($total - $done) * $mean)
    $pct = [math]::Round(100 * $done / $total, 1)
    $w = 50
    $fill = [int][math]::Floor($w * $done / $total)
    $bar = ("#" * $fill) + ("-" * ($w - $fill))
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>$null)
    $free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    Clear-Host
    Write-Host "nnU-Net 2d fold 0 - Dataset001_ProstataCTV        $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[$bar] $pct %"
    Write-Host "Epocas completadas: $done / $total   (entrenando la epoca $cur)"
    if ($mean -gt 0) { Write-Host ("Tiempo medio por epoca: {0:N0} s   Fin estimado: {1:HH:mm} (+ ~10 min validacion)" -f $mean, $eta) }
    Write-Host ""
    if ($tl) { Write-Host ("train_loss: {0}   val_loss: {1}" -f $tl.Matches[0].Groups[1].Value, $vl.Matches[0].Groups[1].Value) }
    if ($dices.Count -gt 0) { Write-Host ("Pseudo-Dice ultima epoca: {0:N4}   Mejor EMA: {1:N4}" -f $dices[-1], ($emas | Measure-Object -Maximum).Maximum) -ForegroundColor Green }
    $spark = ($dices | Select-Object -Last 20 | ForEach-Object { "{0:N2}" -f $_ }) -join " "
    Write-Host "Ultimos pseudo-Dice: $spark"
    Write-Host ""
    Write-Host "GPU (uso, VRAM, temp): $gpu    Disco C: libre: $free GB"
    if (Test-Path "$fold\checkpoint_final.pth") { Write-Host "ENTRENAMIENTO TERMINADO (checkpoint_final.pth). Validacion en curso o lista." -ForegroundColor Yellow }
    if (Test-Path "$fold\validation\summary.json") { Write-Host "Validacion final lista: $fold\validation\summary.json" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "Graficos: $fold\progress.png (nnU-Net lo actualiza cada epoca)" -ForegroundColor DarkGray
    Write-Host "Cerrar esta ventana o Ctrl+C NO detiene el entrenamiento. Se actualiza cada 15 s." -ForegroundColor DarkGray
    Start-Sleep 15
}
