$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$outDir = Join-Path $projectRoot "outputs\environment"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$osInfo = Get-CimInstance Win32_OperatingSystem
$machineInfo = Get-CimInstance Win32_ComputerSystem
$info = [ordered]@{
    collectedAt = (Get-Date).ToString("o")
    windows = $osInfo.Caption
    windowsVersion = $osInfo.Version
    systemMemoryGB = [math]::Round($machineInfo.TotalPhysicalMemory / 1GB, 2)
    disks = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object {
        @{ drive = $_.DeviceID; freeGB = [math]::Round($_.FreeSpace / 1GB, 2) }
    })
}
try { $info["nvidiaSmi"] = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | Out-String) }
catch { $info["nvidiaSmi"] = $_.Exception.Message }
try { $info["wsl"] = (& wsl --list --verbose 2>&1 | Out-String) }
catch { $info["wsl"] = $_.Exception.Message }
$path = Join-Path $outDir "windows.json"
$info | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $path
Write-Host "Environment report saved to $path"
