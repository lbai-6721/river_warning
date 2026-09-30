param(
    [string]$Python = "python",
    [int]$Port = 8877,
    [string]$Config = "configs/monitor.json"
)
$ErrorActionPreference = "Stop"
$riverRepo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $riverRepo
try {
    & $Python -m riverlab.monitor --config $Config --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Monitor exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
