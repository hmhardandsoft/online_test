param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Start','Stop','Status')][string]$Action,
    [string]$TraceDirectory
)
$ErrorActionPreference = 'Stop'
$traceInstance = 'SSVEPOnlineRender'
$wpr = (Get-Command wpr.exe -ErrorAction Stop).Source
if ($Action -eq 'Status') {
    & $wpr -status -instancename $traceInstance
    exit $LASTEXITCODE
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this capture helper from an Administrator PowerShell. Run the experiment normally in a separate terminal.'
}
if (-not $TraceDirectory) { throw 'Specify -TraceDirectory (use the same directory for Start and Stop).' }
$tracePath = [IO.Path]::GetFullPath($TraceDirectory)
$statePath = Join-Path $tracePath 'capture_state.json'
if ($Action -eq 'Start') {
    if (Test-Path -LiteralPath $tracePath) { throw 'Choose a new trace directory to avoid overwriting an earlier capture.' }
    New-Item -ItemType Directory -Path $tracePath | Out-Null
    # Potentially slow inventory runs BEFORE tracing/stimulation.
    $inventory = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); computer = $env:COMPUTERNAME }
    try { $inventory.video = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,PNPDeviceID) }
    catch { $inventory.video_error = $_.Exception.Message }
    try { $inventory.os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber }
    catch { $inventory.os_error = $_.Exception.Message }
    $inventory.power_scheme = (& powercfg.exe /getactivescheme | Out-String).Trim()
    $inventory | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $tracePath 'system_snapshot.json') -Encoding UTF8
    $profile = (Join-Path $PSScriptRoot 'render_trace.wprp') + '!SSVEPRender'
    # GeneralProfile includes scheduler/CPU stack information; GPU adds graphics events.
    # A named instance avoids stopping/cancelling another user's WPR recording.
    & $wpr -start GeneralProfile -start GPU -start $profile -filemode -instancename $traceInstance
    if ($LASTEXITCODE -ne 0) { throw 'WPR did not start. No existing trace was cancelled. Inspect the WPR error.' }
    @{ instance=$traceInstance; started_utc=[DateTime]::UtcNow.ToString('o'); status='running'; directory=$tracePath } |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Host 'Capture started. Now run the experiment in its normal terminal. Stop the experiment, then run this helper with -Action Stop.'
} else {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.instance -ne $traceInstance -or $state.status -ne 'running' -or $state.directory -ne $tracePath) {
        throw 'This directory does not identify a running SSVEP capture.'
    }
    $etl = Join-Path $tracePath 'render.etl'
    if (Test-Path -LiteralPath $etl) { throw 'render.etl already exists; refusing to overwrite it.' }
    & $wpr -stop $etl -instancename $traceInstance
    if ($LASTEXITCODE -ne 0) { throw 'WPR could not save the trace. The capture state is retained; inspect the error before retrying.' }
    $state.status = 'saved'
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Host "Saved $etl. Keep the corresponding online_results run directory alongside this trace."
}
