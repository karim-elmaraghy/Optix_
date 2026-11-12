# Optix worker process handling. Timeouts kill ONLY the verified worker process
# tree (root identity checked by PID + start time, then native taskkill /T).
# Unrelated processes are never touched.

function Protect-OptixArg {
    param([string]$Text)
    if ($Text -match '[\s"]') { return ('"' + ($Text -replace '"', '\"') + '"') }
    return $Text
}

function Start-OptixWorker {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$WorkerArgs,
        [Parameter(Mandatory = $true)][string]$Cwd,
        [Parameter(Mandatory = $true)][string]$StdoutFile,
        [Parameter(Mandatory = $true)][string]$StderrFile
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Join-Path $env:SystemRoot 'System32\cmd.exe')
    $inner = @('"' + $Exe + '"') + ($WorkerArgs | ForEach-Object { Protect-OptixArg $_ })
    $psi.Arguments = '/S /C "' + ($inner -join ' ') + ' > "' + $StdoutFile + '" 2> "' + $StderrFile + '"' + '"'
    $psi.WorkingDirectory = $Cwd
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    if (-not $p.Start()) { throw 'failed to start worker process' }
    return $p
}

function Invoke-OptixShellCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$Cwd,
        [Parameter(Mandatory = $true)][int]$TimeoutSec,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [Parameter(Mandatory = $true)][string]$ErrFile
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Join-Path $env:SystemRoot 'System32\cmd.exe')
    $psi.Arguments = '/S /C "' + $Command + ' > "' + $OutFile + '" 2> "' + $ErrFile + '"' + '"'
    $psi.WorkingDirectory = $Cwd
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    if (-not $p.Start()) { throw 'failed to start shell command' }
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        Stop-OptixProcessTree -TargetPid $p.Id -Started $p.StartTime
        return @{ TimedOut = $true; ExitCode = -1 }
    }
    return @{ TimedOut = $false; ExitCode = $p.ExitCode }
}

function Stop-OptixProcessTree {
    param(
        [Parameter(Mandatory = $true)][int]$TargetPid,
        [Parameter(Mandatory = $true)][datetime]$Started,
        [switch]$WhatIf
    )
    $root = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if (-not $root) { return 'already-gone' }
    if ([Math]::Abs(($root.StartTime - $Started).TotalSeconds) -gt 5) {
        throw ("refusing to kill PID {0}: start-time mismatch, possible PID reuse" -f $TargetPid)
    }
    if ($WhatIf) {
        Write-Output ("would run: taskkill /PID {0} /T /F ({1})" -f $TargetPid, $root.ProcessName)
        return 'whatif'
    }
    $out = & taskkill /PID $TargetPid /T /F 2>&1
    if ($LASTEXITCODE -ne 0) { throw ("taskkill failed for PID {0}: {1}" -f $TargetPid, ($out -join "`n")) }
    Start-Sleep -Milliseconds 500
    $still = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
    if ($still -and [Math]::Abs(($still.StartTime - $Started).TotalSeconds) -le 5) {
        throw ("PID {0} survived taskkill" -f $TargetPid)
    }
    return 'stopped'
}
