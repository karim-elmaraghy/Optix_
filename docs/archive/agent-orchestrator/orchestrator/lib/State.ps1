# Optix local state (gitignored management/.optix). Atomic writes, PID lock file.

function Get-OptixStateDir {
    param([Parameter(Mandatory = $true)][string]$OrchestratorRoot)
    $d = Join-Path (Split-Path $OrchestratorRoot -Parent) '.optix'
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    return $d
}

function Read-OptixState {
    param([Parameter(Mandatory = $true)][string]$StateDir)
    $s = @{ tasks = @{}; daily = @{ date = ''; counts = @{} } }
    $p = Join-Path $StateDir 'state.json'
    if (Test-Path $p) {
        $raw = (Get-Content $p -Raw -ErrorAction Stop)
        if ($raw.Trim() -ne '') {
            $j = ConvertFrom-Json $raw
            if ($j.tasks) {
                foreach ($prop in $j.tasks.PSObject.Properties) { $s.tasks[$prop.Name] = $prop.Value }
            }
            if ($j.daily) {
                $s.daily.date = $j.daily.date
                if ($j.daily.counts) {
                    foreach ($prop in $j.daily.counts.PSObject.Properties) { $s.daily.counts[$prop.Name] = $prop.Value }
                }
            }
        }
    }
    return $s
}

function Write-OptixState {
    param([Parameter(Mandatory = $true)][string]$StateDir, [Parameter(Mandatory = $true)][hashtable]$State)
    if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Path $StateDir -Force | Out-Null }
    $p = Join-Path $StateDir 'state.json'
    $tmp = $p + '.tmp'
    ($State | ConvertTo-Json -Depth 8) | Set-Content $tmp -Encoding UTF8
    Move-Item $tmp $p -Force
}

function Get-OptixRunDir {
    param([Parameter(Mandatory = $true)][string]$StateDir, [Parameter(Mandatory = $true)][int]$Issue)
    $d = Join-Path (Join-Path $StateDir 'runs') ("{0}" -f $Issue)
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    return $d
}

# Returns $null (free), 'stale' (dead owner), or the lock object (held).
function Test-OptixLock {
    param([Parameter(Mandatory = $true)][string]$StateDir)
    $p = Join-Path $StateDir 'dispatcher.lock.json'
    if (-not (Test-Path $p)) { return $null }
    $l = ConvertFrom-Json (Get-Content $p -Raw)
    $proc = Get-Process -Id $l.pid -ErrorAction SilentlyContinue
    if ($proc -and $proc.StartTime.ToString('o') -eq $l.started) { return $l }
    return 'stale'
}

function Set-OptixLock {
    param([Parameter(Mandatory = $true)][string]$StateDir)
    $me = Get-Process -Id $PID
    @{ pid = $PID; started = $me.StartTime.ToString('o') } | ConvertTo-Json | Set-Content (Join-Path $StateDir 'dispatcher.lock.json') -Encoding UTF8
}

function Remove-OptixLock {
    param([Parameter(Mandatory = $true)][string]$StateDir)
    $p = Join-Path $StateDir 'dispatcher.lock.json'
    if (Test-Path $p) { Remove-Item $p -Force }
}
